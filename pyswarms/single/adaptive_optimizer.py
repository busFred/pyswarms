# Import standard library
import logging
from typing import Callable, Deque, Dict, Optional, Tuple, Union
from enum import Enum, auto

# Import modules
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import Pool

from collections import deque

from ..backend.operators import compute_evolutionary_factor, compute_pbest, compute_objective_function, compute_particle_mean_distances
from ..backend.topology import Topology
from ..backend.handlers import BoundaryHandler, VelocityHandler
from ..base import SwarmOptimizer
from ..utils.reporter import Reporter


class AdaptiveOptimizerPSO(SwarmOptimizer):

    class EvolutionState(Enum):
        S1_EXPLORATION = 1
        S2_EXPLOITATION = 2
        S3_CONVERGENCE = 3
        S4_JUMPING_OUT = 4

        @staticmethod
        def compute_exploration_numeric(evo_factor: float) -> float:
            if 0.0 <= evo_factor and evo_factor <= 0.4:
                return 0.0
            elif 0.4 < evo_factor and evo_factor <= 0.6:
                return 5.0 * evo_factor - 2.0
            elif 0.6 < evo_factor and evo_factor <= 0.7:
                return 1.0
            elif 0.7 < evo_factor and evo_factor <= 0.8:
                return -10.0 * evo_factor + 8.0
            elif 0.8 < evo_factor and evo_factor <= 1.0:
                return 0.0
            raise ValueError("evo_factor not in range [0.0, 1.0]")

        @staticmethod
        def compute_exploitation_numeric(evo_factor: float) -> float:
            if 0.0 <= evo_factor and evo_factor <= 0.2:
                return 0.0
            elif 0.2 < evo_factor and evo_factor <= 0.3:
                return 10.0 * evo_factor - 2.0
            elif 0.3 < evo_factor and evo_factor <= 0.4:
                return 1.0
            elif 0.4 < evo_factor and evo_factor <= 0.6:
                return -5.0 * evo_factor + 3.0
            elif 0.6 < evo_factor and evo_factor <= 1.0:
                return 0.0
            raise ValueError("evo_factor not in range [0.0, 1.0]")

        @staticmethod
        def compute_convergence_numeric(evo_factor: float) -> float:
            if 0.0 <= evo_factor and evo_factor <= 0.1:
                return 1.0
            elif 0.1 < evo_factor and evo_factor <= 0.3:
                return -5.0 * evo_factor + 1.5
            elif 0.3 < evo_factor and evo_factor <= 1.0:
                return 0.0
            raise ValueError("evo_factor not in range [0.0, 1.0]")

        @staticmethod
        def compute_jump_out_numeric(evo_factor: float) -> float:
            if 0.0 <= evo_factor and evo_factor <= 0.7:
                return 0.0
            elif 0.7 < evo_factor and evo_factor <= 0.9:
                return 5.0 * evo_factor - 3.5
            elif 0.9 < evo_factor and evo_factor <= 1.0:
                return 1.0
            raise ValueError("evo_factor not in range [0.0, 1.0]")

    rep: Reporter
    top: Topology
    bh: BoundaryHandler
    vh: VelocityHandler
    name: str
    state: EvolutionState

    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        options: Dict[str, float],
        topology: Topology,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        bh_strategy: str = "periodic",
        velocity_clamp: Optional[Tuple[float, float]] = None,
        vh_strategy: str = "unmodified",
        center: Union[np.ndarray, float] = 1.00,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
        init_pos: Optional[np.ndarray] = None,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}` or :code:`{'c1',
                'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                if used with the :code:`Ring`, :code:`VonNeumann` or
                :code:`Random` topology the additional parameter k must be
                included
                * k : int
                    number of neighbors to be considered. Must be a positive
                    integer less than :code:`n_particles`
                if used with the :code:`Ring` topology the additional
                parameters k and p must be included
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the sum-of-absolute
                    values (or L1 distance) while 2 is the Euclidean (or L2)
                    distance.
                if used with the :code:`VonNeumann` topology the additional
                parameters p and r must be included
                * r: int
                    the range of the VonNeumann topology.  This is used to
                    determine the number of neighbours in the topology.
        topology : pyswarms.backend.topology.Topology
            a :code:`Topology` object that defines the topology to use in the
            optimization process. The currently available topologies are:
                * Star
                    All particles are connected
                * Ring (static and dynamic)
                    Particles are connected to the k nearest neighbours
                * VonNeumann
                    Particles are connected in a VonNeumann topology
                * Pyramid (static and dynamic)
                    Particles are connected in N-dimensional simplices
                * Random (static and dynamic)
                    Particles are connected to k random particles
                Static variants of the topologies remain with the same
                neighbours over the course of the optimization. Dynamic
                variants calculate new neighbours every time step.
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : numpy.ndarray or float, optional
            controls the mean or center whenever the swarm is generated randomly.
            Default is :code:`1`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        """
        super(AdaptiveOptimizerPSO, self).__init__(
            n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            ftol_iter=ftol_iter,
            init_pos=init_pos,
        )
        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology and check for type
        if not isinstance(topology, Topology):
            raise TypeError("Parameter `topology` must be a Topology object")
        else:
            self.top = topology
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.name = __name__

    def optimize(self,
                 objective_func: Callable,
                 iters: int,
                 n_processes: Optional[int] = None,
                 verbose: bool = True,
                 **kwargs) -> Tuple[float, np.ndarray]:
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """
        # Apply verbosity
        log_level: int
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )

        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool: Union[Pool, None] = None if n_processes is None else mp.Pool(
            n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history: Deque = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = compute_objective_function(self.swarm,
                                                                 objective_func,
                                                                 pool=pool,
                                                                 **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm)
            best_cost_yet_found = self.swarm.best_cost
            # fmt: on
            # Update swarm
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, **self.options)
            # Print to console
            if verbose:
                self.rep.hook(best_cost=self.swarm.best_cost)
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            self.__perform_ese()
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (np.abs(self.swarm.best_cost - best_cost_yet_found) <
                     relative_measure)
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break

            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds)
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh)
        # Obtain the final best_cost and the final best_position
        final_best_cost: float = self.swarm.best_cost.copy()
        final_best_pos: np.ndarray = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
        return (final_best_cost, final_best_pos)

    def reset(self):
        EvolutionState = AdaptiveOptimizerPSO.EvolutionState
        self.reset()
        self.state = EvolutionState.S1_EXPLORATION

    def __perform_ese(self):
        mean_distances: np.ndarray = compute_particle_mean_distances(self.swarm)
        evo_factor: float = compute_evolutionary_factor(
            swarm=self.swarm, mean_distances=mean_distances)
        state_numeric: np.ndarray = self.__compute_state_numeric(
            evo_factor=evo_factor)

    def __compute_state_numeric(self, evo_factor: float) -> np.ndarray:
        """Compute evolutionary state numeric.

        Parameters
        ----------
        evo_factor : float
            the current evolutionary factor

        Returns
        -------
        np.ndarray
            (4,) in the order of [s1_num, s2_num, s3_num, s4_num].
        """
        EvolutionState = AdaptiveOptimizerPSO.EvolutionState
        state_num: np.ndarray = np.full(shape=(4), fill_value=0.0)
        state_num[0] = EvolutionState.compute_exploration_numeric(
            evo_factor=evo_factor)
        state_num[1] = EvolutionState.compute_exploitation_numeric(
            evo_factor=evo_factor)
        state_num[2] = EvolutionState.compute_convergence_numeric(
            evo_factor=evo_factor)
        state_num[3] = EvolutionState.compute_jump_out_numeric(
            evo_factor=evo_factor)
        return state_num
