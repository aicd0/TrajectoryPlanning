import numpy as np
import utils.math
import utils.print
from abc import abstractmethod
from envs.simulator import Simulator
from utils.stopwatch import Stopwatch

class Planner:
    def __init__(self, sim: Simulator, plot: bool=False) -> None:
        self.sim = sim
        self.plot = plot
        self.stopwatch = Stopwatch()

    def reach(self, pos: np.ndarray, verbose=False) -> bool:
        self.stopwatch.reset()
        self.stopwatch.start()
        success = self._reach(pos)
        self.stopwatch.pause()
        if verbose:
            span = self.stopwatch.span()
            result = 'completed' if success else 'failed'
            utils.print.put('Task %s. Time used: %fs' % (result, span))

    @abstractmethod
    def _reach(self, pos: np.ndarray) -> bool:
        raise NotImplementedError()

    def _direct_reach(self, joint_position: np.ndarray) -> bool:
        self.stopwatch.pause()
        while True:
            state = self.sim.state()
            action = joint_position - state.joint_position
            if np.sum(np.abs(action)) < 1e-5:
                success = True
                break
            new_state = self.sim.step(action / self.sim.action_amp)
            if self.plot:
                self.sim.plot_step()
            if new_state.collision:
                success = False
                break
            if np.max(np.abs(new_state.joint_position - joint_position)) < 1e-5:
                success = True
                break
            if np.max(np.abs(new_state.joint_position, state.joint_position)) < 1e-5:
                success = False
                break
        self.stopwatch.start()
        return success