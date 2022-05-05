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
            error = utils.math.distance(self.sim.state().achieved, pos)
            utils.print.put('Task %s. Error=%f. Time used: %fs' % (result, error, span))
        return success

    @abstractmethod
    def _reach(self, position: np.ndarray) -> bool:
        raise NotImplementedError()

    def _simple_act(self, action: np.ndarray, preamp=True) -> bool:
        self.stopwatch.pause()
        while True:
            if preamp:
                action /= self.sim.action_amp
            state = self.sim.step(action)
            if self.plot:
                self.sim.plot_step()
            if state.collision:
                success = False
                break
            success = True
            break
        self.stopwatch.start()
        return success

    def _simple_reach(self, joint_position: np.ndarray) -> bool:
        self.stopwatch.pause()
        while True:
            state = self.sim.state()
            action = joint_position - state.joint_position
            if np.max(np.abs(action)) < 1e-5:
                success = True
                break
            if not self._simple_act(action):
                success = False
                break
            new_state = self.sim.state()
            if np.max(np.abs(new_state.joint_position - state.joint_position)) < 1e-5:
                success = False
                break
        self.stopwatch.start()
        return success