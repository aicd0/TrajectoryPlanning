import numpy as np
import utils.math
import utils.print
from abc import abstractmethod
from envs.simulator import Simulator
from typing import Generator
from utils.stopwatch import Stopwatch

class Planner:
    def __init__(self, sim: Simulator, plot: bool=False) -> None:
        self.sim = sim
        self.plot = plot
        self.stopwatch = Stopwatch()

    def reach(self, position: np.ndarray, verbose=False) -> bool:
        track = self._plan(position)
        success = True
        steps = 0
        d_window = []

        self.stopwatch.reset()
        self.stopwatch.start()
        for joint_position in track:
            self.stopwatch.pause()

            # Exceeds max steps.
            steps += 1
            if steps > 150:
                success = False
                break

            # Planner failure.
            if joint_position is None:
                success = False
                break
            
            # Simulator failure.
            if not self.__simple_reach(joint_position):
                success = False
                break
            
            # Distance forzon.
            state = self.sim.state()
            d = utils.math.distance(state.achieved, position)
            d_window.append(d)
            window_size = 30
            if len(d_window) >= window_size:
                d_window = d_window[-window_size:]
                if max(d_window) - min(d_window) < 0.15:
                    success = False
                    break

            self.stopwatch.start()
        self.stopwatch.pause()
        
        if success:
            state = self.sim.state()
            d = utils.math.distance(state.achieved, position)
            if d > 0.5:
                success = False
        if verbose:
            span = self.stopwatch.span()
            result = 'completed' if success else 'failed'
            error = utils.math.distance(self.sim.state().achieved, position)
            utils.print.put('Task %s. Steps=%d. Error=%f. Time used: %fs'
                % (result, steps, error, span))
        return success

    @abstractmethod
    def _plan(self, position: np.ndarray) -> Generator[np.ndarray | None, None, None]:
        raise NotImplementedError()

    def __simple_act(self, action: np.ndarray) -> bool:
        while True:
            state = self.sim.step(action / self.sim.action_amp)
            if self.plot:
                self.sim.plot_step()
            if state.collision:
                success = False
                break
            success = True
            break
        return success

    def __simple_reach(self, joint_position: np.ndarray) -> bool:
        while True:
            state = self.sim.state()
            action = joint_position - state.joint_position
            if np.max(np.abs(action)) < 1e-5:
                success = True
                break
            if not self.__simple_act(action):
                success = False
                break
            new_state = self.sim.state()
            if np.max(np.abs(new_state.joint_position - state.joint_position)) < 1e-5:
                success = False
                break
        return success