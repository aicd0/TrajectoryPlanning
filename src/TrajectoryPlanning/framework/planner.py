import numpy as np
import utils.math
from abc import abstractmethod
from envs.simulator import Simulator
from framework.agent import AgentBase

class Planner:
    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        self.plot = False

    def _reach(self, joint_position: np.ndarray) -> bool:
        while True:
            state = self.sim.state()
            action = joint_position - state.joint_position
            if np.sum(np.abs(action)) < 1e-5:
                return False
            new_state = self.sim.step(action / self.sim.action_amp)
            if self.plot:
                self.sim.plot_step()
            if new_state.collision:
                return False
            if utils.math.manhattan_distance(new_state.joint_position, joint_position) < 1e-5:
                return True
            if utils.math.manhattan_distance(new_state.joint_position, state.joint_position) < 1e-5:
                return False

    @abstractmethod
    def reach(self, pos: np.ndarray) -> None:
        raise NotImplementedError()