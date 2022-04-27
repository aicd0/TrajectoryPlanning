import numpy as np
import utils.math
from abc import abstractmethod
from envs.simulator import Simulator
from framework.agent import AgentBase

class Planner:
    def __init__(self, sim: Simulator, iks: AgentBase) -> None:
        self.sim = sim
        self.iks = iks
        self.plot = False

    def _reach(self, pos: np.ndarray, max_steps=-1) -> None:
        step = 0
        last_state = self.sim.state()
        while max_steps < 0 or step < max_steps:
            state = self.sim.state()
            if utils.math.distance(state.achieved, pos) < 0.05:
                break
            state.desired = pos
            action = self.iks.sample_action(state, deterministic=True)
            self.sim.step(action)
            if self.plot:
                self.sim.plot_step()
            if np.sum(np.abs(state.achieved - last_state.achieved)) < 1e-5:
                break
            last_state = state
            step += 1

    @abstractmethod
    def reach(self, pos: np.ndarray) -> None:
        raise NotImplementedError()