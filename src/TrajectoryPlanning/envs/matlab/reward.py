import numpy as np
from .state import MatlabState
from envs.reward import Reward

class MatlabReward(Reward):
    def __init__(self) -> None:
        super().__init__(100)

    def _update(self, action: np.ndarray, state: MatlabState) -> None:
        self.reward = 0
        self.done = False
        # self.done = any([
        #     state.self_collision,
        #     state.world_collision,
        #     state.deadlock,
        # ])
        if state.collision:
            return
        d = np.linalg.norm(state.achieved - state.desired)
        self.reward = MatlabReward.__distance2reward(d)

    @staticmethod
    def __distance2reward(self, d: float) -> float:
        return 10 / (d * 5 + 1)