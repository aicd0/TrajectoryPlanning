import numpy as np
from .state import MatlabState
from envs.reward import Reward

class MatlabReward(Reward):
    def __init__(self) -> None:
        super().__init__(100)

    def _update(self, state: MatlabState, action: np.ndarray, next_state: MatlabState) -> None:
        self.reward = 0
        self.done = False
        # self.done = any([
        #     next_state.self_collision,
        #     next_state.world_collision,
        #     next_state.deadlock,
        # ])
        if next_state.collision:
            return
        d = np.linalg.norm(next_state.achieved - next_state.desired)
        self.reward = MatlabReward.__distance2reward(d)

    @staticmethod
    def __distance2reward(self, d: float) -> float:
        return 10 / (d * 5 + 1)