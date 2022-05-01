import config
import numpy as np
import utils.math
from .state import GazeboState
from envs.reward import Reward
from framework.configuration import global_configs as configs

class GazeboReward(Reward):
    def __init__(self) -> None:
        super().__init__(configs.get(config.Environment.Gazebo.MaxSteps_))

    def _update(self, action: np.ndarray, state: GazeboState) -> None:
        self.reward = 0
        self.done = False
        if state.collision:
            return
        d = utils.math.distance(state.achieved, state.desired)
        self.reward = GazeboReward.__distance2reward(d)

    @staticmethod
    def __distance2reward(d: float) -> float:
        return 10 / (d * 5 + 1) + (10 if d < 0.05 else 0)