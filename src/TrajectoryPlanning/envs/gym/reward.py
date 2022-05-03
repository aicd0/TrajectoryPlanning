import config
import numpy as np
from .state import GymState
from envs.reward import Reward
from framework.configuration import global_configs as configs

class GymReward(Reward):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = configs.get(config.Environment.Gym.Environment_)

    def _update(self, action: np.ndarray, state: GymState) -> None:
        reward = self.__state2reward(state)
        return reward, state.raw_done

    def __state2reward(self, state: GymState):
        if self.env_name == 'CartPole-v0':
            return -1 if state.raw_done else 1
        if self.env_name == 'CartPole-v1':
            return -1 if state.raw_done else 1
        if self.env_name == 'FetchReach-v1':
            if state.raw_reward < 0:
                return -np.linalg.norm(state.desired - state.achieved) * 10.
            return 0
        return state.raw_reward