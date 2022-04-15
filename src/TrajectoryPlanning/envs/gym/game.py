import config
import numpy as np
from envs.gym.game_state import GameState
from framework.configuration import global_configs as configs
from typing import Tuple

class Game:
    def __init__(self) -> None:
        pass

    def __state2reward(self, state: GameState):
        env_name = configs.get(config.Environment.Gym.Environment_)
        
        if env_name == 'CartPole-v0':
            return -1 if state.done else 1
        if env_name == 'CartPole-v1':
            return -1 if state.done else 1
        if env_name == 'FetchReach-v1':
            if state.reward_raw < 0:
                return -np.linalg.norm(state.desired - state.achieved) * 10.
            return 0.
        return state.reward_raw

    def reset(self) -> None:
        pass

    def update(self, action: np.ndarray, state: GameState) -> Tuple:
        reward = self.__state2reward(state)
        return reward, state.done