import config
import numpy as np
from simulator.gym.game_state import GameState
from typing import Tuple

class Game:
    def __init__(self) -> None:
        pass

    def __state2reward(self, state: GameState):
        # env_name = config.Simulator.Gym.Environment
        # if env_name == 'CartPole-v0':
        #     return -1 if state.done else 1
        # if env_name == 'CartPole-v1':
        #     return -1 if state.done else 1
        # if env_name == 'FetchReach-v1':
        #     if state.reward_raw < 0:
        #         return -np.linalg.norm(state.desired - state.achieved) * 10.
        #     return 0.
        # return state.reward_raw
        pass

    def reset(self) -> None:
        pass

    def update(self, action: np.ndarray, state: GameState) -> Tuple:
        # reward = self.__state2reward(state)
        # return reward, state.done
        pass