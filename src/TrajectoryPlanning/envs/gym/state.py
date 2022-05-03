import config
import numpy as np
from envs.state import State
from framework.configuration import global_configs as configs

class GymState(State):
    def __init__(self, raw_state, raw_reward, raw_done):
        super().__init__()
        self.__from_raw_state(raw_state)
        self.raw_reward = raw_reward
        self.raw_done = raw_done

    def __from_raw_state(self, raw_state) -> None:
        env_name = configs.get(config.Environment.Gym.Environment_)
        if env_name == 'FetchReach-v1':
            self.achieved = raw_state['achieved_goal']
            self.desired = raw_state['desired_goal']
            self.state = np.concatenate((
                raw_state['observation'],
                self.desired,
            ), dtype=config.Common.DataType.Numpy)
        else:
            self.states = raw_state
        
    def support_her(self) -> bool:
        env_name = configs.get(config.Environment.Gym.Environment_)
        if env_name == 'FetchReach-v1':
            return True
        return False

    def _as_input(self) -> np.ndarray:
        return self.state