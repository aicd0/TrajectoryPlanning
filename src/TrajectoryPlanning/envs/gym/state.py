import config
import numpy as np
from envs.state import State
from framework.configuration import global_configs as configs

class GymState(State):
    def __init__(self):
        super().__init__()
        self.achieved = None
        self.desired = None
        self.states = None
        self.reward_raw = None
        self.done = None

    def __from_raw_state(self, state_raw) -> None:
        env_name = configs.get(config.Environment.Gym.Environment_)
        if env_name == 'FetchReach-v1':
            self.achieved = state_raw['achieved_goal']
            self.desired = state_raw['desired_goal']
            self.states = np.concatenate((
                state_raw['observation'],
                self.desired,
            ), dtype=config.Common.DataType.Numpy)
        else:
            self.states = state_raw

    def from_reset(self, state_raw) -> None:
        self.__from_raw_state(state_raw)

    def from_step(self, state_raw, reward_raw, done: bool) -> None:
        self.__from_raw_state(state_raw)
        self.reward_raw = reward_raw
        self.done = done
        
    def support_her(self) -> bool:
        env_name = configs.get(config.Environment.Gym.Environment_)
        if env_name == 'FetchReach-v1':
            return True
        return False

    def _as_input(self) -> np.ndarray:
        return self.states