import config
import numpy as np

class GameState:
    def __init__(self):
        pass

    def __from_raw_state(self, state_raw):
        env_name = config.Simulator.Gym.Environment
        if env_name == 'FetchReach-v1':
            self.achieved = state_raw['achieved_goal']
            self.desired = state_raw['desired_goal']
            self.state =  np.hstack((state_raw['observation'], self.desired))
            return
        self.state = state_raw

    def from_reset(self, state_raw) -> None:
        self.__from_raw_state(state_raw)

    def from_step(self, state_raw, reward_raw, done: bool) -> None:
        self.__from_raw_state(state_raw)
        self.reward_raw = reward_raw
        self.done = done

    def as_input(self):
        return self.state

    def dim_state(self) -> int:
        return len(self.state)