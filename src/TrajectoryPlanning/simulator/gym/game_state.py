import config
import numpy as np

class GameState:
    def __init__(self):
        pass

    def __from_state(self, state):
        env_name = config.Simulator.Gym.Environment
        if env_name == 'FetchReach-v1':
            self.achieved = state['achieved_goal']
            self.desired = state['desired_goal']
            self.state = np.hstack((state['observation'], self.achieved, self.desired))
            return
        self.state = state

    def from_reset(self, dim_state: int, dim_action: int, state) -> None:
        self.__dim_state = dim_state
        self.__dim_action = dim_action
        self.__from_state(state)

    def from_step(self, state, reward_raw, done: bool) -> None:
        self.__from_state(state)
        self.reward_raw = reward_raw
        self.done = done

    def as_input(self):
        return self.state

    def dim_state(self) -> int:
        return self.__dim_state

    def dim_action(self) -> int:
        return self.__dim_action