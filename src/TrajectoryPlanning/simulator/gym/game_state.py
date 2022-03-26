import config
import numpy as np
from typing import Any

class GameState:
    def __init__(self):
        self.reward_raw = None
        self.done = None
        self.achieved = None
        self.desired = None

    def __from_raw_state(self, state_raw):
        env_name = config.Simulator.Gym.Environment

        if env_name == 'FetchReach-v1':
            self.achieved = state_raw['achieved_goal']
            self.desired = state_raw['desired_goal']
            self.state =  np.hstack((state_raw['observation'], self.desired))
            return

        self.state = state_raw

    def update(self) -> None:
        pass

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

    def to_serializable(self) -> Any:
        x = [
            self.state,
            self.reward_raw,
            self.done,
            self.achieved,
            self.desired,
        ]
        for i in range(len(x)):
            if isinstance(x[i], np.ndarray):
                x[i] = x[i].tolist()
            elif isinstance(x[i], np.float32):
                x[i] = float(x[i])
        return x

    @staticmethod
    def from_serializable(self, x) -> None:
        obj = GameState()
        obj.state = x[0]
        obj.reward_raw = x[1]
        obj.done = x[2]
        obj.achieved = x[3]
        obj.desired = x[4]
        for i in range(len(x)):
            if isinstance(x[i], list):
                x[i] = np.array(x[i])
        return obj