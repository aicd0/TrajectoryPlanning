import config
import numpy as np
from simulator.game_state import GameStateBase
from typing import Any

class GameState (GameStateBase):
    def __init__(self):
        GameStateBase.__init__(self)
        self.states = None
        self.reward_raw = None
        self.done = None

    def __from_raw_state(self, state_raw):
        env_name = config.Simulator.Gym.Environment

        if env_name == 'FetchReach-v1':
            self.achieved = state_raw['achieved_goal']
            self.desired = state_raw['desired_goal']
            self.states = np.concatenate((
                state_raw['observation'],
                self.desired,
            ), dtype=config.DataType.Numpy)
        else:
            self.states = state_raw

    def from_reset(self, state_raw) -> None:
        self.__from_raw_state(state_raw)

    def from_step(self, state_raw, reward_raw, done: bool) -> None:
        self.__from_raw_state(state_raw)
        self.reward_raw = reward_raw
        self.done = done

    def _as_input(self) -> np.ndarray:
        return self.states

    def _to_list(self) -> list:
        return [
            self.states,
            self.reward_raw,
            self.done,
        ]

    @staticmethod
    def _from_list(x: list) -> Any:
        o = GameState()
        (
            o.states,
            o.reward_raw,
            o.done,
        ) = x
        return o