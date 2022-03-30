import config
import numpy as np
from simulator.game_state import GameStateBase
from typing import Any

class GameState (GameStateBase):
    def __init__(self):
        GameStateBase.__init__(self)
        self.joint_states: np.ndarray = None
        self.collision: bool = None

    def _as_input(self) -> np.ndarray:
        obj_rel_pos = self.desired - self.achieved

        return np.concatenate((
            self.joint_states,
            self.achieved,
            self.desired,
            obj_rel_pos,
        ), dtype=config.DataType.Numpy)

    def _to_list(self) -> list:
        return [
            self.joint_states,
            self.collision,
        ]

    @staticmethod
    def _from_list(x: list) -> Any:
        o = GameState()
        (
            o.joint_states,
            o.collision,
        ) = x
        return o