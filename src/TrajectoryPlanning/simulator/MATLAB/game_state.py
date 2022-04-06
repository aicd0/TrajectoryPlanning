import config
import numpy as np
from simulator.game_state import GameStateBase
from typing import Any

class GameState (GameStateBase):
    def __init__(self):
        GameStateBase.__init__(self)
        self.joint_position = None
        self.collision = None
        self.deadlock = None

    def from_matlab(self, src):
        """Convert states from MATLAB."""
        self.joint_position = np.array(src['config']._data, dtype=config.DataType.Numpy)
        self.achieved = np.array(src['achieved']._data, dtype=config.DataType.Numpy)
        self.desired = np.array(src['desired']._data, dtype=config.DataType.Numpy)
        self.collision = bool(src['collision'])
        self.deadlock = bool(src['deadlock'])
        self.update()

    def _as_input(self) -> np.ndarray:
        obj_rel_pos = self.desired - self.achieved

        return np.concatenate((
            self.joint_position,
            self.achieved,
            self.desired,
            obj_rel_pos,
        ), dtype=config.DataType.Numpy)

    def _to_list(self) -> list:
        return [
            self.joint_position,
            self.collision,
            self.deadlock,
        ]

    @staticmethod
    def _from_list(x: list) -> Any:
        o = GameState()
        (
            o.joint_position,
            o.collision,
            o.deadlock,
        ) = x
        return o