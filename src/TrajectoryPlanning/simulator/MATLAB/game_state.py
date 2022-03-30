import config
import numpy as np
from simulator.game_state import GameStateBase
from typing import Any

class GameState (GameStateBase):
    def __init__(self):
        GameStateBase.__init__(self)
        self.config = None
        self.self_collision = None
        self.world_collision = None
        self.deadlock = None

    def from_matlab(self, src):
        """Convert states from MATLAB."""
        self.config = np.array(src['config']._data, dtype=config.DataType.Numpy)
        self.achieved = np.array(src['achieved']._data, dtype=config.DataType.Numpy)
        self.desired = np.array(src['desired']._data, dtype=config.DataType.Numpy)
        collision = src['collision'][0]
        self.self_collision = bool(collision[0])
        self.world_collision = bool(collision[1])
        self.deadlock = bool(src['deadlock'])
        self.update()

    def _as_input(self) -> np.ndarray:
        obj_rel_pos = self.desired - self.achieved

        return np.concatenate((
            self.config,
            self.achieved,
            self.desired,
            obj_rel_pos,
        ), dtype=config.DataType.Numpy)

    def _to_list(self) -> list:
        return [
            self.config,
            self.self_collision,
            self.world_collision,
            self.deadlock,
        ]

    @staticmethod
    def _from_list(x: list) -> Any:
        o = GameState()
        (
            o.config,
            o.self_collision,
            o.world_collision,
            o.deadlock,
        ) = x
        return o