import config
import numpy as np
from envs.game_state import GameStateBase
from math import pi
from typing import Any

class GameState (GameStateBase):
    def __init__(self):
        GameStateBase.__init__(self)
        self.achieved = None
        self.desired = None
        self.joint_position: np.ndarray = None
        self.collision: bool = None

    def from_joint_states(self, joint_states) -> None:
        self.joint_position = np.array(joint_states.position, dtype=config.Common.DataType.Numpy)
        
    def support_her(self) -> bool:
        return not self.collision

    def _as_input(self) -> np.ndarray:
        obj_rel_pos = self.desired - self.achieved

        return np.concatenate([
            self.joint_position,
            self.achieved,
            self.desired,
            obj_rel_pos,
        ], dtype=config.Common.DataType.Numpy)