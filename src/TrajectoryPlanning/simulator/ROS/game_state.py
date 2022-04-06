import config
import numpy as np
import utils.math
from math import pi
from simulator.game_state import GameStateBase
from typing import Any

class GameState (GameStateBase):
    def __init__(self):
        GameStateBase.__init__(self)
        self.joint_position: np.ndarray = None
        self.collision: bool = None

    def from_joint_states(self, joint_states) -> None:
        self.joint_position = np.array(joint_states.position, dtype=config.DataType.Numpy)
        self.joint_position[0] = utils.math.period_map(self.joint_position[0], -pi, pi)
        
    def support_her(self) -> bool:
        return not self.collision

    def _as_input(self) -> np.ndarray:
        obj_rel_pos = self.desired - self.achieved

        return np.concatenate([
            self.joint_position,
            self.achieved,
            self.desired,
            obj_rel_pos,
        ], dtype=config.DataType.Numpy)

    def _to_list(self) -> list:
        return [
            self.joint_position,
            self.collision,
        ]

    @staticmethod
    def _from_list(x: list) -> Any:
        o = GameState()
        (
            o.joint_position,
            o.collision,
        ) = x
        return o