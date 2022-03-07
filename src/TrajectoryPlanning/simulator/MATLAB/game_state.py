import config
import numpy as np

class GameState:
    def __init__(self):
        self.__as_input = None

    def update(self):
        self.__as_input = None

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

    def as_input(self) -> np.ndarray:
        if self.__as_input is None:
            obj_rel_pos = self.desired - self.achieved
            input_tuple = (
                self.config,
                self.achieved,
                self.desired,
                obj_rel_pos
            )
            self.__as_input = np.concatenate(input_tuple, dtype=config.DataType.Numpy)
        return self.__as_input

    def dim_state(self) -> int:
        return len(self.as_input())