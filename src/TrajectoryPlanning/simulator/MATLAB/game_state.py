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
            self.__as_input = np.concatenate((self.config, self.desired - self.achieved), dtype=config.DataType.Numpy)
        return self.__as_input

    def dim_state(self) -> int:
        return len(self.as_input())

    def dim_action(self) -> int:
        return len(self.config)