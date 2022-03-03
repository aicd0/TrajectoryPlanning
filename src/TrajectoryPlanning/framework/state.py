import config
import numpy as np

class State:
    def __init__(self) -> None:
        pass

    def from_matlab(self, src=None):
        """Convert states from MATLAB."""
        if not src is None:
            self.config = np.array(src['config']._data, dtype=config.DataType.Numpy)
            self.achieved = np.array(src['achieved']._data, dtype=config.DataType.Numpy)
            self.desired = np.array(src['desired']._data, dtype=config.DataType.Numpy)
            collision = src['collision'][0]
            self.self_collision = bool(collision[0])
            self.world_collision = bool(collision[1])
            self.deadlock = bool(src['deadlock'])

        self.as_input = np.concatenate((self.config, self.achieved, self.desired), dtype=config.DataType.Numpy)
        