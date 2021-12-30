import config
import torch
import numpy as np

class State:
    def __init__(self) -> None:
        pass

    def from_matlab(self, src=None):
        """Convert state from MATLAB."""
        if not src is None:
            self.config = np.array(src['config']._data, dtype=config.NumpyDType)
            self.achieved = np.array(src['achieved']._data, dtype=config.NumpyDType)
            self.desired = np.array(src['desired']._data, dtype=config.NumpyDType)
            self.obstacle = np.array(src['obstacle']._data, dtype=config.NumpyDType)
            collision = src['collision'][0]
            self.self_collision = bool(collision[0])
            self.world_collision = bool(collision[1])

        self.as_input = np.concatenate((self.config, self.achieved,
            self.desired, self.obstacle), dtype=config.NumpyDType)