import numpy as np
import utils.math

class WorkspaceEstimator:
    def __init__(self) -> None:
        self.count = 0

    def initialized(self) -> bool:
        return self.count > 0

    def push(self, pt: np.ndarray) -> None:
        if self.initialized():
            self.high = np.max(np.concatenate((self.high[np.newaxis, :], pt[np.newaxis, :])), axis=0, keepdims=False)
            self.low = np.min(np.concatenate((self.low[np.newaxis, :], pt[np.newaxis, :])), axis=0, keepdims=False)
            self.count += 1
            self.center += (pt - self.center) / self.count
            self.radius = max(self.radius, utils.math.distance(pt, self.center))
        else:
            self.high = pt
            self.low = pt
            self.count = 1
            self.center = pt
            self.radius = 0

    def sample(self) -> np.ndarray:
        if not self.initialized():
            raise Exception()
        while True:
            pt = np.random.uniform(self.low, self.high)
            if utils.math.distance(self.center, pt) <= self.radius:
                break
        return pt