import numpy as np
from framework.noise import Noise

class UniformNoise(Noise):
    def __init__(self, dim, low, high) -> None:
        super().__init__(dim)
        self.low = low
        self.high = high
    
    def sample(self):
        return np.random.uniform(self.low, self.high, self.dim)