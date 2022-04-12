from abc import abstractmethod

class Noise:
    def __init__(self, dim) -> None:
        self.dim = dim

    @abstractmethod
    def sample(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()