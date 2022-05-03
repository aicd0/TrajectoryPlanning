import numpy as np
from abc import abstractmethod

class State:
    def __init__(self):
        self.__as_input = None

    def dim_state(self) -> int:
        return len(self.as_input())

    def as_input(self):
        if self.__as_input is None:
            self.__as_input = self._as_input()
        return self.__as_input

    def update(self):
        self.__as_input = None

    @abstractmethod
    def support_her(self) -> bool:
        return False

    @abstractmethod
    def _as_input(self) -> np.ndarray:
        raise NotImplementedError()