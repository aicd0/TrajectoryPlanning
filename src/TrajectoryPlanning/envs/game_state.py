import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Any

class GameStateBase:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.achieved: np.ndarray = None
        self.desired: np.ndarray = None
        self.__as_input = None

    def dim_state(self) -> int:
        return len(self.as_input())

    def as_input(self):
        if self.__as_input is None:
            self.__as_input = self._as_input()
        return self.__as_input

    def update(self):
        self.__as_input = None

    def to_list(self) -> list:
        x = [
            self.achieved,
            self.desired,
        ]
        x.extend(self._to_list())
        return x

    @classmethod
    def from_list(cls, x: list) -> Any:
        o: GameStateBase = cls._from_list(x[2:])
        (
            o.achieved,
            o.desired,
        ) = x[:2]
        return o

    @abstractmethod
    def support_her(self) -> bool:
        return False

    @abstractmethod
    def _as_input(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _to_list(self) -> list:
        raise NotImplementedError()

    @staticmethod
    def _from_list(x: list) -> Any:
        raise NotImplementedError()