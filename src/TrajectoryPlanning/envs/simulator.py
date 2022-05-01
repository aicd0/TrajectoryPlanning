import numpy as np
import utils.name
from .state import State
from abc import abstractmethod
from framework.configuration import Configuration

class Simulator:
    __names = utils.name.Name('sim')

    def __init__(self, name: str | None):
        self.name = Simulator.__names.get(self.__class__.__name__, name)
        self.configs = Configuration(self.name)
        self._state = None

    @abstractmethod
    def _get_state(self) -> State:
        raise NotImplementedError()

    def state(self) -> State:
        if self._state is None:
            self._state = self._get_state()
        return self._state
        
    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _reset(self) -> None:
        raise NotImplementedError()

    def reset(self) -> State:
        self._reset()
        return self.state()

    @abstractmethod
    def step(self, action: np.ndarray) -> State:
        raise NotImplementedError()
    
    @abstractmethod
    def plot_reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def plot_step(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def dim_action(self) -> int:
        raise NotImplementedError()