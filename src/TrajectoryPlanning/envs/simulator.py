from copy import deepcopy
import numpy as np
import utils.name
from .reward import Reward
from .state import State
from abc import abstractmethod
from framework.configuration import Configuration

class Simulator:
    __names = utils.name.Name('sim')

    def __init__(self, name: str | None):
        self.name = Simulator.__names.get(self.__class__.__name__, name)
        self.configs = Configuration(self.name)
        self.record = False
        self.records: list[State] = []
        self._state = None
        
    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _get_state(self) -> State:
        raise NotImplementedError()

    def state(self) -> State:
        if self._state is None:
            self._state = self._get_state()
        return self._state

    @abstractmethod
    def _reset(self) -> None:
        raise NotImplementedError()

    def reset(self) -> State:
        self._reset()
        return self.state()

    @abstractmethod
    def _step(self, action: np.ndarray) -> None:
        raise NotImplementedError()

    def step(self, action: np.ndarray) -> State:
        self._step(action)
        state = self.state()
        if self.record:
            self.records.append(deepcopy(state))
        return state
    
    @abstractmethod
    def plot_reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def plot_step(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def dim_action(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def reward(self) -> Reward:
        raise NotImplementedError()