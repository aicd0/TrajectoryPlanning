import numpy as np
import utils.name
from .game_state import GameStateBase
from abc import abstractmethod
from framework.configuration import Configuration

class Simulator:
    __names = utils.name.Name('sim')

    def __init__(self, name: str | None):
        self.name = Simulator.__names.get(self.__class__.__name__, name)
        self.configs = Configuration(self.name)
        self._state = None

    @abstractmethod
    def _get_state(self) -> GameStateBase:
        raise NotImplementedError()

    def state(self) -> GameStateBase:
        if self._state is None:
            self._state = self._get_state()
        return self._state
        
    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _reset(self) -> None:
        raise NotImplementedError()

    def reset(self) -> GameStateBase:
        self._reset()
        return self.state()

    @abstractmethod
    def step(self, action: np.ndarray) -> GameStateBase:
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
 
def create_simulator(platform: str, *arg, **kwarg) -> Simulator:
    if platform == 'gym':
        from .gym.simulator import Gym
        return Gym(*arg, **kwarg)
    elif platform == 'matlab':
        from .matlab.simulator import Matlab
        return Matlab(*arg, **kwarg)
    elif platform == 'ros':
        from .ros.simulator import ROS
        return ROS(*arg, **kwarg)
    else:
        raise Exception('Unrecognized platform')