import numpy as np
from .state import State
from abc import abstractmethod
from typing import Tuple

class Reward:
    def __init__(self, max_steps: int=-1) -> None:
        assert max_steps != 0
        self.max_steps = max_steps
        self.reward = 0
        self.done = False
        self.step = 0

    def reset(self) -> None:
        self.step = 0
        self.done = False

    def update(self, state: State, action: np.ndarray, next_state: State) -> Tuple:
        if not self.done:
            self._update(state, action, next_state)
            self.step += 1
            if self.max_steps > 0 and self.step >= self.max_steps:
                self.done = True
        return self.reward, self.done

    @abstractmethod
    def _update(self, state: State, action: np.ndarray, next_state: State) -> None:
        raise NotImplementedError()