import numpy as np
import random
from simulator.targets import GameState

class Transition:
    def __init__(self, state: GameState, action: np.ndarray, reward: float, next_state: GameState) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.__capacity = size
        self.__size = 0
        self.__buffer: list[Transition] = []
        self.__begin = 0

    def __len__(self) -> int:
        return self.__size

    def append(self, transition: Transition) -> None:
        if self.__size < self.__capacity:
            self.__buffer.append(transition)
            self.__size += 1
        else:
            self.__buffer[self.__begin] = transition
            self.__begin += 1
            if self.__begin >= self.__size:
                self.__begin = 0

    def sample(self, count: int) -> list[Transition]:
        assert 0 <= count <= self.__size
        return random.sample(self.__buffer, count)