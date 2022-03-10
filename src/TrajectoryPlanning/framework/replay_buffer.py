import numpy as np
import random
from simulator.targets import GameState
from typing import Any

class Transition:
    def __init__(self, state: GameState, action: np.ndarray, reward: float, next_state: GameState) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def to_serializable(self) -> Any:
        return [
            self.state.to_serializable(),
            self.action.tolist(),
            self.reward,
            self.next_state.to_serializable(),
        ]

    @staticmethod
    def from_serializable(x):
        return Transition(
            GameState.from_serializable(x[0]),
            np.array(x[1]),
            x[2],
            GameState.from_serializable(x[3])
        )

class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.__capacity = size
        self.__size = 0
        self.__begin = 0
        self.__buffer: list[Transition] = []

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

    def clear(self):
        self.__size = 0
        self.__begin = 0
        self.__buffer.clear()

    def sample(self, count: int) -> list[Transition]:
        assert 0 <= count <= self.__size
        return random.sample(self.__buffer, count)

    def to_serializable(self) -> Any:
        return [x.to_serializable() for x in self.__buffer]

    def from_serializable(self, x) -> None:
        self.clear()
        for item in x:
            self.append(Transition.from_serializable(item))