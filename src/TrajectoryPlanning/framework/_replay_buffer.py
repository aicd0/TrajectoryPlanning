from tkinter import N
import numpy as np
import random
from simulator import GameState
from typing import Any, Iterator

class Transition:
    def __init__(self, state: GameState, action: np.ndarray, reward: float, next_state: GameState, p: float) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.p = p
        self.node = None

    def to_serializable(self) -> Any:
        return [
            self.state.to_list(),
            self.action.tolist(),
            self.reward,
            self.next_state.to_list(),
            self.p,
        ]

    @staticmethod
    def from_serializable(x):
        return Transition(
            GameState.from_list(x[0]),
            np.array(x[1]),
            x[2],
            GameState.from_list(x[3]),
            x[4],
        )

class ReplayBufferNode:
    def __init__(self, val: Transition, parent=None):
        self.count = 0
        self.left = None
        self.right = None
        self.val = val
        self.sum = val.p
        val.node = self
        self.parent = parent

        if not parent is None:
            parent.count += 1

    def insert(self, trans: Transition):
        if self.count == 0:
            self.left = ReplayBufferNode(self.val, self)
            self.right = ReplayBufferNode(trans, self)
            self.val = None
        else:

        if self.left is None:
            trans.parent = self
            self.left = trans

        if self.right is None:
            trans.parent = self
            
        is_node_left = isinstance(self.left, ReplayBufferNode)
        is_node_right = isinstance(self.right, ReplayBufferNode)

        if not is_node_left:
            self.left.insert(trans)

        if is_node_left:
            left_count = self.left.count
        else:
            left_count = 0

        if is_node_right:
            right_count = self.right.count
        else:
            right_count = 0

        if left_count <= right_count:
            if is_node_left:
                self.left.insert(trans)
            else:
                node = ReplayBufferNode()
                node.insert(self.left)
                node.insert(trans)
                self.count += 1
        else:
            if is_node_right:
                self.right.insert(trans)
            else:
                node = ReplayBufferNode()
                node.insert(self.right)
                node.insert(trans)
                self.count += 1

        self.sum += trans.p

class ReplayBufferIterator (Iterator):
    def __init__(self, buffer: list, begin: int) -> None:
        self.__buffer = buffer
        self.__offset = begin
        self.__index = 0

    def __next__(self) -> Transition:
        i = self.__index
        self.__index += 1
        if i >= len(self.__buffer):
            raise StopIteration
        i += self.__offset
        if i >= len(self.__buffer):
            i -= len(self.__buffer)
        return self.__buffer[i]

class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.__capacity = size
        self.__size = 0
        self.__begin = 0
        self.__buffer: list[Transition] = []

    def __len__(self) -> int:
        return self.__size
        
    def __iter__(self) -> ReplayBufferIterator:
        return ReplayBufferIterator(self.__buffer, self.__begin)

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
        x = []
        for item in self:
            x.append(item.to_serializable())
        return x

    @staticmethod
    def from_serializable(x, size) -> Any:
        o = ReplayBuffer(size)
        for item in x:
            o.append(Transition.from_serializable(item))
        return o