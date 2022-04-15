import config
import numpy as np
import random
from envs import GameState
from framework.configuration import Configuration
from typing import Any, Iterator

class Transition:
    def __init__(self, state: GameState, action: np.ndarray, reward: float, next_state: GameState, p: float=1) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.node = None
        self.__p = p

    @property
    def p(self) -> float:
        return self.__p

    @p.setter
    def p(self, value) -> None:
        self.__p = value
        self.node.update_sum()

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
    def __init__(self, trans: Transition, parent=None):
        self.count = 0
        self.left = None
        self.right = None
        self.trans = trans
        self.sum = trans.p
        self.parent = parent
        trans.node = self

        if not parent is None:
            parent.count += 1

    def push(self, trans: Transition):
        if self.trans is None:
            if self.left.count <= self.right.count:
                self.left.push(trans)
            else:
                self.right.push(trans)
        else:
            # Split into two nodes.
            self.left = ReplayBufferNode(self.trans, self)
            self.right = ReplayBufferNode(trans, self)
            self.trans = None

        if not self.parent is None:
            self.parent.count += 1
        self.sum += trans.p

    def find(self, pos: float) -> Transition:
        assert 0 <= pos <= self.sum
        if self.trans is None:
            if pos < self.left.sum:
                return self.left.find(pos)
            else:
                return self.right.find(pos - self.left.sum)
        return self.trans

    def update_sum(self) -> None:
        if self.trans is None:
            self.sum = self.left.sum + self.right.sum
        else:
            self.sum = self.trans.p
        if not self.parent is None:
            self.parent.update_sum()

    def update_trans(self, trans: Transition) -> None:
        assert not self.trans is None
        self.trans = trans
        trans.node = self
        self.update_sum()

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
    def __init__(self, configs: Configuration) -> None:
        self.__configs = configs
        self.__capacity = self.__configs.get(config.Training.Agent.ReplayBuffer_)
        assert self.__capacity > 0
        self.__size = 0
        self.__begin = 0
        self.__buffer: list[Transition] = []
        self.__head: ReplayBufferNode = None

    def __len__(self) -> int:
        return self.__size
        
    def __iter__(self) -> ReplayBufferIterator:
        return ReplayBufferIterator(self.__buffer, self.__begin)

    def append(self, transition: Transition) -> None:
        if self.__size < self.__capacity:
            # Push to binary tree.
            if self.__head is None:
                self.__head = ReplayBufferNode(transition)
            else:
                self.__head.push(transition)
            
            # Append to buffer.
            self.__buffer.append(transition)
            self.__size += 1
        else:
            # Update node.
            node = self.__buffer[self.__begin].node
            node.update_trans(transition)

            # Replace the oldest item.
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
        per_enabled = self.__configs.get(config.Training.Agent.PER.Enabled_)
        if per_enabled:
            samples = []
            for _ in range(count):
                pos = random.uniform(0, self.__head.sum)
                samples.append(self.__head.find(pos))
        else:
            samples = random.sample(self.__buffer, count)
        return samples

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