import numpy as np
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class ActorBase(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, dim_state, dim_action):
        super(ActorBase, self).__init__()

    @abstractmethod
    def forward(self, states):
        raise NotImplementedError()

class CriticBase(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(CriticBase, self).__init__()

    @abstractmethod
    def forward(self, states, actions):
        raise NotImplementedError()