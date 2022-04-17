from .ddpg import *
from .sac import *
from torch import nn
from typing import Any

def create(model: str, *args) -> nn.Module:
    if (model == 'ddpg/l5/actor'):
        return ddpg.l5.Actor(*args)
    elif (model == 'ddpg/l5/critic'):
        return ddpg.l5.Critic(*args)
    elif (model == 'sac/l3/actor'):
        return sac.l3.Actor(*args)
    elif (model == 'sac/l3/critic'):
        return sac.l3.Critic(*args)
    else:
        raise Exception('Unsupported model type')