from .ddpg import *
from .sac import *
from torch import nn
from typing import Any

def create(model: str, *arg, **kwarg) -> nn.Module:
    if model == 'ddpg/l5/actor':
        return ddpg.l5.Actor(*arg, **kwarg)
    elif model == 'ddpg/l5/critic':
        return ddpg.l5.Critic(*arg, **kwarg)
    elif model == 'sac/l3/actor':
        return sac.l3.Actor(*arg, **kwarg)
    elif model == 'sac/l3/critic':
        return sac.l3.Critic(*arg, **kwarg)
    elif model == 'sac/l4/actor':
        return sac.l4.Actor(*arg, **kwarg)
    elif model == 'sac/l4/critic':
        return sac.l4.Critic(*arg, **kwarg)
    else:
        raise Exception('Unsupported model type')