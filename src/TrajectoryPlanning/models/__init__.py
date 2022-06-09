from .ddpg import *
from .sac import *
from torch import nn
from typing import Any

def create(model: str, *arg, **kwarg) -> nn.Module:
    if model == 'ddpg/h3/actor':
        return ddpg.h3.Actor(*arg, **kwarg)
    elif model == 'ddpg/h3/critic':
        return ddpg.h3.Critic(*arg, **kwarg)
    elif model == 'sac/h2/actor':
        return sac.h2.Actor(*arg, **kwarg)
    elif model == 'sac/h2/critic':
        return sac.h2.Critic(*arg, **kwarg)
    elif model == 'sac/h3/actor':
        return sac.h3.Actor(*arg, **kwarg)
    elif model == 'sac/h3/critic':
        return sac.h3.Critic(*arg, **kwarg)
    else:
        raise Exception("Unsupported model type '%s'" % model)