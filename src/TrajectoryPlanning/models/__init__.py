from .ddpg import *
from typing import Any

def create(model: str, *args) -> Any:
    if False:
        pass
    elif (model == 'ddpg/l5/actor'):
        return ddpg.l5.Actor(*args)
    elif (model == 'ddpg/l5/critic'):
        return ddpg.l5.Critic(*args)
    else:
        raise Exception('Unsupported model type')