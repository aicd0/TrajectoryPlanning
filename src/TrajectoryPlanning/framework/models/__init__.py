from . import controller
from . import planner
from typing import Tuple

def create(model: str, arg1, arg2) -> Tuple:
    if False:
        pass
    elif (model == 'controller/actor'):
        return controller.Actor(arg1, arg2)
    elif (model == 'controller/critic'):
        return controller.Critic(arg1, arg2)
    elif (model == 'planner/actor'):
        return planner.Actor(arg1, arg2)
    elif (model == 'planner/critic'):
        return planner.Critic(arg1, arg2)
    else:
        raise Exception('Unsupported model type')