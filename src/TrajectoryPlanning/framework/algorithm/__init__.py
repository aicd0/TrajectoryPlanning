from . import ddpg
from . import sac
from framework.agent import AgentBase

def create_agent(algorithm: str, *args) -> AgentBase:
    if algorithm == 'ddpg':
        return ddpg.DDPG(*args)
    elif algorithm == 'sac':
        return sac.SAC(*args)
    else:
        raise Exception('Unrecognized algorithm')