import config
from copy import deepcopy
from framework.model import Actor, Critic
from simulator.engine import Connector
from simulator.conversion import decode_state

def main():
    # Attach to a running session.
    connector = Connector()
    assert connector.connect(config.SessionName)
    eng = connector.engine()

    # Initialize simulator.
    eng.sim_initialize(nargout=0)
    
    # Reset simulator to get sizes of states and actions.
    eng.sim_reset(nargout=0)
    state_raw = eng.workspace['state']
    num_actions = len(state_raw['config'])
    num_states = len(decode_state(state_raw))

    # Initialize critic and actor.
    critic = Critic(num_states)
    actor = Actor(num_states, num_actions)

    # Initialize target critic and target actor.
    critic_target = deepcopy(critic)
    actor_target = deepcopy(actor)

    # Initialize replay buffer.
    # raise NotImplementedError()

    

    return 0

if __name__ == '__main__':
    _ = main()