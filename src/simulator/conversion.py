import numpy as np

def decode_state(state):
    # Robot configurations.
    config = state['config']._data

    # Achieved position.
    achieved = state['achieved']._data

    # Target position.
    target = state['target']._data

    # Obstacle information.
    obstacle = state['obstacle']._data

    # Colliding check.
    collide = state['collide']
    collide = np.array([1.0 if collide else -1.0])

    # Concatenate all states.
    state_concat = np.concatenate((config, achieved, target, obstacle, collide))
    return state_concat