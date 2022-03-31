import config

if config.Simulator.ROS.DynamicEnabled:
    from .game_states.dynamic import GameState
else:
    from .game_states.discrete import GameState