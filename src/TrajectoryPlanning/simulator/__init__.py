import config

if config.Simulator.Platform == 'matlab':
    from .MATLAB.game import Game
    from .MATLAB.game_state import GameState
    from .MATLAB.simulator import Simulator
elif config.Simulator.Platform == 'gym':
    from .gym.game import Game
    from .gym.game_state import GameState
    from .gym.simulator import Simulator
elif config.Simulator.Platform == 'ros':
    from .ROS.game import Game
    from .ROS.game_state import GameState
    from .ROS.simulator import Simulator
else:
    raise Exception('Platform not supported')