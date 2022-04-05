import config
from framework.configuration import global_configs as configs

__platform = configs.get(config.Simulator.FieldPlatform)

if False:
    pass
elif __platform == 'matlab':
    from .MATLAB.game import Game
    from .MATLAB.game_state import GameState
    from .MATLAB.simulator import Simulator
elif __platform == 'gym':
    from .gym.game import Game
    from .gym.game_state import GameState
    from .gym.simulator import Simulator
elif __platform== 'ros':
    from .ROS.game import Game
    from .ROS.game_state import GameState
    from .ROS.simulator import Simulator
else:
    raise Exception('Platform not supported')