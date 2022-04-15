import config
from framework.configuration import global_configs as configs

__platform = configs.get(config.Environment.Platform_)

if __platform == 'matlab':
    from .matlab.game import Game
    from .matlab.game_state import GameState
    from .matlab.simulator import Simulator
elif __platform == 'gym':
    from .gym.game import Game
    from .gym.game_state import GameState
    from .gym.simulator import Simulator
elif __platform== 'ros':
    from .ros.game import Game
    from .ros.game_state import GameState
    from .ros.simulator import Simulator
else:
    raise Exception('Platform not supported')