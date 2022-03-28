import config

if config.Simulator.Platform == 'matlab':
    from simulator.MATLAB.game import Game
    from simulator.MATLAB.game_state import GameState
    from simulator.MATLAB.simulator import Simulator
elif config.Simulator.Platform == 'gym':
    from simulator.gym.game import Game
    from simulator.gym.game_state import GameState
    from simulator.gym.simulator import Simulator
elif config.Simulator.Platform == 'ros':
    from simulator.ROS.game import Game
    from simulator.ROS.game_state import GameState
    from simulator.ROS.simulator import Simulator
else:
    raise Exception('Platform not supported')