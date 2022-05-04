from typing import Tuple
from .reward import Reward
from .simulator import Simulator

def create_environment(platform: str, *arg, **kwarg) -> Tuple[Simulator, Reward]:
    if platform == 'gazebo':
        from .gazebo.simulator import Gazebo
        from .gazebo.reward import GazeboReward
        return Gazebo(*arg, **kwarg), GazeboReward()

    elif platform == 'gym':
        from .gym.simulator import Gym
        from .gym.reward import GymReward
        return Gym(*arg, **kwarg), GymReward()

    elif platform == 'matlab':
        from .matlab.simulator import Matlab
        from .matlab.reward import MatlabReward
        return Matlab(*arg, **kwarg), MatlabReward()
        
    else:
        raise Exception("Unrecognized platform '%s'" % platform)