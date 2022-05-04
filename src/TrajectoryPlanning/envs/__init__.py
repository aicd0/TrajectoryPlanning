from .simulator import Simulator

def create_simulator(platform: str, *arg, **kwarg) -> Simulator:
    if platform == 'gazebo':
        from .gazebo.simulator import Gazebo
        return Gazebo(*arg, **kwarg)
    elif platform == 'gym':
        from .gym.simulator import Gym
        return Gym(*arg, **kwarg)
    elif platform == 'matlab':
        from .matlab.simulator import Matlab
        return Matlab(*arg, **kwarg)
    else:
        raise Exception("Unrecognized platform '%s'" % platform)