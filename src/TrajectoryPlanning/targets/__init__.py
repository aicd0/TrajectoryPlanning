import config
from framework.configuration import global_configs as configs

__target = configs.get(config.Common.Target_)
if   __target == 'a_star_rect': from . import a_star_rect as target
elif __target == 'a_star_static': from . import a_star_static as target
elif __target == 'apf_rect': from . import apf_rect as target
elif __target == 'apf_static': from . import apf_static as target
elif __target == 'debug': from . import debug as target
elif __target == 'plot': from . import plot as target
elif __target == 'rl_rect': from . import rl_rect as target
elif __target == 'rl_static': from . import rl_static as target
elif __target == 'rl_train': from . import rl_train as target
elif __target == 'robot': from . import robot as target
elif __target == 'stopwatch': from . import stopwatch as target
elif __target == 'user1_rect': from . import user1_rect as target
elif __target == 'user1_static': from . import user1_static as target
elif __target == 'workspace': from . import workspace as target
else: raise Exception("Unrecognized target '%s'" % __target)