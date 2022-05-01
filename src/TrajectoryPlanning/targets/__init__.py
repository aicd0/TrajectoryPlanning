import config
from framework.configuration import global_configs as configs

__target = configs.get(config.Common.Target_)

if __target == 'train':
    from . import train as target
elif __target == 'test':
    from . import test as target
elif __target == 'a_star':
    from . import a_star as target
elif __target == 'apf':
    from . import apf as target
elif __target == 'debug':
    from . import debug as target
else:
    raise Exception('Unrecognized target')