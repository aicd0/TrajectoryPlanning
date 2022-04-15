import config

__target = config.Common.Target

if __target == 'train':
    from . import train as target
elif __target == 'test':
    from . import test as target
elif __target == 'debug':
    from . import debug as target
else:
    raise Exception('Unrecognized target')