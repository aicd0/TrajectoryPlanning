import config

if config.Target == 'train':
    from . import train as target
elif config.Target == 'test':
    from . import test as target
elif config.Target == 'debug':
    from . import debug as target
else:
    raise Exception('Unsupported target')