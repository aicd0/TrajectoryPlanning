import config

if config.Target == 'train':
    import targets.train as target
elif config.Target == 'test':
    import targets.test as target
elif config.Target == 'debug':
    import targets.debug as target
else:
    raise Exception('Unsupported target')