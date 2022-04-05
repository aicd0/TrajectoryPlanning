from framework.configuration import Configuration

def main():
    config = Configuration()
    print(config.get('a', 1))
    print(config.get('b', 2))
    print(config.get('c', 3))
    print(config.get('a', 0))
    print(config.get('b', 0))
    print(config.get('c', 0))
    config.save('test.txt')
    
    config = Configuration()
    config.load('test.txt')
    print(config.get('a', 0))
    print(config.get('b', 0))
    print(config.get('c', 0))