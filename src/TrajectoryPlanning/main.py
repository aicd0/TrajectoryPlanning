import config
import utils.platform
from framework.configuration import Configuration
from targets import target

def main():
    utils.platform.check_platform()
    target.main()

if __name__ == '__main__':
    main()