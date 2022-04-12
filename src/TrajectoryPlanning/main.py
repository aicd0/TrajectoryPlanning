import utils.platform
from targets import target

def main():
    utils.platform.check_platform()
    target.main()

if __name__ == '__main__':
    main()