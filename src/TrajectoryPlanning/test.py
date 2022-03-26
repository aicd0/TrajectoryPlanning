import utils.platform
import targets.test

def main():
    utils.platform.check_platform()
    targets.test.main()

if __name__ == '__main__':
    main()