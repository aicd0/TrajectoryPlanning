import utils.platform
import targets.train

def main():
    utils.platform.check_platform()
    targets.train.main()

if __name__ == '__main__':
    main()