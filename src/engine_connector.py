import matlab.engine

class EngineConnector():
    def __init__(self) -> None:
        self.__engine = None

    def connect(self, name) -> bool:
        if not self.__engine is None:
            return True

        eng_names = matlab.engine.find_matlab()

        if not name in eng_names:
            print('MATLAB session "%s" not found.' % name)
            return False

        self.__engine = matlab.engine.connect_matlab(name)
        return True

    def engine(self):
        assert not self.__engine is None
        return self.__engine