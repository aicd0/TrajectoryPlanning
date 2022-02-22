import config
import matlab.engine
import os
import utils.string_utils

class Connector():
    def __init__(self) -> None:
        self.__engine = None

    def connect(self) -> bool:
        if not self.__engine is None:
            return True

        # Read session name from target file.
        session_file = utils.string_utils.to_file_path(config.MatlabSessionFile)
        if not os.path.exists(session_file):
            print('MATLAB not started.')
            return False
        with open(session_file, 'rb') as f:
            eng_name = f.read().decode()

        # Check if the session exists.
        eng_names = matlab.engine.find_matlab()
        if not eng_name in eng_names:
            print('MATLAB session "%s" not found.' % eng_name)
            return False

        # Connect to the existing session.
        self.__engine = matlab.engine.connect_matlab(eng_name)
        return True

    def engine(self):
        assert not self.__engine is None
        return self.__engine