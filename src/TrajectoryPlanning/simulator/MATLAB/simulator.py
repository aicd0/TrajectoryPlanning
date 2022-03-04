import config
import matlab
import numpy as np
import utils.fileio
import utils.string_utils
from simulator.MATLAB.engine import Connector
from simulator.MATLAB.game_state import GameState

class Simulator:
    def __init__(self):
        # Attach to a running session.
        connector = Connector()
        assert connector.connect()
        self.eng = connector.engine()

        # Robot initialization.
        output_dir = utils.string_utils.to_folder_path(config.Simulator.MATLAB.OutputLocation)
        utils.fileio.mktree(output_dir)
        self.eng.workspace['output_dir'] = output_dir
        self.eng.simInit(nargout=0)

        # Plot initialization.
        self.__plot_initialized = False

    def close(self):
        pass

    def __state(self) -> GameState:
        state = GameState()
        state.from_matlab(self.eng.workspace['state'])
        return state

    def reset(self) -> GameState:
        self.eng.simReset(nargout=0)
        return self.__state()

    def step(self, action: np.ndarray) -> GameState:
        action = action[:, np.newaxis].tolist() # row order
        self.eng.workspace['action'] = matlab.double(action)
        self.eng.simStep(nargout=0)
        return self.__state()

    def stage(self) -> GameState:
        self.eng.simStage(nargout=0)
        return self.__state()

    def __plot_init(self) -> None:
        assert not self.__plot_initialized
        self.eng.simPlotInit(nargout=0)
        self.__plot_initialized = True

    def plot_reset(self) -> None:
        if not self.__plot_initialized:
            self.__plot_init()
        self.eng.simPlotReset(nargout=0)

    def plot_step(self) -> None:
        if not self.__plot_initialized:
            self.__plot_init()
            self.plot_reset()
        self.eng.simPlotStep(nargout=0)
        