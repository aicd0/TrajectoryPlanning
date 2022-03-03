import matlab
import numpy as np
from framework.state import State
from simulator.engine import Connector

class Simulator:
    def __init__(self):
        # Attach to a running session.
        connector = Connector()
        assert connector.connect()
        self.eng = connector.engine()

        # Robot initialization.
        self.eng.simInit(nargout=0)

        # Plot initialization.
        self.__plot_initialized = False
        self.__plot_reset = False

    def __state(self) -> State:
        state = State()
        state.from_matlab(self.eng.workspace['state'])
        return state

    def reset(self) -> State:
        self.eng.simReset(nargout=0)
        return self.__state()

    def step(self, action: np.ndarray) -> State:
        action = action[:, np.newaxis].tolist() # row order
        self.eng.workspace['action'] = matlab.double(action)
        self.eng.simStep(nargout=0)
        return self.__state()

    def stage(self) -> State:
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
        