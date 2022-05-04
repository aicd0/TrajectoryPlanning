import config
import matlab
import numpy as np
import utils.fileio
import utils.string_utils
from .engine import Connector
from .reward import MatlabReward
from .state import MatlabState
from envs.simulator import Simulator

class Matlab(Simulator):
    def __init__(self):
        self.__state = None
        self.__plot_initialized = False

        # Attach to a running session.
        connector = Connector()
        assert connector.connect()
        self.eng = connector.engine()

        # Robot initialization.
        output_dir = config.Environment.MATLAB.OutputDir
        utils.fileio.mktree(output_dir)
        self.eng.workspace['output_dir'] = output_dir
        self.eng.simInit(nargout=0)

        # Get dim_action.
        state = self.reset()
        self.__dim_action = len(state.joint_position)

    def close(self):
        pass

    def __get_state(self) -> MatlabState:
        if self.__state is None:
            self.__state = MatlabState()
            self.__state.from_matlab(self.eng.workspace['state'])
        return self.__state

    def __step_world(self) -> None:
        self.eng.simStep(nargout=0)
        self.__state = None

    def __step(self, joint_position: np.ndarray) -> None:
        action = joint_position[:, np.newaxis].tolist() # row order
        self.eng.workspace['action'] = matlab.double(action)
        self.__step_world()

    def reset(self) -> MatlabState:
        self.eng.simReset(nargout=0)
        return self.__get_state()

    def step(self, action: np.ndarray) -> MatlabState:
        action_amp = self.configs.get(config.Environment.MATLAB.ActionAmp_)
        last_position = self.__get_state().joint_position
        this_position = last_position + action * action_amp
        self.__step(this_position)
        new_state = self.__get_state()
        if new_state.collision:
            self.__step(last_position)
            new_state = self.__get_state()
            new_state.collision = True
        return new_state

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
        
    def dim_action(self) -> int:
        return self.__dim_action

    def reward(self) -> MatlabReward:
        return MatlabReward()