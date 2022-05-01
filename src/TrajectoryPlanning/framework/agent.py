import config
import numpy as np
import os
import utils.fileio
import utils.name
import utils.print
import utils.string_utils
from abc import abstractmethod
from envs.state import State
from framework.configuration import Configuration
from framework.plot import PlotManager
from framework.replay_buffer import ReplayBuffer

replay_buffer_file = 'replay_buffer.npz'

class Agent:
    __names = utils.name.Name('agent')

    def __init__(self, dim_state: int, dim_action: int, name) -> None:
        self.name = Agent.__names.get(self.__class__.__name__, name)
        self.configs = Configuration(self.name)
        self.save_dir = utils.string_utils.to_folder_path(config.Agent.SaveDir + self.name)
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.model_group = self.configs.get(config.Agent.ModelGroup_)

        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer(self.configs)

        # Plots.
        self.plot_manager = PlotManager()

    @abstractmethod
    def sample_action(self, state: State, deterministic: bool) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def learn(self):
        raise NotImplementedError()

    @abstractmethod
    def _save(self) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        utils.fileio.mktree(self.save_dir)

        # Save model.
        self._save()

        # Save replay buffer.
        np.savez(self.save_dir + replay_buffer_file, data=np.array(self.replay_buffer.to_list(), dtype=object))
        
    @abstractmethod
    def _load(self) -> None:
        raise NotImplementedError()

    def load(self, enable_learning: bool=True) -> bool:
        if not os.path.exists(self.save_dir):
            return False
        
        # Load model.
        self._load()

        # [optional] Load replay buffer.
        if enable_learning:
            obj = np.load(self.save_dir + replay_buffer_file, allow_pickle=True)['data'].tolist()
            self.replay_buffer = ReplayBuffer.from_list(obj, self.configs)
        
        utils.print.put('Agent loaded (' + self.name + ')')
        return True

from .algorithm import ddpg
from .algorithm import sac

def create_agent(algorithm: str, *arg, **kwarg) -> Agent:
    if algorithm == 'ddpg':
        return ddpg.DDPG(*arg, **kwarg)
    elif algorithm == 'sac':
        return sac.SAC(*arg, **kwarg)
    else:
        raise Exception('Unrecognized algorithm')