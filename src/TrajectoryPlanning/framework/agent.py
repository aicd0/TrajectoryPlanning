import numpy as np
import os
import utils.fileio
import utils.print
import utils.string_utils
from abc import abstractmethod
from envs.game_state import GameStateBase
from framework.configuration import Configuration
from framework.plot import PlotManager
from framework.replay_buffer import ReplayBuffer

replay_buffer_file = 'replay_buffer.npz'

class AgentBase:
    __agent_number = {}

    def __init__(self, dim_state: int, dim_action: int, model_group: str, name: str = None) -> None:
        if name is None:
            cls_name = self.__class__.__name__
            if cls_name in AgentBase.__agent_number:
                AgentBase.__agent_number[cls_name] += 1
            else:
                AgentBase.__agent_number[cls_name] = 1
            self.name = 'agent_' + cls_name + '_' + str(AgentBase.__agent_number[cls_name])
        else:
            self.name = name

        self.configs = Configuration(self.name)
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.model_group = model_group

        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer(self.configs)

        # Plots.
        self.plot_manager = PlotManager()

    @abstractmethod
    def sample_action(self, state: GameStateBase, deterministic: bool) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def learn(self):
        raise NotImplementedError()

    def save(self, path: str) -> None:
        path = utils.string_utils.to_folder_path(path)
        utils.fileio.mktree(path)

        # Save model.
        self._save(path)

        # Save replay buffer.
        np.savez(path + replay_buffer_file, data=np.array(self.replay_buffer.to_list(), dtype=object))

    @abstractmethod
    def _save(self, path: str) -> None:
        raise NotImplementedError()

    def load(self, path: str, enable_learning: bool=True) -> bool:
        path = utils.string_utils.to_folder_path(path)

        if not os.path.exists(path):
            return False
        
        self._load(path)

        # [optional] Load replay buffer.
        if enable_learning:
            obj = np.load(path + replay_buffer_file, allow_pickle=True)['data'].tolist()
            self.replay_buffer = ReplayBuffer.from_list(obj, self.configs)
        
        utils.print.put('Agent loaded')
        return True
        
    @abstractmethod
    def _load(self, path: str) -> None:
        raise NotImplementedError()