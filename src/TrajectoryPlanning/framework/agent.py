import config
import json
import numpy as np
import os
import utils.fileio
import utils.print
import utils.string_utils
from abc import abstractmethod
from framework.configuration import Configuration
from framework.replay_buffer import ReplayBuffer
from simulator.game_state import GameStateBase

replay_buffer_file = 'replay_buffer.txt'

class AgentBase:
    __agent_number = {}

    def __init__(self, dim_state: int, dim_action: int, model_base: str, name: str = None) -> None:
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
        self.model_base = model_base

        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer(self.configs)

    @abstractmethod
    def sample_action(self, state: GameStateBase) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def learn(self):
        raise NotImplementedError()

    def save(self, path: str) -> None:
        path = utils.string_utils.to_folder_path(path)
        utils.fileio.mktree(path)

        self._save(path)

        # Save replay buffer.
        replay_buffer_checkpoint_file_path = path + replay_buffer_file
        replay_buffer_checkpoint_temp_file_path = replay_buffer_checkpoint_file_path + '.tmp'
        with open(replay_buffer_checkpoint_temp_file_path, 'w') as f:
            tmp = self.replay_buffer.to_serializable()
            json.dump(tmp, f)
        if os.path.exists(replay_buffer_checkpoint_file_path):
            os.remove(replay_buffer_checkpoint_file_path)
        os.rename(replay_buffer_checkpoint_temp_file_path, replay_buffer_checkpoint_file_path)

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
            capacity = self.configs.get(config.Train.DDPG.FieldReplayBuffer)
            with open(path + replay_buffer_file, 'r') as f:
                raw_obj = json.load(f)
                self.replay_buffer = ReplayBuffer.from_serializable(raw_obj, capacity)
        
        utils.print.put('Agent loaded')
        return True
        
    @abstractmethod
    def _load(self, path: str) -> None:
        raise NotImplementedError()