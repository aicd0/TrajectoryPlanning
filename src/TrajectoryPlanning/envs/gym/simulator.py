import config
import numpy as np
import os
import utils.platform
from .state import GymState
from envs.simulator import Simulator
from framework.configuration import global_configs as configs

# Import gym packages.
if utils.platform.is_windows():
    os.add_dll_directory(config.Environment.Gym.MujocoLibPath)
import gym
from gym.spaces import Discrete

class Gym(Simulator):
    def __init__(self):
        self.__env = gym.make(configs.get(config.Environment.Gym.Environment_))
        self.raw_state = None
        self.raw_reward = None
        self.raw_done = None

        # Determine action space
        self.action_discrete = isinstance(self.__env.action_space, Discrete)
        if self.action_discrete:
            self.__dim_action = 1
            self.n_action = self.__env.action_space.n
        else:
            assert len(self.__env.action_space.shape) == 1
            self.__dim_action = self.__env.action_space.shape[0]

    def _get_state(self) -> GymState:
        return GymState(self.raw_state, self.raw_reward, self.raw_done)

    def close(self):
        self.__env.close()

    def _reset(self) -> None:
        self.raw_state = self.__env.reset()
        self._state = None

    def _step(self, action: np.ndarray) -> None:
        if self.action_discrete:
            action = np.clip(int((action[0] + 1) / 2 * self.n_action), 0, self.n_action - 1)
        self.raw_state, self.raw_reward, self.raw_done, _ = self.__env.step(action)
        self._state = None

    def plot_reset(self) -> None:
        pass

    def plot_step(self) -> None:
        self.__env.render(mode='human')

    def dim_action(self) -> int:
        return self.__dim_action