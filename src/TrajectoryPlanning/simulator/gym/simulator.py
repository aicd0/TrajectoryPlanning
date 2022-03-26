import os

# Requires to add mujoco lib path on Windows before we import gym.
import utils.platform
if utils.platform.is_windows():
    os.add_dll_directory(config.Simulator.Gym.MujocoLibPath)

import config
import gym
import numpy as np
from gym.spaces import Dict, Discrete
from simulator.gym.game import Game
from simulator.gym.game_state import GameState

class Simulator:
    def __init__(self):
        self.env = gym.make(config.Simulator.Gym.Environment)

        # Analyse action space
        self.action_discrete = isinstance(self.env.action_space, Discrete)
        if self.action_discrete:
            self.__dim_action = 1
            self.n_action = self.env.action_space.n
        else:
            self.__dim_action = self.env.action_space.shape[0]

    def close(self):
        self.env.close()

    def reset(self) -> GameState:
        state_raw = self.env.reset()
        game_state = GameState()
        game_state.from_reset(state_raw)
        return game_state

    def step(self, action: np.ndarray) -> GameState:
        if self.action_discrete:
            action = np.clip(int((action[0] + 1) / 2 * self.n_action), 0, self.n_action - 1)

        state, reward_raw, done, _ = self.env.step(action)
        game_state = GameState()
        game_state.from_step(state, reward_raw, done)
        return game_state

    def stage(self) -> GameState:
        raise Exception()

    def plot_reset(self) -> None:
        pass

    def plot_step(self) -> None:
        self.env.render(mode='human')

    def dim_action(self) -> int:
        return self.__dim_action