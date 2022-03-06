import config
import os
os.add_dll_directory(config.Simulator.Gym.MujocoLibPath)
import gym
import numpy as np
from gym.spaces import Dict, Discrete
from simulator.gym.game import Game
from simulator.gym.game_state import GameState

class Simulator:
    def __init__(self):
        self.env = gym.make(config.Simulator.Gym.Environment)

        # Analyse state space
        if isinstance(self.env.observation_space, Dict):
            self.dim_state = 0
            for v in self.env.observation_space.spaces.values():
                self.dim_state += v.shape[0]
        else:
            self.dim_state = self.env.observation_space.shape[0]

        # Analyse action space
        self.action_discrete = isinstance(self.env.action_space, Discrete)
        if self.action_discrete:
            self.dim_action = 1
            self.n_action = self.env.action_space.n
        else:
            self.dim_action = self.env.action_space.shape[0]

    def close(self):
        self.env.close()

    def reset(self) -> GameState:
        state = self.env.reset()
        game_state = GameState()
        game_state.from_reset(self.dim_state, self.dim_action, state)
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