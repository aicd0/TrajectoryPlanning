import config
import gym
import numpy as np
from gym.spaces import Discrete
from simulator.gym.game_state import GameState

class Simulator:
    def __init__(self):
        self.env_name = config.Simulator.Gym.Envoronment
        self.env = gym.make(self.env_name)

        self.dim_state = self.env.observation_space.shape[0]
        self.action_discrete = len(self.env.action_space.shape) == 0

        if self.action_discrete:
            self.dim_action = 1
            self.action_n = self.env.action_space.n
        else:
            self.dim_action = self.env.action_space.shape[0]

    def close(self):
        self.env.close()

    def __to_real_reward(self, state, reward, done):
        if self.env_name == 'CartPole-v0':
            return -1 if done else 1
        if self.env_name == 'CartPole-v1':
            return -1 if done else 1
        if self.env_name == 'Pendulum-v1':
            return reward
        return reward

    def reset(self) -> GameState:
        state = self.env.reset()
        game_state = GameState()
        game_state.from_reset(self.dim_state, self.dim_action, state)
        return game_state

    def step(self, action: np.ndarray) -> GameState:
        if self.action_discrete:
            action = np.clip(int((action[0] + 1) / 2 * self.action_n), 0, self.action_n - 1)

        state, reward, done, _ = self.env.step(action)
        reward = self.__to_real_reward(state, reward, done)
        game_state = GameState()
        game_state.from_step(state, reward, done)
        return game_state

    def stage(self) -> GameState:
        raise Exception()

    def plot_reset(self) -> None:
        pass

    def plot_step(self) -> None:
        self.env.render(mode='human')