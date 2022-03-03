import gym
import numpy as np
from simulator.Pendulum_v1.game_state import GameState

class Simulator:
    def __init__(self):
        self.env = gym.make('Pendulum-v1')

    def reset(self) -> GameState:
        state = self.env.reset()
        game_state = GameState()
        game_state.from_reset(self.env.observation_space, self.env.action_space, state)
        return game_state

    def step(self, action: np.ndarray) -> GameState:
        state, reward, done, _ = self.env.step(action)
        game_state = GameState()
        game_state.from_step(state, reward, done)
        return game_state

    def stage(self) -> GameState:
        raise Exception()

    def plot_reset(self) -> None:
        pass

    def plot_step(self) -> None:
        self.env.render(mode='human')