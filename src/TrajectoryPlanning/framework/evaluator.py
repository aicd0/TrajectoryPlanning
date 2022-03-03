import config
import numpy as np
import matplotlib.pyplot as plt
from simulator.simulator import Game, Simulator

class Evaluator(object):
    def __init__(self, simulator: Simulator, max_episodes: int):
        self.sim = simulator
        self.max_episodes = max_episodes
        self.save_path = '../../outputs/Results'
        self.results = np.array([]).reshape(max_episodes, 0)
        self.x = []

    def __call__(self, policy, visualize=False, save=True, step=None):
        state = self.sim.reset()
        game = Game()
        result = []
    
        for _ in range(self.max_episodes):
            state = self.sim.reset()
            game.reset()
            if visualize: self.sim.plot_reset()
            done = False
            iteration = 0
            total_reward = 0

            while not done and iteration < config.Evaluator.MaxIterations:
                iteration += 1
                action = policy(state)
                state = self.sim.step(action)
                reward, game_over, stage_over = game.update(action, state)
                total_reward += reward

                if visualize: self.sim.plot_step()

                if game_over:
                    break
                elif stage_over:
                    state = self.sim.stage()

            result.append(total_reward)

        result = np.array(result).reshape(-1, 1)
        
        if save:
            assert not step is None
            self.results = np.hstack([self.results, result])
            self.x.append(step)
            self.save_fig()
        
        return np.mean(result)

    def save_fig(self):
        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(self.x, y, yerr=error, fmt='-o')
        plt.savefig(self.save_path + '/result.png')