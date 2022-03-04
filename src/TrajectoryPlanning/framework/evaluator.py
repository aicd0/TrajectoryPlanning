import config
import matplotlib.pyplot as plt
import numpy as np
import utils.fileio
import utils.string_utils
from simulator.targets import Game, Simulator

class Evaluator(object):
    def __init__(self, simulator: Simulator, max_episodes: int):
        self.sim = simulator
        self.max_episodes = max_episodes
        self.output_path = utils.string_utils.to_folder_path(config.Evaluator.OutputLocation)
        self.rewards = np.array([], dtype=config.DataType.Numpy).reshape(max_episodes, 0)
        self.reward_steps = []

    def __call__(self, policy, visualize=False, save=True, step=None):
        state = self.sim.reset()
        game = Game()
        step_rewards = []
        episode_rewards = []
    
        for _ in range(self.max_episodes):
            state = self.sim.reset()
            game.reset()
            if visualize: self.sim.plot_reset()
            done = False
            iteration = 0
            episode_reward = 0

            while not done and iteration < config.Evaluator.MaxIterations:
                iteration += 1
                action = policy(state)
                state = self.sim.step(action)
                reward, done = game.update(action, state)
                episode_reward += reward
                step_rewards.append(reward)
                if visualize: self.sim.plot_step()
            
            episode_rewards.append(episode_reward)

        step_rewards = np.array(step_rewards, dtype=config.DataType.Numpy)
        episode_rewards = np.array(episode_rewards, dtype=config.DataType.Numpy).reshape(-1, 1)
        
        if save:
            assert not step is None
            self.rewards = np.hstack([self.rewards, episode_rewards])
            self.reward_steps.append(step)
            self.save_fig()
        
        res = {}
        if not step is None: res['Step'] = step
        res['EpRwd'] = np.mean(episode_rewards)
        res['EpRwdStd'] = np.std(episode_rewards)
        res['StpRwd'] = np.mean(step_rewards)
        res['StpRwdStd'] = np.std(step_rewards)
        return res

    def save_fig(self):
        utils.fileio.mktree(self.output_path)
        y = np.mean(self.rewards, axis=0)
        yerr = np.std(self.rewards, axis=0)
        _, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Step')
        plt.ylabel('Reward')
        ax.errorbar(self.reward_steps, y, yerr=yerr, fmt='-o')
        plt.savefig(self.output_path + 'result.png')
        plt.close()