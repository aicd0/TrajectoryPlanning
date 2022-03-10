import config
import matplotlib.pyplot as plt
import numpy as np
import utils.fileio
import utils.print
import utils.string_utils
from framework.ddpg import Agent

checkpoint_file = 'checkpoint.npz'

class Evaluator(object):
    def __init__(self, agent: Agent=None):
        self.__agent = agent
        self.__checkpoint_path = utils.string_utils.to_folder_path(config.Evaluator.CheckpointLocation)
        self.__output_path = utils.string_utils.to_folder_path(config.Evaluator.OutputLocation)
        self.__iterations = 0
        self.__epoch_step_rewards = []
        self.__last_plot_epoch = 0

        self.__epoches = 0
        self.__steps = 0
        self.__max_save_val = None
        self.__step_rewards = []
        self.__x_step = []
        self.__x_epoch = []
        self.__y_epoch_reward = []
        self.__y_step_reward_avg = []
        self.__y_step_reward_std = []
        self.__y_win_epoch_reward_avg = []
        self.__y_win_epoch_reward_std = []
        self.__y_win_step_reward_avg = []
        self.__y_win_step_reward_std = []

    def get_epoch(self) -> int:
        return self.__epoches

    def get_iteration(self) -> int:
        return self.__iterations

    def get_step(self) -> int:
        return self.__steps

    def step(self, reward: float) -> None:
        self.__epoch_step_rewards.append(reward)
        self.__iterations += 1
        self.__steps += 1

    def epoch(self, save: bool=True) -> None:
        epoch_step_rewards = np.array(self.__epoch_step_rewards, dtype=config.DataType.Numpy)
        self.__epoches += 1
        self.__iterations = 0
        self.__epoch_step_rewards = []
        self.__step_rewards.append(epoch_step_rewards)
        self.__x_step.append(self.__steps)
        self.__x_epoch.append(self.__epoches)

        self.__y_epoch_reward.append(np.sum(epoch_step_rewards))
        self.__y_step_reward_avg.append(np.mean(epoch_step_rewards))
        self.__y_step_reward_std.append(np.std(epoch_step_rewards))

        win_size = config.Evaluator.EpochWindowSize
        epoch_reward_window = np.array(self.__y_epoch_reward[len(self.__y_epoch_reward) - win_size:])
        step_reward_window = np.concatenate(self.__step_rewards[len(self.__step_rewards) - win_size:])
        self.__y_win_epoch_reward_avg.append(np.mean(epoch_reward_window))
        self.__y_win_epoch_reward_std.append(np.std(epoch_reward_window))
        self.__y_win_step_reward_avg.append(np.mean(step_reward_window))
        self.__y_win_step_reward_std.append(np.std(step_reward_window))

        need_plot = self.get_epoch() - self.__last_plot_epoch >= config.Evaluator.Figure.MaxSaveEpochInterval

        if save and not self.__agent is None:
            save_val = self.__y_win_epoch_reward_avg[-1]
            if self.__max_save_val is None or save_val > self.__max_save_val:
                self.__max_save_val = save_val
                need_plot = True
                self.save()

        if need_plot:
            self.__last_plot_epoch = self.get_epoch()
            self.plot()

    def summary(self, shortterm: bool=False) -> dict:
        res_dict = {
            'Ep': self.get_epoch(),
            'Iter': self.get_iteration(),
            'Stp': self.get_step(),
        }
        if shortterm:
            if len(self.__y_epoch_reward): res_dict['REp'] = self.__y_epoch_reward[-1]
            if len(self.__y_step_reward_avg): res_dict['RStp'] = self.__y_step_reward_avg[-1]
            if len(self.__y_step_reward_std): res_dict['RStpStd'] = self.__y_step_reward_std[-1]
        else:
            if len(self.__y_win_epoch_reward_avg): res_dict['REp'] = self.__y_win_epoch_reward_avg[-1]
            if len(self.__y_win_epoch_reward_std): res_dict['REpStd'] = self.__y_win_epoch_reward_std[-1]
            if len(self.__y_win_step_reward_avg): res_dict['RStp'] = self.__y_win_step_reward_avg[-1]
            if len(self.__y_win_step_reward_std): res_dict['RStpStd'] = self.__y_win_step_reward_std[-1]
        return res_dict

    def plot(self) -> None:
        utils.fileio.mktree(self.__output_path)
        width = config.Evaluator.Figure.Width
        height = config.Evaluator.Figure.Height
        dpi = config.Evaluator.Figure.DPI

        plt.figure(figsize=(width, height))
        plt.plot(self.__x_step, self.__y_win_epoch_reward_avg)
        # plt.errorbar(x=self.__x_step, y=self.__y_win_epoch_reward_avg, yerr=self.__y_win_epoch_reward_std) # fmt='-o'
        plt.title('Epoch rewards')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.savefig(self.__output_path + 'reward_epoch.png', dpi=dpi)
        plt.close()
        
        plt.figure(figsize=(width, height))
        plt.plot(self.__x_step, self.__y_win_step_reward_avg)
        # plt.errorbar(x=self.__x_step, y=self.__y_win_step_reward_avg, yerr=self.__y_win_step_reward_std) # fmt='-o'
        plt.title('Step rewards')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.savefig(self.__output_path + 'reward_step.png', dpi=dpi)
        plt.close()

    def save(self) -> None:
        if self.__agent is None:
            raise RuntimeError('Agent required.')
        self.__agent.save(self.__checkpoint_path)

        np.savez(self.__checkpoint_path + checkpoint_file,
            epoches=self.__epoches,
            steps=self.__steps,
            max_save_val=self.__max_save_val,
            step_rewards=np.array(self.__step_rewards, dtype=object),
            x_step=self.__x_step,
            x_epoch=self.__x_epoch,
            y_epoch_reward=self.__y_epoch_reward,
            y_step_reward_avg=self.__y_step_reward_avg,
            y_step_reward_std=self.__y_step_reward_std,
            y_win_epoch_reward_avg=self.__y_win_epoch_reward_avg,
            y_win_epoch_reward_std=self.__y_win_epoch_reward_std,
            y_win_step_reward_avg=self.__y_win_step_reward_avg,
            y_win_step_reward_std=self.__y_win_step_reward_std)

        utils.print.put('Checkpoint saved at %f' % self.__max_save_val)

    def load(self) -> None:
        if self.__agent is None:
            raise RuntimeError('Agent required.')
        if not self.__agent.load(self.__checkpoint_path):
            raise RuntimeError('Failed to load agent.')
        
        checkpoint = np.load(self.__checkpoint_path + checkpoint_file, allow_pickle=True)
        self.__epoches = int(checkpoint['epoches'])
        self.__steps = int(checkpoint['steps'])
        self.__max_save_val = float(checkpoint['max_save_val'])
        self.__step_rewards = checkpoint['step_rewards'].tolist()
        self.__x_step = checkpoint['x_step'].tolist()
        self.__x_epoch = checkpoint['x_epoch'].tolist()
        self.__y_epoch_reward = checkpoint['y_epoch_reward'].tolist()
        self.__y_step_reward_avg = checkpoint['y_step_reward_avg'].tolist()
        self.__y_step_reward_std = checkpoint['y_step_reward_std'].tolist()
        self.__y_win_epoch_reward_avg = checkpoint['y_win_epoch_reward_avg'].tolist()
        self.__y_win_epoch_reward_std = checkpoint['y_win_epoch_reward_std'].tolist()
        self.__y_win_step_reward_avg = checkpoint['y_win_step_reward_avg'].tolist()
        self.__y_win_step_reward_std = checkpoint['y_win_step_reward_std'].tolist()

        utils.print.put('Checkpoint loaded at %f' % self.__max_save_val)