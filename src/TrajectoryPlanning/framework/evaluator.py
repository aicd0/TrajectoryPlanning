import config
import numpy as np
import utils.fileio
import utils.print
import utils.string_utils
from framework.agent import AgentBase
from framework.configuration import Configuration

global_checkpoint_file = 'global.npz'

class Evaluator:
    def __init__(self, agent: AgentBase):
        self.configs = Configuration('evaluator_' + agent.name)
        window = self.configs.get(config.Evaluator.EpochWindowSize_)

        self.agent = agent
        self.save_dir = utils.string_utils.to_folder_path(config.Evaluator.SaveDir + agent.name)
        self.model_dir = utils.string_utils.to_folder_path(self.save_dir + 'model')
        self.plot_data_dir = utils.string_utils.to_folder_path(self.save_dir + 'plot_data')
        self.figure_dir = utils.string_utils.to_folder_path(config.Evaluator.FigureDir + agent.name)
        self.plot_manager = agent.plot_manager
        self.epoches = 0
        self.steps = 0
        self.iterations = 0
        self.epoch_reward = 0
        self.last_save_step = 0
        self.last_plot_epoch = 0
        self.max_save_val = None

        self.plot_epoch_reward = 'epoch_rewards'
        self.plot_step_reward = 'step_rewards'
        self.plot_manager.create_plot(self.plot_epoch_reward, 'Epoch rewards', 'Reward', window=window)
        self.plot_manager.create_plot(self.plot_step_reward, 'Step rewards', 'Reward', window=window)

    def step(self, reward: float) -> None:
        self.iterations += 1
        self.steps += 1
        self.plot_manager.push(self.plot_step_reward, reward)
        self.epoch_reward += reward

    def epoch(self, allow_save: bool = True) -> None:
        self.epoches += 1
        self.iterations = 0

        self.plot_manager.push(self.plot_epoch_reward, self.epoch_reward)
        self.epoch_reward = 0

        self.plot_manager.stage(self.epoches, self.steps)

        min_save_step_interval = self.configs.get(config.Evaluator.MinSaveStepInterval_)
        allow_save = allow_save and self.steps - self.last_save_step >= min_save_step_interval

        max_plot_epoch_interval = self.configs.get(config.Evaluator.Figure.MaxSaveEpochInterval_)
        need_plot = self.epoches - self.last_plot_epoch >= max_plot_epoch_interval
        
        if allow_save and not self.agent is None:
            save_val = self.plot_manager.get_plot(self.plot_epoch_reward).y_avg_win[-1]
            if self.max_save_val is None or save_val > self.max_save_val:
                self.max_save_val = save_val
                self.last_save_step = self.steps
                need_plot = True
                self.save()

        if need_plot:
            self.last_plot_epoch = self.epoches
            self.plot_manager.plot(self.figure_dir, self.configs)

    def summary(self, shortterm: bool=False) -> dict:
        res_dict = {
            'Ep': self.epoches,
            'Iter': self.iterations,
            'Stp': self.steps,
        }
        ep_rwd = self.plot_manager.get_plot(self.plot_epoch_reward)
        stp_rwd = self.plot_manager.get_plot(self.plot_step_reward)
        if shortterm:
            if len(ep_rwd.y_avg): res_dict['REp'] = ep_rwd.y_avg[-1]
            if len(stp_rwd.y_avg): res_dict['RStp'] = stp_rwd.y_avg[-1]
            if len(stp_rwd.y_std): res_dict['RStpStd'] = stp_rwd.y_std[-1]
        else:
            if len(ep_rwd.y_avg_win): res_dict['REp'] = ep_rwd.y_avg_win[-1]
            if len(ep_rwd.y_std_win): res_dict['REpStd'] = ep_rwd.y_std_win[-1]
            if len(stp_rwd.y_avg_win): res_dict['RStp'] = stp_rwd.y_avg_win[-1]
            if len(stp_rwd.y_std_win): res_dict['RStpStd'] = stp_rwd.y_std_win[-1]
        return res_dict

    def save(self) -> None:
        # Save model.
        self.agent.save(self.model_dir)

        # Save plot data.
        np.savez(self.save_dir + global_checkpoint_file,
            epoches=self.epoches,
            steps=self.steps,
            last_save_step=self.last_save_step,
            max_save_val=self.max_save_val)
        self.plot_manager.save(self.plot_data_dir)

        utils.print.put('Checkpoint saved at %f' % self.max_save_val)

    def load(self, enable_learning=True) -> None:
        # Load model.
        if not self.agent.load(self.model_dir, enable_learning=enable_learning):
            raise RuntimeError('Failed to load agent.')
        
        # Load plot data.
        checkpoint = np.load(self.save_dir + global_checkpoint_file)
        self.epoches = int(checkpoint['epoches'])
        self.steps = int(checkpoint['steps'])
        self.last_save_step = int(checkpoint['last_save_step'])
        self.max_save_val = float(checkpoint['max_save_val'])
        self.plot_manager.load(self.plot_data_dir)

        utils.print.put('Checkpoint loaded at %f' % self.max_save_val)