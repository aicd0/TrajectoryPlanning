import config
import matplotlib.pyplot as plt
import numpy as np
import os
import utils.fileio
import utils.string_utils
from framework.configuration import Configuration

class Plot:
    def __init__(self, plot_name: str, title: str, label: str, window: int) -> None:
        self.plot_name = plot_name
        self.title = title
        self.label = label
        self.window = window
        self.y = [[]]
        self.y_avg = []
        self.y_std = []
        self.y_avg_win = []
        self.y_std_win = []
        self.x_epoch = []
        self.x_step = []
    
    def push(self, x) -> None:
        self.y[-1].append(x)

    def stage(self, epoch: int, step: int) -> None:
        if (len(self.x_epoch) > 0 and epoch <= self.x_epoch[-1]) or (len(self.x_step) > 0 and step <= self.x_step[-1]):
            raise Exception()
        
        batch = self.y[-1]
        if len(batch) <= 0:
            return

        self.x_step.append(step)
        self.x_epoch.append(epoch)
        self.y_avg.append(np.mean(self.y[-1]))
        self.y_std.append(np.std(self.y[-1]))

        win_begin = max(0, len(self.x_epoch) - self.window)
        y_window = np.array(self.y[win_begin:])
        y_avg_window = np.array(self.y_avg[win_begin:])
        self.y_avg_win.append(np.mean(y_avg_window))
        self.y_std_win.append(np.std(y_window))

        self.y.append([])

    def plot(self, path: str, configs: Configuration) -> None:
        width = configs.get(config.Evaluator.Figure.Width_)
        height = configs.get(config.Evaluator.Figure.Height_)
        dpi = configs.get(config.Evaluator.Figure.DPI_)
        filepath = path + self.plot_name + '.png'

        plt.figure(figsize=(width, height))
        plt.plot(self.x_step, self.y_avg_win)
        plt.title(self.title)
        plt.xlabel('Step')
        plt.ylabel(self.label)
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi)
        plt.close()

    def save(self, path: str) -> None:
        np.savez(path + self.plot_name + '.npz',
            title=self.title,
            plot_name=self.plot_name,
            label=self.label,
            window=self.window,
            y=np.array(self.y, dtype=object),
            y_avg=self.y_avg,
            y_std=self.y_std,
            y_avg_win=self.y_avg_win,
            y_std_win=self.y_std_win,
            x_epoch=self.x_epoch,
            x_step=self.x_step)

    def load(self, filepath: str) -> bool:
        checkpoint = np.load(filepath, allow_pickle=True)
        self.plot_name = str(checkpoint['plot_name'])
        self.title = str(checkpoint['title'])
        self.label = str(checkpoint['label'])
        self.window = int(checkpoint['window'])
        self.y = checkpoint['y'].tolist()
        self.y_avg = checkpoint['y_avg'].tolist()
        self.y_std = checkpoint['y_std'].tolist()
        self.y_avg_win = checkpoint['y_avg_win'].tolist()
        self.y_std_win = checkpoint['y_std_win'].tolist()
        self.x_epoch = checkpoint['x_epoch'].tolist()
        self.x_step = checkpoint['x_step'].tolist()

class PlotManager:
    def __init__(self) -> None:
        self.plots: dict[str, Plot] = {}

    def create_plot(self, plot_name: str, title: str, label: str, window: int=1) -> Plot:
        if not plot_name in self.plots:
            self.plots[plot_name] = Plot(plot_name, title, label, window)
        return self.plots[plot_name]

    def get_plot(self, plot_name: str) -> Plot:
        if not plot_name in self.plots:
            raise Exception('Plot not found')
        return self.plots[plot_name]

    def push(self, plot_name: str, val) -> None:
        if not plot_name in self.plots:
            raise Exception('Plot not found')
        self.plots[plot_name].push(val)

    def stage(self, epoch: int, step: int) -> None:
        for plot in self.plots.values():
            plot.stage(epoch, step)

    def plot(self, path: str, configs: Configuration):
        utils.fileio.mktree(path)
        plt.ioff()
        for plot in self.plots.values():
            plot.plot(path, configs)

    def save(self, path: str):
        utils.fileio.mktree(path)
        for plot in self.plots.values():
            plot.save(path)

    def load(self, path: str):
        for filename in os.listdir(path):
            plot_name = utils.string_utils.to_display_name(filename)
            plot = self.create_plot(plot_name, '_', '_')
            plot.load(path + filename)