from visdom import Visdom

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.moving = {}
        self.moving_counter = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array(
                [y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')

    def plot_moving_avg(self, var_name, split_name, title_name, x, y, window=10):
        if var_name + split_name not in self.moving:
            self.moving[var_name + split_name] = np.zeros(window)
            self.moving_counter[var_name + split_name] = 0

        self.moving_counter[var_name + split_name] += 1
        index = self.moving_counter[var_name + split_name]
        self.moving[var_name + split_name][index % window] = y
        
        if self.moving_counter[var_name + split_name] >= window:
            avg = self.moving[var_name + split_name].mean()
            if var_name not in self.plots:
                self.plots[var_name] = self.viz.line(X=np.array([avg, avg]), Y=np.array([y, y]), env=self.env, opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel='Epochs',
                    ylabel=var_name
                ))
            else:
                self.viz.line(X=np.array([x]), Y=np.array(
                    [avg]), env=self.env, win=self.plots[var_name], name=split_name, update='append')
