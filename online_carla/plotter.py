from visdom import Visdom

import numpy as np
import time
import torch

from scipy.spatial.transform import Rotation as R

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
        self.rotation = torch.Tensor(R.from_euler('zyz', [-90,0,0], degrees=True).as_dcm())
        print("Rotation", self.rotation)

    def plot_point_cloud(self, points, var_name):
      if(len(list(points.shape)) == 3):
        points = points.view(3,-1)
      points = points.transpose(1,0)
      points = torch.mm(points, self.rotation)

      # Rotate matrix to plot better

      print(points.shape)
      if var_name not in self.plots:
        self.plots[var_name] = self.viz.scatter(X=points, env=self.env, name="pointplot", opts=dict(
          markersize=1
        ))
        # print("YOU HAVE 10s to rotate the graph")
        # time.sleep(10)
      else:
        self.viz.scatter(X=points, env=self.env, name="pointplot", win=self.plots[var_name], opts=dict(
          markersize=1
        ))

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
