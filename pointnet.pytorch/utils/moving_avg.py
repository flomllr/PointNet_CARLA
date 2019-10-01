import numpy as np

def running_mean(x, N):
  np.convolve(x, np.ones((N,))/N, mode='same')