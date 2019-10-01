from __future__ import print_function
import torch.utils.data as data
from torch._utils import _accumulate
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
import glob
import random
import pdb
from PIL import Image

class ClusterDataset(data.Dataset):
  """
  Loads data to train deepcluster on
  """
  def __init__(self,
         root):
    self.root = root
    self.fns = sorted(glob.glob(os.path.join(root, "*.png")))
    self.data = np.zeros((0,3))
    for path in self.fns:
      img = Image.open(path)
      np_img = np.array(img)
      print(np_img.shape)
      np_img = np_img.reshape(-1, 3)
      self.data = np.append(self.data, np_img, axis=0)
    print(self.data.shape)
    print("Data:", self.data)
    #pdb.set_trace()

  def __getitem__(self, index):
    item = self.data[index].astype(np.float32)
    item = torch.from_numpy(item)
    print("Returning", item)
    return item, 0

  def __len__(self):
    return self.data.shape[0]
