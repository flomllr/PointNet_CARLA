import torch.utils.data as data
import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler
import numpy as np

class BalancedIterator(object):
  """
  Iterates over left, s_left, s_right, right in a balanced way
  by returning 1/4 of batch size from each bin while maintaining order

  Arguments:
    left, s_left, s_right, right (sequences): List containing indices of elements of respective bin
    batch_size (number): Number of elements to be returned
  """
  def __init__(self, batch_size, left, s_left, s_right=None, right=None):
    self.only_left = (s_right is None) or (right is None)
    
    self.left = left
    self.s_left = s_left
    print(left.size, s_left.size)
    self.length = len(self.left) + len(self.s_left)
    if not(self.only_left):
      print(s_right.size, right.size)
      self.s_right = s_right
      self.right = right
      self.length += len(self.s_right) + len(self.right)

    self.i = 0
    self.batch_size = batch_size
    print("Sampler length: ", self.length)

  def __iter__(self):
    return self
  
  def __next__(self):
    return self.next()

  def next(self):
    # Distinguish between only left and left+right
    if self.only_left:
      if self.i >= self.length / 2:
        raise StopIteration
      else:
        elem = []
        for i in range(self.i, self.i + self.batch_size / 2):
          elem.append(self.left[i % self.left.size])
          elem.append(self.s_left[i % self.s_left.size])
          self.i += 1
        return elem
  
    else:
      if self.i >= self.length / 4:
        raise StopIteration
      else:
        elem = []
        for i in range(self.i, self.i + self.batch_size / 4):
          elem.append(self.left[i % self.left.size])
          elem.append(self.s_left[i % self.s_left.size])
          elem.append(self.s_right[i % self.s_right.size])
          elem.append(self.right[i % self.right.size])
          self.i += 1
        return elem


class BalancedSteeringSampler(Sampler):
  """
  Organizes given list of steering angles into 4 bins
  Returns a list of indices of site batch_size consisting of 1/4 examples from each bin per iteration
  
  Arguments:
    steerings (sequence): List of steering angles from which should be sampled
    batch_size (number): Number of indices being returned on each iteration
  """
  def __init__(self, steerings, batch_size, only_left=False):
    # Convert to numpy array to do filtering
    steerings = np.array(steerings)
    indices = np.arange(steerings.size)
    self.only_left = only_left
    # Filter steering array and save the respective indices in 4 bins
    if self.only_left:
      self.left = indices[steerings < -0.05]
      self.s_left = indices[steerings >= -0.05]
    else:
      self.left = indices[steerings < -0.03]
      self.s_left = indices[np.logical_and((steerings >= -0.03),(steerings < 0))]
      self.s_right = indices[np.logical_and((steerings >= 0), (steerings < 0.03))]
      self.right = indices[steerings >= 0.03]
    self.batch_size = batch_size

  def __iter__(self):
    if self.only_left:
      return BalancedIterator(self.batch_size, self.left, self.s_left, )
    else:
      return BalancedIterator(self.batch_size, self.left, self.s_left, self.s_right, self.right)
    

  def __len__(self):
    if self.only_left:
      return (len(self.left) + len(self.s_left)) / self.batch_size
    else:
      return (len(self.left) + len(self.s_left) + len(self.s_right) + len(self.right)) / self.batch_size
    # To iterate over each example at least once, you have to iterate 4*len(largest_bin)
    #return max(len(self.left) + len(self.s_left) + len(self.s_right) + len(self.right)) * 4


class WeightedSteeringSampler(Sampler):
  """
  New approach for a balanced sampler:
  Draw from each bin with a certain probability
  """
  def __init__(self, steerings, num_samples, replacement=True, only_left=False):
    self.steerings = np.array(steerings)
    self.num_samples = num_samples
    self.replacement = replacement
    self.only_left = only_left
    self.threshold_left = -0.03
    self.threshold_right = 0.03

    self.weights = torch.as_tensor(self.define_weights(), dtype=torch.double)

  def define_weights(self):
    # Assign each example to a class
    # (straight = 0, left = 1, right = 2)
    classes = np.zeros(self.steerings.size, dtype='uint8')
    classes = classes + np.where(self.steerings < self.threshold_left, 1, 0)
    if not self.only_left: # if there are only left examples in the training set, assign everything above -0.03 to straight
      classes = classes + np.where(self.steerings > self.threshold_right, 2, 0)

    # Count elements of each class
    num = [
      sum(classes == 0),  # number of straight examples
      sum(classes == 1),  # number of left examples
    ]
    if not self.only_left:
      num.append(sum(classes == 2))   # number of right examples)
    print("Sums", num)
    # Calculate weight of each class
    weight_per_class = [1.0 / num[c] for c in range(len(num))]
    print("Weight_per_class", weight_per_class)

    # Define weights for each example
    weights = [weight_per_class[classes[el]] for el in classes]
    
    return weights

  def __iter__(self):
    return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

  def __len__(self):
    return self.num_samples