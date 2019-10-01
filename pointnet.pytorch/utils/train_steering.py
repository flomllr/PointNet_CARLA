''' This file is a copy of train_classification.py with slight modifications '''
####
# Modifications to use our Data and use PointNetReg classifier to predict steering angle
####

from __future__ import print_function
import plotter
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
from pointnet.dataset import SteeringDataset, SteeringDatasetEpisodes, SteeringDatasetNested, SteeringDatasetCombined, random_split  # changed to SteeringDataset
from pointnet.sampler import BalancedSteeringSampler, WeightedSteeringSampler # added
# changed to PointNetReg
from pointnet.model import PointNetReg, PointNetReg2, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import logging
np.set_printoptions(threshold=np.inf)
logging.basicConfig(filename='training.log', level=logging.DEBUG)
global plotter
pred_file = open("pred_file.txt", "w+")
cuda_available = torch.cuda.is_available()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=64, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=1000, help='number of points per cloud')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='steering', help="dataset type (steering)")
parser.add_argument('--feature_transform', action='store_true', default=False, help="use feature transform")
parser.add_argument('--only_left', action='store_true', help='name of the visdom plot')
parser.add_argument('--sampler', type=str, help='should the custom balanced sampler be used?', default='weighted')
parser.add_argument('--running_mean', type=int, help='Size of the running mean window used to smoothen steering angles')
parser.add_argument('--use_whole_dataset', action='store_true', help='Determine whether dataset should be split into training and test')
parser.add_argument('--steering_indicator', action="store_true", help='Should an additional steering indicator be passed to the model?')
parser.add_argument('--data_augmentation', action='store_true', help='Data augmentation during training?')
parser.add_argument('--no_norm', action='store_true', help='Data augmentation during training?')
parser.add_argument('--num_steering_indicators', type=int, help='Number of steering indicators to use (2 = left/right, 3 = left/straight/right)', default=2)
parser.add_argument('--no_visdom', action='store_true', help='Don\'t plot to visdom')


opt = parser.parse_args()
opt.visdom = os.path.basename(opt.outf)
print(opt)
plot_enabled = not opt.no_visdom
if plot_enabled:
  print("Vidsom is enabled")
else:
  print("Visdom is not enabled")

if plot_enabled:
  plt = plotter.VisdomLinePlotter(env_name=opt.visdom)

def blue(x): return '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'steering':
  print("Steering")
  full_dataset = SteeringDataset(
      root=opt.dataset,
      npoints=opt.num_points,
      running_mean=opt.running_mean,
      steering_indicator=opt.steering_indicator,
      data_augmentation=opt.data_augmentation,
      no_norm=opt.no_norm,
      num_steering_indicators=opt.num_steering_indicators
  )
elif opt.dataset_type == 'steering_episodes':
  print("Steering episodes")
  full_dataset = SteeringDatasetEpisodes(
    root=opt.dataset,
    npoints=opt.num_points,
    running_mean=opt.running_mean,
    episodes=range(20), # only episode 0-4
    steering_indicator=opt.steering_indicator,
    data_augmentation=opt.data_augmentation,
    no_norm=opt.no_norm,
    num_steering_indicators=opt.num_steering_indicators

  )
elif opt.dataset_type == 'steering_nested':
  print("Steering nested")
  full_dataset = SteeringDatasetNested(
    root=opt.dataset,
    npoints=opt.num_points,
    running_mean=opt.running_mean,
    levels=2,
    steering_indicator=opt.steering_indicator,
    data_augmentation=opt.data_augmentation,
    no_norm=opt.no_norm,
    num_steering_indicators=opt.num_steering_indicators

  )
elif opt.dataset_type == 'steering_combined':
  print("Steering combination")
  datasets = [
    ('/usr/prakt/s0050/PointCloudProcessed/lidar_town02', 2),
    ('/usr/prakt/s0050/PointCloudProcessed/lidar_town01', 2),
  ]
  full_dataset = SteeringDatasetCombined(
    roots=datasets,
    npoints=opt.num_points,
    running_mean=opt.running_mean,
    steering_indicator=opt.steering_indicator,
    data_augmentation=opt.data_augmentation,
    no_norm=opt.no_norm,
    num_steering_indicators=opt.num_steering_indicators,
    data_type="point_clouds",
    lidar_fov=360
  )


else:
  exit('wrong dataset type')

print("Dataset length:", len(full_dataset))

if not opt.use_whole_dataset:
  # Split dataset in train and test
  train_size = int(0.8 * len(full_dataset))
  test_size = len(full_dataset) - train_size
  split = random_split(
      full_dataset, [train_size, test_size])
  dataset, test_dataset = split["datasets"]
  steering_vector = split["steering"]
  print(len(dataset), len(test_dataset))
  
  testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=int(opt.workers))
else:
  dataset = full_dataset
  steering_vector = dataset.get_steering_vector()

print("Sampler", opt.sampler)
if(opt.sampler == "balanced"):
  print("using the balanced sampler")
  balancedSampler = BalancedSteeringSampler(
    steering_vector,
    batch_size=int(opt.batch_size),
    only_left=opt.only_left)
  dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_sampler=balancedSampler,
      num_workers=int(opt.workers))
elif(opt.sampler == "weighted"):
  print("using the weighted sampler")
  weightedSampler = WeightedSteeringSampler(
    steering_vector,
    num_samples=len(steering_vector),
    replacement=True,
    only_left=opt.only_left)
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    sampler=weightedSampler,
    num_workers=int(opt.workers)
  )
else:
  # Initialize custom BalancedSteeringSampler to remove straight-steering bias
  print("using the default sampler")
  dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=int(opt.workers))

print(len(dataloader))

if(cuda_available):
  print("CUDA AVAILABLE")
else:
  print("CUDA NOT AVAILABLE")

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.steering_indicator:
  classifier = PointNetReg2(feature_transform=opt.feature_transform)
else:
  classifier = PointNetReg(feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batch_size

# visualize the data in visdom
# steering_indicator_vector = dataset.get_steering_indicator_vector()
# for index in range(len(dataset)):
#   plt.plot('visualize', 'indicator_vetor', 'Dataset visualization', index, steering_indicator_vector[index])
# for index in range(len(dataset)):
#   plt.plot('visualize', 'steering', 'Dataset visualization', index, dataset.get_steering(index))
#   plt.plot('visualize', 'indicator', 'Dataset visualization', index, dataset.get_steering_indicator(index))

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        if opt.steering_indicator:
          points, target, steering_indicator = data
          #print(steering_indicator)
          #print("Steering indicator", steering_indicator)
          steering_indicator = steering_indicator.cuda()
        else:
          points, target = data

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        if opt.steering_indicator:
          pred, trans, trans_feat = classifier((points, steering_indicator))
        else:
          pred, trans, trans_feat = classifier(points)
        loss = F.mse_loss(pred, target)  # changed from nll to mse loss
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        # A prediction is correct, if the steering angle is less than 5% different than the correct one
        correct_train = (torch.abs(target - pred) < 0.05).sum()
        accuracy_train = correct_train.item() / float(opt.batch_size)
        logmessage = '[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), accuracy_train)
        print(logmessage)
        logging.debug(logmessage)

        if plot_enabled:
          # Plot to visdom
          plt.plot('loss', 'train', 'Loss', epoch * num_batch + i, loss.item())
          plt.plot('acc', 'train', 'Accuracy', epoch *
                  num_batch + i, accuracy_train)
          plt.plot_moving_avg('loss', 'train_avg10', 'Loss',
                              epoch * num_batch + i, loss.item(), 10)
          plt.plot_moving_avg('acc', 'train_avg10', 'Accuracy', epoch *
                              num_batch + i, accuracy_train, 10)
          plt.plot_moving_avg('loss', 'train_avg50', 'Loss',
                              epoch * num_batch + i, loss.item(), 50)
          plt.plot_moving_avg('acc', 'train_avg50', 'Accuracy', epoch *
                              num_batch + i, accuracy_train, 50)
        # Plot training progress
        # the number of predictions are the number of elements in pred / target
        # we want to plot every prediction on a graph
        # per plot there are opt.batch_size = pred predictions
        # epoch * num_batch + i are the number of plots that were already there
        # (epoch * num_batch + i) * opt.batch_size are the number of elems on the graph
        # (epoch * num_batch + i) * opt.batch_size + k (where k = 1 to opt.batch_size) is the x we need to give to the plot

        # for k in range(pred.cpu().detach().size()[0]):
        #     plt.plot('visual-train', 'pred', 'Steering Training', (epoch * num_batch + i)
        #              * opt.batch_size + k, pred.cpu().detach().data[k])
        #     plt.plot('visual-train', 'target', 'Steering Training', (epoch * num_batch + i)
        #              * opt.batch_size + k, target.cpu().detach().data[k])
        
        # Skip testing for now
        """
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.mse_loss(pred, target)  # changed from nll to mse loss
            correct_test = (torch.abs(target - pred) < 0.05).sum()
            accuracy_test = correct_test.item() / float(opt.batch_size)
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i,
                                                            num_batch, blue('test'), loss.item(), accuracy_test))
            plt.plot('loss', 'test', 'Loss', epoch *
                     num_batch + i, loss.item())
            plt.plot('acc', 'test', 'Accuracy', epoch *
                     num_batch + i, accuracy_test)
            plt.plot_moving_avg('loss', 'test_avg10', 'Loss', epoch *
                                (num_batch // 10 + 1) + i, loss.item(), 10)
            plt.plot_moving_avg('acc', 'test_avg10', 'Accuracy', epoch *
                                (num_batch // 10 + 1) + i, accuracy_test, 10)

            # Plot test progress
            # per epoch there are 3 plots (0,10,20)
            # per plot there are opt.batch_size points
            # (epoch * 3 + i/10) * opt.batch_size is the number of points before the current plot
            # (epoch * 3 + i/10) * opt.batch_size + k is the number we need to give to the model
            for k in range(opt.batch_size):
                plt.plot('visual-test', 'pred', 'Steering Test', (epoch * (num_batch // 10 + 1) + i/10)
                         * opt.batch_size + k, pred.cpu().detach().data[k])
                plt.plot('visual-test', 'target', 'Steering Test', (epoch * (num_batch // 10 + 1) + i/10)
                         * opt.batch_size + k, target.cpu().detach().data[k])
        """
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' %
               (opt.outf, epoch))

if not opt.use_whole_dataset:
  total_correct = 0
  total_testset = 0
  for i, data in tqdm(enumerate(testdataloader, 0)):
      if opt.steering_indicator:
        points, target, steering_indicator = data
        steering_indicator = steering_indicator.cuda()
      else:
        points, target = data
      points = points.transpose(2, 1)
      points, target = points.cuda(), target.cuda()
      classifier = classifier.eval()
      if opt.steering_indicator:
        pred, _, _ = classifier((points, steering_indicator))
      else:
        pred, _, _ = classifier(points)
      print("Size", pred.size()[0])
      if plot_enabled:
        for k in range(pred.size()[0]):
          plt.plot('visual-test', 'pred', 'Steering Testdata', (i * opt.batchSize + k), pred.cpu().detach().data[k])
          plt.plot('visual-test', 'target', 'Steering Testdata', (i * opt.batchSize + k), target.cpu().detach().data[k])
      correct = (torch.abs(target.data - pred.data) < 0.05).cpu().sum()
      total_correct += correct.item()
      total_testset += points.size()[0]

  print("final accuracy {}".format(total_correct / float(total_testset)))
