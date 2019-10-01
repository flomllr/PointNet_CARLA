import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import argparse
import plotter
from tqdm import tqdm
from pointnet.model import PointNetReg, PointNetReg2
from pointnet.dataset import SteeringDataset, SteeringDatasetEpisodes

parser = argparse.ArgumentParser()
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--model', type=str, required=True, help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument(
  '--num_points', type=int, default=2500, help='number of points per cloud')
parser.add_argument(
  '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
  '--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--visdom', type=str, default='Training Plots', help='name of the visdom plot')
parser.add_argument('--dataset_type', type=str,
                    default='steering', help="dataset type (steering)")
parser.add_argument('--running_mean', type=int, help='Size of the running mean window used to smoothen steering angles')
parser.add_argument('--steering_indicator', action='store_true', help='Should an additional steering indicator be passed to the model?')
parser.add_argument('--debug', type=bool, help='Running the tester in debug mode? (Saving the pedicted pointcloud')
parser.add_argument('--num_episodes', type=int, help='Number of episodes')

opt = parser.parse_args()

num_episodes = opt.num_episodes or 10

plt = plotter.VisdomLinePlotter(env_name=opt.visdom)

def save_point_cloud(array, filename):

  """
  Modified version of carla.sensor.PointCloud.save_to_disk
  Save this point cloud to disk as PLY format
  """

  def construct_ply_header():
    """Generates a PLY header given a total number of 3D points and
    coloring property if specified
    """
    points = array.shape[0]  # Total point number
    header = ['ply',
                'format ascii 1.0',
                'element vertex {}',
                'property float32 x',
                'property float32 y',
                'property float32 z',
                'end_header']
    return '\n'.join(header).format(points)

  ply = '\n'.join(['{:.2f} {:.2f} {:.2f}'.format(*p) for p in array.tolist()])

  # Open the file and save with the specific PLY format.
  with open(filename, 'w+') as ply_file:
    ply_file.write('\n'.join([construct_ply_header(), ply]))

# Loading the trained model
if opt.steering_indicator:
  classifier = PointNetReg2(feature_transform=opt.feature_transform)
else:
  classifier = PointNetReg(feature_transform=opt.feature_transform)
classifier.load_state_dict(torch.load(opt.model))
classifier.cuda()

# Defining the dataset
if opt.dataset_type == "steering":
  dataset = SteeringDataset(
    root=opt.dataset,
    npoints=opt.num_points,
    running_mean=opt.running_mean,
    steering_indicator=opt.steering_indicator,
    data_augmentation=False)
elif opt.dataset_type == "steering_episodes":
  dataset = SteeringDatasetEpisodes(
    root=opt.dataset,
    npoints=opt.num_points,
    running_mean=opt.running_mean,
    episodes=range(num_episodes),#range(10),
    steering_indicator=opt.steering_indicator,
    data_augmentation=False)
else:
  exit('wrong dataset type')

dataloader = torch.utils.data.DataLoader(
  dataset,
  batch_size=opt.batch_size,
  shuffle=False,
  num_workers=int(opt.workers))

# Testing the model
total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(dataloader, 0)):
    if opt.steering_indicator:
      points, target, steering_indicator = data
      steering_indicator = steering_indicator.cuda()
    else:
      points, target = data

    # Save the drawn pointcloud for debugging
    if opt.debug:
      np_points = points.numpy()
      chosen_pointcloud = np_points[0]
      save_point_cloud(chosen_pointcloud, 'debugging_points.ply')

    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    points
    classifier = classifier.eval()
    if opt.steering_indicator:
      pred, _, _ = classifier((points, steering_indicator))
    else:
      pred, _, _ = classifier(points)
    print("Size", pred.size()[0])
    
    if opt.debug:
      print("Prediction ", pred.cpu().detach().data[0])
      exit()

    for k in range(pred.size()[0]):
      plt.plot('visual-test', 'pred', 'Steering Test', (i * opt.batch_size + k), pred.cpu().detach().data[k])
      plt.plot('visual-test', 'target', 'Steering Test', (i * opt.batch_size + k), target.cpu().detach().data[k])
      #print(steering_indicator)
      if opt.steering_indicator:
        plt.plot(
          'visual-test',
          'steering-indicator',
          'Steering Test',
          (i * opt.batch_size + k),
          -1 if steering_indicator[k][0] else 1)
    correct = (torch.abs(target.data - pred.data) < 0.05).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))


