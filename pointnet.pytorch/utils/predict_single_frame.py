from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import argparse
from pointnet.model import PointNetReg, PointNetReg2
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser()
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--model', type=str, required=True, help='model path')
parser.add_argument('--pointcloud', type=str, required=True, help='point cloud')
parser.add_argument('--steering_indicator', type=bool, help='Should an additional steering indicator be passed to the model?')
parser.add_argument('--num_points', type=int, help='Number of points to be drawn from the pointcloud')



# parser.add_argument('--dataset', type=str, required=True, help="dataset path")
# parser.add_argument(
#   '--num_points', type=int, default=2500, help='number of points per cloud')
# parser.add_argument(
#   '--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument(
#   '--batchSize', type=int, default=32, help='input batch size')
# parser.add_argument('--visdom', type=str, default='Training Plots', help='name of the visdom plot')
# parser.add_argument('--dataset_type', type=str,
#                     default='steering', help="dataset type (steering)")
# parser.add_argument('--running_mean', type=int, help='Size of the running mean window used to smoothen steering angles')


opt = parser.parse_args()

# Loading the trained model
if opt.steering_indicator:
  print("NOT IMPLEMENTED YET")
  exit()
  #classifier = PointNetReg2(feature_transform=opt.feature_transform)
else:
  classifier = PointNetReg(feature_transform=opt.feature_transform)
classifier.load_state_dict(torch.load(opt.model))
classifier.cuda()

# Load the pointcloud
plydata = PlyData.read(opt.pointcloud)
points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

# subsample pointcloud
if opt.num_points:
  choice = np.random.choice(points.shape[0], size=opt.num_points, replace=False)
  points = points[choice]

print(points)
print(points.shape)

points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
points = points / dist  # scale

points = points.transpose(1, 0)
points = np.expand_dims(points, axis=0)
points = torch.from_numpy(points.astype(np.float32))
points = points.cuda()
classifier = classifier.eval()
pred, _, _ = classifier(points)
print(pred)
