#import Image
import numpy as np
import os
import sys
import math
import pdb

from numpy.matlib import repmat
from carla.image_converter import depth_to_local_point_cloud, labels_to_cityscapes_palette
from skimage import io
from scipy import ndimage
from plyfile import PlyData, PlyElement

def process_edge(image, label):
  # filter_road
  sem_img_filtered = np.where(image == label, label, 0)
  # Road edge processing
  sx = ndimage.sobel(sem_img_filtered, axis=0, mode='constant')
  sy = ndimage.sobel(sem_img_filtered, axis=1, mode='constant')
  sob = np.hypot(sx, sy)
  edge = np.where(sob > 0, label, 0)
  return edge

def filter_seg_image(image):
  # Edge processing
  road_edge = process_edge(image, 7)
  sidewalk_edge = process_edge(image, 8)
  line_edge = process_edge(image, 6)

  # filter intersections of (road and sidewalk) plus (road and lane)
  return np.where(
    np.logical_or(
      np.logical_and(road_edge > 0, sidewalk_edge > 0),
      np.logical_and(road_edge > 0, line_edge > 0)
    ), 1, 0)

def filter_fov(array, degree=180):
  if degree != 180:
    print("FOV different from 180 degrees is not implemented yet.")
    exit()
  array = array.transpose()
  res = array[(array < 0)[:,1]]
  return res

def depth_seg_to_pointcloud(depth, seg):
  # Converting depth image to grayscale
  grayscale = np.dot(depth[:, :, :3], [1.0, 256, 256.0 * 256.0])
  grayscale /= (256.0 * 256.0 * 256.0 - 1.0)

  point_cloud_obj = depth_to_local_point_cloud(
      grayscale, color=filtered_color, max_depth=0.05)


def depth_to_local_point_cloud(depth, fov, filter_array=None, max_depth=0.9, full=False):
  """
  Modified version of carla.image_converter.depth_to_local_point_cloud
  ---
  Convert an array containing the depth values of an image to a 2D array containing
  the 3D position (relative to the camera) and apply the filter (if given)
  "max_depth" is used to omit the points that are far enough.
  """
  far = 1000.0  # max depth in meters.
  height, width = depth.shape

  # (Intrinsic) K Matrix
  k = np.identity(3)
  k[0, 2] = width / 2.0
  k[1, 2] = height / 2.0
  k[0, 0] = k[1, 1] = width / \
      (2.0 * math.tan(fov * math.pi / 360.0))

  # 2d pixel coordinates
  pixel_length = width * height
  u_coord = repmat(np.r_[width-1:-1:-1],
                    height, 1).reshape(pixel_length)
  v_coord = repmat(np.c_[height-1:-1:-1],
                    1, width).reshape(pixel_length)
  depth = np.reshape(depth, pixel_length)

  # Search for pixels where the depth is greater than max_depth to
  # delete them
  max_depth_filter = np.where(depth > max_depth, True, False)
  if not full:
    filter_array_filter = np.where(filter_array.reshape(-1) == 0, True, False)
  else:
    # If we want to save the full point cloud, don't apply any filter
    filter_array_filter = np.zeros(max_depth_filter.size)
  delete_filter = np.logical_or(max_depth_filter, filter_array_filter)
  delete_indices = np.nonzero(delete_filter)
  depth = np.delete(depth, delete_indices)
  u_coord = np.delete(u_coord, delete_indices)
  v_coord = np.delete(v_coord, delete_indices)

  # p2d = [u,v,1]
  p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

  # P = [X,Y,Z]
  p3d = np.dot(np.linalg.inv(k), p2d)
  p3d *= depth * far

  # Formating the output to:
  # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
  return np.transpose(p3d)



#########
# Functions for testing purposes
#########

def save_seg_image(image):
  io.imsave('temp.png', image)

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

# Function copied and modified from cara/sensor.py
def save_to_disk(array, filename):
  """Save this point-cloud to disk as PLY format"""

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

  # Create folder to save if does not exist.
  folder = os.path.dirname(filename)
  if not os.path.isdir(folder):
    os.makedirs(folder)

  # Open the file and save with the specific PLY format.
  with open(filename, 'w+') as ply_file:
    ply_file.write('\n'.join([construct_ply_header(), ply]))

def open_ply(file, rgb=False):
    with open(file, 'rb') as f:
        plydata = PlyData.read(f)
        if rgb:
          pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'],
                          plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T
        else:
          pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']])

        return pts

if __name__ == "__main__":
  path = "/usr/prakt/s0050/PointCloudProcessed/lidar/natural_turns/pos42_/randomness_0.0/lidar/point_cloud_00030.ply"
  pts = open_ply(path)
  pts = filter_fov(pts)
  save_point_cloud(pts, "test.ply")
