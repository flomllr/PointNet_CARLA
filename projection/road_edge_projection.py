# https://github.com/carla-simulator/carla/blob/master/Docs/cameras_and_sensors.md#depth-map
# https://github.com/carla-simulator/carla/tree/lidar_gpu/Deprecated/PythonClient/carla
from carla.image_converter import depth_to_local_point_cloud, labels_to_cityscapes_palette
from skimage import io
from scipy import ndimage
from filter_segmentation import filter_image
import cv2
import numpy as np
import pdb
import os
import glob
import argparse

def fns_to_pointcloud(fns, source, destination):
  for index, imagename in enumerate(fns):
    print '{}/{}'.format(index, len(fns))
    depth_img = io.imread(os.path.join(source, 'CameraDepth0', imagename))
    sem_img = io.imread(os.path.join(source, 'CameraSemSeg0', imagename))

    filtered = filter_image(sem_img)
    filtered_color = labels_to_cityscapes_palette(filtered)
    grayscale = np.dot(depth_img[:, :, :3], [1.0, 256, 256.0 * 256.0])
    grayscale /= (256.0 * 256.0 * 256.0 - 1.0)

    point_cloud_obj = depth_to_local_point_cloud(
      grayscale, color=filtered_color, max_depth=0.05)
    
    point_cloud_obj.save_to_disk(
      os.path.join(
        destination, 'point_cloud_{}.ply'.format("{:0>5d}".format(index))
      )
    )

def generate_pointclouds(datapath, destination, num_episodes=None):
  if num_episodes:
    for ep in range(num_episodes):
      print 'Episode {}/{}'.format(ep, num_episodes)
      fns = sorted(
        [os.path.basename(x) for x in 
          glob.glob(
            os.path.join(
              datapath, 'episode_{}/CameraSemSeg0/*.png'.format("{:0>3d}".format(ep))
            )
          )
        ]
      )
      fns_to_pointcloud(
        fns,
        os.path.join(datapath, 'episode_{}'.format("{:0>3d}".format(ep))),
        os.path.join(destination, 'episode_{}'.format("{:0>3d}".format(ep))),
      )
  else:
    fns = sorted(
      [os.path.basename(x) for x in
        glob.glob(
          os.path.join(
            datapath, 'CameraSemSeg0/*.png'
          )
        )
      ]
    )
    fns_to_pointcloud(fns, datapath, destination)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_episodes', type=int, help='Number of episodes in the dataset')
  parser.add_argument(
      '--datapath', type=str, help='Root path of dataset')
  parser.add_argument(
      '--out', type=str, help='Root path of output dir')
  opt = parser.parse_args()
  generate_pointclouds(opt.datapath, opt.out, opt.num_episodes)
