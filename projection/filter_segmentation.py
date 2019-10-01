# https://github.com/carla-simulator/carla/blob/master/Docs/cameras_and_sensors.md#depth-map
# https://github.com/carla-simulator/carla/tree/lidar_gpu/Deprecated/PythonClient/carla
from carla.image_converter import depth_to_local_point_cloud, labels_to_cityscapes_palette
from skimage import io
from scipy import ndimage
import numpy as np
import os
import sys

def process_edge(image, label, gaussian=False):
  # filter_road
  sem_img_filtered = np.where(image == label, label, 0)
  # Road edge processing
  sx = ndimage.sobel(sem_img_filtered, axis=0, mode='constant')
  sy = ndimage.sobel(sem_img_filtered, axis=1, mode='constant')
  sob = np.hypot(sx, sy)
  edge = np.where(sob > 0, label, 0)
  if gaussian:
    edge = ndimage.maximum_filter(edge, size=5)
  return edge

def filter_image(image):
  # Convert to numpy array and last axis to tuple
  sem_img_np = np.array(image)

  # Edge processing
  thick = False
  road_edge = process_edge(sem_img_np, 7, gaussian=thick)
  sidewalk_edge = process_edge(sem_img_np, 8, gaussian=thick)
  line_edge = process_edge(sem_img_np, 6, gaussian=thick)

  # filter sidewalk
  return np.where(
    np.logical_or(
      np.logical_and(road_edge > 0, sidewalk_edge > 0),
      np.logical_and(road_edge > 0, line_edge > 0)
    ), 7, 0)



if __name__ == "__main__":
  segfiles = os.environ["DLPDATAPATH"] + "/CameraSemSeg0/"
  num = sys.argv[1]
  print "Reading image"
  imagename = 'image_' + '{:0>5d}'.format(int(num)) + '.png'
  sem_img = io.imread(segfiles + imagename)

  print "Filter image"
  intersect = filter_image(sem_img)

  # Sidewalk edge processing
  print "Saving"
  intersect_color = labels_to_cityscapes_palette(intersect)
  intersect_color = intersect_color / 255
  io.imsave('out_edge_gaussian.png', intersect_color)

  sem_img = labels_to_cityscapes_palette(sem_img)
  sem_img = sem_img / 255
  io.imsave('out.png',sem_img)
