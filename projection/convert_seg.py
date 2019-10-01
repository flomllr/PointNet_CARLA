# https://github.com/carla-simulator/carla/blob/master/Docs/cameras_and_sensors.md#depth-map
# https://github.com/carla-simulator/carla/tree/lidar_gpu/Deprecated/PythonClient/carla
from carla.image_converter import depth_to_local_point_cloud, labels_to_cityscapes_palette
from skimage import io
from scipy import ndimage
import cv2
import numpy as np
import pdb
import os
import glob
import argparse
from tqdm import tqdm

def noisy(noise_typ,image):
  if noise_typ == "gauss":
    row,col,ch= image.shape
    mean = 0
    var = 200
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy
  elif noise_typ == "s&p":
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.05
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    return out
  elif noise_typ == "poisson":
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy
  elif noise_typ =="speckle":
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss * 0.3
    return noisy
  else:
    return image

def convert_seg(root, dest):
  if not os.path.exists(dest):
    os.makedirs(dest)
  fns = sorted(
    glob.glob(os.path.join(root, "*.png"))
  )
  print(fns)
  for index, path in tqdm(enumerate(fns)):
    sem_img = io.imread(path)
    #pdb.set_trace()
    sem_img = labels_to_cityscapes_palette(sem_img)
    #sem_img = noisy("nothing", sem_img)
    #sem_img = sem_img / np.amax(sem_img)
    sem_img = sem_img / 255
    io.imsave(os.path.join(dest, str(index) + '.png'), sem_img)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--root', type=str, help='Root path of dataset')
  parser.add_argument(
      '--out', type=str, help='Root path of output dir')
  opt = parser.parse_args()
  convert_seg(opt.root, opt.out)
