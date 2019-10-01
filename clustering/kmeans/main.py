import numpy as np
import cv2
import argparse
import glob
from skimage import io
from skimage.util import img_as_float
import os
from tqdm import tqdm
import pdb

def kmeans(img, K):
  Z = img.reshape((-1,3))

  # convert to np.float32
  Z = np.float32(Z)

  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((img.shape))
  return res2

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--datapath', type=str, help='Root path of dataset')
  parser.add_argument(
      '--out', type=str, help='Root path of output dir')
  args = parser.parse_args()
  
  fns = sorted(
    glob.glob(os.path.join(args.datapath, "*.png"))
  )
  print(fns)
  for path in tqdm(fns):
    img = io.imread(path)
    img = kmeans(img, 3)
    io.imsave(os.path.join(args.out, os.path.basename(path)), img)


