import pptk
import numpy as np
import plyfile
import glob
import pdb
import time
import os
import sys
from tqdm import tqdm

def visualize_point_cloud(source, dest, mode='point_cloud'):
  data = plyfile.PlyData.read(source)['vertex']
  xyz = np.c_[data['x'], data['y'], data['z']]
  v = pptk.viewer(np.vstack(xyz))
  if mode == 'lidar':
    v.set(lookat=(0,1,10), show_grid=False, theta=-2.4, phi=4.8, r=97.5, point_size=0.0005, show_axis=False)    
  else:
    v.set(lookat=(0,2,0), show_grid=False, theta=-0.15, phi=1.6, r=0.1, point_size=0.0005, show_axis=False)
    v.set(lookat=(0,2,15), r=50)
  time.sleep(0.2)
  v.capture(dest)
  time.sleep(0.4)
  v.close()

def visualize_point_clouds(source, dest, mode="point_cloud"):
  pointsList = glob.glob(source)
  pointsList.sort()
  if not os.path.exists(dest):
      os.makedirs(dest)
  for index, pointcloud_src in tqdm(enumerate(pointsList)):
    try:
      visualize_point_cloud(pointcloud_src, "{}/image_{:0>5d}.png".format(dest, index), mode)
    except:
      pass

def main():
  # pos = 42
  # visualize_point_cloud(
  #   "/usr/prakt/s0050/ss19_extendingpointnet/online_carla/_capture/pos{}/point_clouds/point_cloud_00000.ply".format(pos),
  #   "/usr/prakt/s0050/ss19_extendingpointnet/online_carla/_capture/pos{}/visual.png".format(pos), 'lidar')
  #pos = sys.argv[1]
  typ = ""
  mode = "lidar"
  for pos in tqdm([42]):
    # specify data folder
    data_folder = '/usr/prakt/s0050/ss19_extendingpointnet/online_carla/_capture{}/pos{}/point_clouds/*.ply'.format(typ, pos)
    out = '_capture{}/pos{}/point_cloud_visual'.format(typ, pos)
    visualize_point_clouds(data_folder,out,mode)

if __name__ == "__main__":
  main()