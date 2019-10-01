from __future__ import print_function
import numpy as np
import argparse
import os
import glob
from plyfile import PlyData, PlyElement

## Segmentation colors
# Line	(157, 234, 50)
# Road	(128, 64, 128)
# Sidewalk	(244, 35, 232)

# Returs only the points in the specified rgb collor
def filter_rgb(array, rgb_array):
  res = array[(array == rgb_array[0])[:,3]] # all rows which have pattern[0] at their 3rd posotion (red value)
  res = res[(res == rgb_array[1])[:,4]] # all rows which have pattern[1] at their 4th position (green value)
  res = res[(res == rgb_array[2])[:,5]] # all rows which have pattern[1] at their 4th position (green value)
  return res

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

def filter_save_return(array, rgb_array, filename):
  # filter
  array = filter_rgb(array, rgb_array)
  # remove color data
  array = np.array(array[:,0:3])
  # save
  save_to_disk(array, filename)
  # return filtered points
  return array

def process_ply(file, dest):
  with open(file, 'rb') as f:
    plydata = PlyData.read(f)
    pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'],
                    plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T
    # only for comparison purposes
    #save_to_disk(pts[:,0:3], os.path.join(dest, 'original/', os.path.basename(file))

    #sidewalk = filter_save_return(pts, [244,35,232], os.path.join(dest, 'sidewalk/', os.path.basename(file)))
    #print('sidewalk ', sidewalk.shape)
    
    road = filter_save_return(pts, [128, 64, 128], os.path.join(dest, os.path.basename(file)))
    #print('road ', road.shape)

    #line = filter_save_return(pts, [157, 234, 50], os.path.join(dest, 'line/', os.path.basename(file)))
    #print('line', line.shape)
    
    # put road, line and sidewalk in one pointcloud and save it
    #united = np.concatenate((road, line, sidewalk))
    #print('united', united.shape)
    #save_to_disk(united, os.path.join(dest, 'united/', os.path.basename(file)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--datapath', type=str, help='path of dataset')
  parser.add_argument(
    '--out', type=str, help='path of destination')
  parser.add_argument(
    '--num_episodes', type=int, help='Number of episodes')

  opt = parser.parse_args()
  print(opt.datapath)
  datapath = opt.datapath
  #filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), opt.file)

  if opt.num_episodes:
    for ep in range(opt.num_episodes):
      print('Episode {}/{}'.format(ep, opt.num_episodes))
      fns = sorted(
        glob.glob(
          os.path.join(
            datapath, 'episode_{:0>3d}/*.ply'.format(ep)
          )
        )
      )
      for index, file in enumerate(fns):
        process_ply(file, os.path.join(opt.out, 'episode_{:0>3d}'.format(ep)))
        print('{}/{}'.format(index, len(fns)))
  else:
    fns = sorted(
      glob.glob(
        os.path.join(
          datapath, '*.ply'
        )
      )
    )
    for index, file in enumerate(fns):
      process_ply(file, opt.out)
      print('{}/{}'.format(index, len(fns)))
