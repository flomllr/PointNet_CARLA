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

segfiles = os.environ["DLPDATAPATH"] + "/CameraSemSeg0/"
depthfiles = os.environ["DLPDATAPATH"] + "/CameraDepth0/"
out = os.environ["DLPUSERPATH"] + "/PointCloudLocalSeg1s/"


for i in range(1000):
    imagename = 'image_' + '{:0>5d}'.format(i) + '.png'
    depth_img = io.imread(depthfiles + imagename)
    sem_img = io.imread(segfiles + imagename)
    sem_img = labels_to_cityscapes_palette(sem_img)

    grayscale = np.dot(depth_img[:, :, :3], [1.0, 256, 256.0 * 256.0])
    grayscale /= (256.0 * 256.0 * 256.0 - 1.0)

    point_cloud_obj = depth_to_local_point_cloud(
        grayscale, color=sem_img, max_depth=0.05)
    point_cloud_obj.save_to_disk(
        out + 'point_cloud_' + "{:0>5d}".format(i) + '.ply')
