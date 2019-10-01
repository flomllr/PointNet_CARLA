#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time
import pdb
import pre_processing as pp
import timeit
import torch
import sys
import numpy as np
import plotter
import os
from keyboard_input import KBHit

sys.path.append('../pointnet.pytorch')
from pointnet.model import PointNetReg, PointNetReg2

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
cuda_available = torch.cuda.is_available()
global plotter
global_steering_indicator = "left"
kb = KBHit()

def set_steering_indicator_from_keypress():
  global global_steering_indicator
  global kb
  if kb.kbhit():
    c = kb.getch()
    if c == 'a':
      global_steering_indicator = "left"
    elif c == 'd':
      global_steering_indicator = "right"
    elif c == 's':
      global_steering_indicator = "straight"

def filter_fov(array, degree=180):
  if degree != 180:
    print("FOV different from 180 degrees is not implemented yet.")
    exit()
  res = array[(array < 0)[:,1]]
  return res

def run_carla_client(args, classifier, plt, plt_index):
    global global_steering_indicator
    # Here we will run 3 episodes with 300 frames each.
    frames_per_episode = args.frames
    
    if args.position:
      number_of_episodes = 1
    else:
      number_of_episodes = 3

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode.

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=0,
                    NumberOfPedestrians=0,
                    WeatherId=random.choice([2]),
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.

                # # The default camera captures RGB images of the scene.
                # camera0 = Camera('CameraRGB')
                # # Set image resolution in pixels.
                # camera0.set_image_size(256, 256)
                # # Set its position relative to the car in meters.
                # camera0.set_position(2.2, 0, 1.30)
                # settings.add_sensor(camera0)

                if args.lidar:
                    # Adding a lidar sensor
                    lidar = Lidar('Lidar32')
                    lidar.set_position(0, 0, 2.50)
                    lidar.set_rotation(0, 0, 0)
                    lidar.set(
                        Channels=32,
                        Range=50,
                        PointsPerSecond=args.lidar_pps,
                        RotationFrequency=10,
                        UpperFovLimit=10,
                        LowerFovLimit=-30)
                    settings.add_sensor(lidar)
                else:
                    # Let's add another camera producing ground-truth depth.
                    camera1 = Camera('CameraDepth', PostProcessing='Depth')
                    camera1.set_image_size(256, 256)
                    camera1.set_position(2.2, 0, 1.30)
                    camera1.set(FOV=90.0)
                    #camera1.set_rotation(pitch=-8, yaw=0, roll=0)
                    settings.add_sensor(camera1)

                    camera2 = Camera('CameraSeg', PostProcessing='SemanticSegmentation')
                    camera2.set_image_size(256, 256)
                    camera2.set_position(2.2, 0, 1.30)
                    camera2.set(FOV=90.0)
                    #camera2.set_rotation(pitch=-8, yaw=0, roll=0)
                    settings.add_sensor(camera2)

                if args.capture:
                  camera3 = Camera('CameraRGB')
                  camera3.set_image_size(512, 256)
                  camera3.set_position(-8, 0, 5)
                  camera3.set(FOV=90.0)
                  camera3.set_rotation(pitch=-20, yaw=0, roll=0)
                  settings.add_sensor(camera3)


            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            if args.position:
              player_start = args.position
            else:
              player_start = random.choice([42,67,69,79,94,97, 70, 44,85,102])
            

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode at %r...' % scene.map_name)

            """ Begin added code """
            if args.capture:
              directory = '_capture/pos{}'.format(player_start)
              if not os.path.exists(directory):
                  os.makedirs(directory)
              print("Capturing point clouds to {}".format(directory))


            """ End added code """

            client.start_episode(player_start)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                """ Begin added code """
                # dict of steering directions of available curves
                # [1,0] if the next curve will be a left one
                # [0,1] if the next curve will be a right one
                directions = {
                  67: [[1,0]], # straight
                  42: [[1,0]], # straight or left
                  69: [[0,1]], # right
                  79: [[0,1]], # right
                  94: [[0,1]], # right
                  97: [[0,1]], # right
                  70: [[1,0]], # left
                  44: [[1,0]], # left
                  85: [[1,0]], # left
                  102: [[1,0]] # left
                }

                #dynamically set the global steering indicator during the game
                if(args.key_control):
                  set_steering_indicator_from_keypress()

                steering_direction = args.steering_indicator or global_steering_indicator

                if args.use_steering_indicator:
                  if steering_direction == "left":
                    steering_indicator = [[1,0]]
                  elif steering_direction == "right":
                    steering_indicator = [[0,1]]
                  else:
                    steering_indicator = [[0,0]]
                  steering_indicator = torch.Tensor(steering_indicator)
                  if cuda_available:
                    steering_indicator = steering_indicator.cuda()

                if args.lidar:
                    point_cloud = sensor_data['Lidar32'].data
                    if args.lidar_fov == 180:
                      print("FOV 180")
                      print("Before", point_cloud.shape)
                      point_cloud = filter_fov(point_cloud)
                      print("After", point_cloud.shape)


                else:
                    # Get depth and seg as numpy array for further processing
                    depth_obj = sensor_data['CameraDepth']
                    depth = depth_obj.data
                    fov = depth_obj.fov
                    
                    # Filter seg to get intersection points
                    seg = sensor_data['CameraSeg'].data
                    filtered_seg = pp.filter_seg_image(seg)

                    # Add following lines to measure performance of generating pointcloud
                    # def f():
                    #   return pp.depth_to_local_point_cloud(depth, fov, filtered_seg)
                    # print(timeit.timeit(f, number=1000) / 1000)

                    # Create pointcloud from seg and depth (but only take intersection points)
                    point_cloud = pp.depth_to_local_point_cloud(depth, fov, filtered_seg)

                # Save point cloud for later visualization
                if args.capture:
                  pp.save_to_disk(point_cloud, "{}/point_clouds/point_cloud_{:0>5d}.ply".format(directory, frame))
                  sensor_data['CameraRGB'].save_to_disk('{}/images/image_{:0>5d}.png'.format(directory, frame))
                

                # Normalizing the point cloud
                if not args.no_center or not args.no_norm:
                  point_cloud = point_cloud - np.expand_dims(np.mean(point_cloud, axis=0), 0)  # center
                
                if not args.no_norm:
                  dist = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)), 0)
                  point_cloud = point_cloud / dist  # scale

                #pp.save_point_cloud(point_cloud, 'test')
                # pp.save_seg_image(filtered_seg)
                """ End added code """
                # Print some of the measurements.
                #print_measurements(measurements)

                # # Save the images to disk if requested.
                # if args.save_images_to_disk:
                #     for name, measurement in sensor_data.items():
                #         filename = args.out_filename_format.format(episode, name, frame)
                #         measurement.save_to_disk(filename)

                # We can access the encoded data of a given image as numpy
                # array using its "data" property. For instance, to get the
                # depth value (normalized) at pixel X, Y
                #
                #     depth_array = sensor_data['CameraDepth'].data
                #     value_at_pixel = depth_array[Y, X]
                #

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.
                
                if not args.autopilot:
                    client.send_control(
                        steer=random.uniform(-1.0, 1.0),
                        throttle=0.5,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)

                else:

                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server. We can modify it if wanted, here for instance we
                    # will add some noise to the steer.


                    """ Beginn added code """
                    control = measurements.player_measurements.autopilot_control
                    #steer = control.steer
                    #print(control)

                    point_cloud = point_cloud.transpose()
                    points = np.expand_dims(point_cloud, axis=0)
                    points = torch.from_numpy(points.astype(np.float32))
                    #print(points)
                    if cuda_available:
                      points = points.cuda()
                    classifier = classifier.eval()

                    if args.use_steering_indicator:
                      #print("Using PointNetReg2")
                      steer, _, _ = classifier((points, steering_indicator))
                    else:
                      #print("Using PointNetReg")
                      steer, _, _ = classifier(points)
                    steer = steer.item()
                    if args.use_steering_indicator:
                      print("Using steering indicator: {} / {}".format(steering_direction, steering_indicator))
                    print("Autopilot steer: ", control.steer)
                    print("Pointnet steer: ", steer)
                    print("Relative difference: ", steer / control.steer if control.steer != 0.0 else 'NaN')
                    print("Absolute difference: ", control.steer - steer)
                    print("FOV:", args.lidar_fov)

                    # Plot some visualizations to visdom
                    if args.visdom:
                      plt.plot_point_cloud(points, var_name='pointcloud')
                      plt.plot('steering', 'PointNet', 'Steering', plt_index, steer)
                      plt.plot('steering', 'Autopilot', 'Steering', plt_index, control.steer)
                      plt_index += 1

                    # Let pointnet take over steering control
                    if True:
                      print("Who's in command: POINTNET")
                      control.steer = steer
                      if args.ignore_red_lights or args.use_steering_indicator:
                        control.throttle = 0.5
                        control.brake=0.0
                        hand_brake=False

                    else:
                      print("Who's in command: AUTOPILOT")

                    print("_________")
                    #pdb.set_trace()
                    """ End added code """
                    client.send_control(control)

                    # TODO :
                    # Replace lines 159-162 (autopilot control mode) with the steering angle predicted by the
                    # PointNet model


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '--model',
        type=str,
        required=True,
        help='model path')
    argparser.add_argument(
        '--feature_transform',
        action='store_true',
        help="use feature transform")
    argparser.add_argument(
        '--visdom',
        type=str,
        help='model path')
    argparser.add_argument(
        '--use_steering_indicator',
        action='store_true',
        help='Should the model be given an indicator of the steering direction of the next turn')
    argparser.add_argument(
        '--position',
        type=int,
        help='At which position should the simulation begin?')
    argparser.add_argument(
        '--steering_indicator',
        type=str,
        help='Will the next curve be a left or a right one?')
    argparser.add_argument(
        '--frames',
        type=int,
        default=100,
        help='How long should the simulation be? (in frames)')
    argparser.add_argument(
        '--capture',
        action='store_true',
        help="capture used point clouds")
    argparser.add_argument(
        '--key_control',
        action='store_true',
        help="Activate key control to dynamically set steering indicator")
    argparser.add_argument(
        '--ignore_red_lights',
        action='store_true',
        help="Ignore red lights")
    argparser.add_argument(
        '--no_center',
        action='store_true',
        help="Don't center the point cloud")
    argparser.add_argument(
        '--no_norm',
        action='store_true',
        help="Don't normalize the point cloud")
    argparser.add_argument(
        '--lidar',
        action='store_true',
        help="Use lidar instead of point cloud for driving")
    argparser.add_argument(
        '--lidar_pps',
        type=int,
        default=100000,
        help="Number of pps of lidar")
    argparser.add_argument(
      '--lidar_fov',
      type=int,
      choices=[180,360],
      default=360,
      help="FOV for lidar (options 360 or 180)")


    args = argparser.parse_args()
    args.autopilot = True

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    """ Begin added code """
    if args.visdom:
      plt = plotter.VisdomLinePlotter(env_name=args.visdom)
      plt_index = 0
    else:
      plt = None
      plt_index = None
    # Instanciate classifier, load trained model from path
    if args.use_steering_indicator:
      classifier = PointNetReg2(feature_transform=args.feature_transform)
    else:
      classifier = PointNetReg(feature_transform=args.feature_transform)
    if cuda_available:
      print("CUDA is available")
      classifier.load_state_dict(torch.load(args.model))
      classifier.cuda()
    else:
      print("CUDA is NOT available")
      classifier.load_state_dict(torch.load(args.model, map_location='cpu'))
      
    """ End added code """

    while True:
        try:

            run_carla_client(args, classifier=classifier, plt=plt, plt_index=plt_index)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

