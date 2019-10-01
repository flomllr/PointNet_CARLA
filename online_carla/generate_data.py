#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modified by Florian Müller


""" Script to generate autonomous driving training data from Carla"""

from __future__ import print_function

import argparse
import logging
import random
import time
import os
import pre_processing as pp
import numpy as np
import shutil

from tqdm import tqdm
from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

def run_carla_client(args):

    # Settings
    # Town 1
    ## Natural turns: [42,67,69,79,94,97,70,44,85,102], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ## T-intersection: [88,90,107,133,136]
    ## Side intersection left: [ 66, 16, 25 ]
    ## Side intersection right: [ 138, 19, 26 ]
    ## Staight: [14, 149, 78, 71, 89]

    # Town 2
    ## Natural turns right: [65, 78, 44, 1]
    ## Natural turns left: [49, 50, 57, 70]
    ## T-intersection [2, 5, 10, 11, 19, 26, 34, 37]
    ## Side intersection left: [7, 23, 16, 39, 6, 43, 79]
    ## Side intersection right: [40, 20, 28, 32, 34, 46, 71, 74]
    ## Straight: [61, 64, 77, 51]


    positions = args.positions or [65, 78, 49, 50, 2, 5, 10, 7, 23, 40, 20, 61]
    levels_of_randomness = args.randomness or [0.0, 0.2, 0.4, 0.6]
    frames_per_level = args.frames or [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]

    print("Positions: ", positions)
    print("Levels of randomness: ", levels_of_randomness)
    print("Frames per level: ", frames_per_level)

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        directions = ["_left", "_right"] if args.force_left_and_right else ["_"]
        for direction in directions:
            print("Direction ", direction)
            # Create a CarlaSettings object. This object is a wrapper around
            # the CarlaSettings.ini file. Here we set the configuration we
            # want for the new episode.
            settings = CarlaSettings()
            direction_var = 6 if direction == "_right" else 0
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=0,
                NumberOfPedestrians=0,
                WeatherId=random.choice([2]),
                QualityLevel=args.quality_level,
                SeedVehicles=direction_var)
            #settings.randomize_seeds()

            if args.capture:
                # To create visualisation of the current run
                camera3 = Camera('CameraRGB')
                camera3.set_image_size(512, 256)
                camera3.set_position(-8, 0, 5)
                camera3.set(FOV=args.fov)
                camera3.set_rotation(pitch=-20, yaw=0, roll=0)
                settings.add_sensor(camera3)
                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.
                if args.point_cloud:
                    # Camera to produce point Cloud
                    camera1 = Camera('CameraDepth', PostProcessing='Depth')
                    camera1.set_image_size(256, 256)
                    camera1.set_position(2.2, 0, 1.30)
                    camera1.set(FOV=args.fov)
                    settings.add_sensor(camera1)

                    camera2 = Camera('CameraSeg', PostProcessing='SemanticSegmentation')
                    camera2.set_image_size(256, 256)
                    camera2.set_position(2.2, 0, 1.30)
                    camera2.set(FOV=args.fov)
                    settings.add_sensor(camera2)


                if args.lidar:
                    lidar = Lidar('Lidar32')
                    lidar.set_position(0, 0, 2.50)
                    lidar.set_rotation(0, 0, 0)
                    lidar.set(
                        Channels=32,
                        Range=50,
                        PointsPerSecond=100000,
                        RotationFrequency=10,
                        UpperFovLimit=10,
                        LowerFovLimit=-30)
                    settings.add_sensor(lidar)

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.

            for episode, position in enumerate(positions):
                for randomness, frames in zip(levels_of_randomness, frames_per_level):
                    crashed = True
                    while crashed:
                        crashed = False
                        # Start a new episode.
                        print('Starting new episode at %r...' % scene.map_name)
                        print('Episode {}, Position {}, Randomness {}'.format(episode, position, randomness))
                        client.start_episode(position)

                        if args.capture:
                            # Make sure directories exist
                            directory = '_out/pos{}{}/randomness_{}'.format(position, direction, randomness)
                            ply_dir = '{}/point_clouds'.format(directory)
                            ply_dir_full = '{}/point_clouds_full'.format(directory)
                            lidar_dir = '{}/lidar'.format(directory)
                            img_dir = '{}/images'.format(directory)
                            if not os.path.exists(img_dir):
                                os.makedirs(img_dir)
                            if args.point_cloud and not os.path.exists(ply_dir):
                                os.makedirs(ply_dir)
                            if args.point_cloud and not os.path.exists(ply_dir_full):
                                os.makedirs(ply_dir_full)
                            if args.lidar and not os.path.exists(lidar_dir):
                                os.makedirs(lidar_dir)
                            # Write steering data to csv file
                            if args.point_cloud:
                                csv = open("{}/driving_data.csv".format(ply_dir), "w")
                            elif args.lidar:
                                csv = open("{}/driving_data.csv".format(lidar_dir), "w")
                            csv.write(",name,speed,throttle,steer\n")

                        # Iterate every frame in the episode
                        for frame in tqdm(range(frames)):
                            # Read the data produced by the server this frame.
                            measurements, sensor_data = client.read_data()

                            if args.capture:
                                if args.point_cloud:
                                # Save processed point clouds and autopilot steering to disk if requested
                                    # Get depth and seg as numpy array for further processing
                                    depth_obj = sensor_data['CameraDepth']
                                    depth = depth_obj.data
                                    fov = depth_obj.fov
                                    
                                    # Filter seg to get intersection points
                                    seg = sensor_data['CameraSeg'].data
                                    filtered_seg = pp.filter_seg_image(seg)

                                    if args.full_point_cloud:
                                        # Converting depth image to grayscale
                                        point_cloud_full = pp.depth_to_local_point_cloud(depth, fov, seg, max_depth=0.05, full=True)
                                        filename_full = "point_cloud_full_{:0>5d}".format(frame)
                                        pp.save_to_disk(point_cloud_full, "{}/{}.ply".format(ply_dir_full, filename_full))

                                    # Create pointcloud from seg and depth (but only take intersection points)
                                    point_cloud = pp.depth_to_local_point_cloud(depth, fov, filtered_seg, max_depth=0.05)
                                    filename = "point_cloud_{:0>5d}".format(frame)
                                    pp.save_to_disk(point_cloud, "{}/{}.ply".format(ply_dir, filename))


                                if args.lidar:
                                    sensor_data['Lidar32'].save_to_disk('{}/point_cloud_{:0>5d}'.format(lidar_dir, frame))
                                
                                # Save steering data of this frame
                                control = measurements.player_measurements.autopilot_control
                                csv.write("0,image_{:0>5d},0,0,{}\n".format(frame, control.steer))

                                # Save rgb image to visualize later
                                sensor_data['CameraRGB'].save_to_disk('{}/image_{:0>5d}.png'.format(img_dir, frame))

                            # Now we have to send the instructions to control the vehicle.
                            # If we are in synchronous mode the server will pause the
                            # simulation until we send this control.

                            control = measurements.player_measurements.autopilot_control
                            speed = measurements.player_measurements.forward_speed
                            old_steer = control.steer
                            control.steer += random.uniform(-randomness,randomness)

                            if args.ignore_red_lights:
                                control.throttle = 0.5
                                control.brake = 0.0
                                control.hand_brake = False
                                control.reverse = False

                            client.send_control(control)
                            crashed = False
                            if speed < 1 and abs(old_steer) > 0.5:
                                print("\nSeems like we crashed.\nTrying again...")
                                crashed = True
                                break

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
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
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '--randomness',
        nargs='*',
        type=float,
        help='set magnitude of randomness during steering')
    argparser.add_argument(
        '-c', '--capture',
        action='store_true',
        dest='capture',
        help='save point clouds and steering data to disk')
    argparser.add_argument(
        '--ignore_red_lights',
        action='store_true',
        help='Should red lights be ignored?')
    argparser.add_argument(
        '--full_point_cloud',
        action='store_true',
        help='Should the full point cloud be stored in addition to the filtered one?')
    argparser.add_argument(
        '--frames',
        nargs='*',
        type=int,
        help='How long should the simulation be? (in frames)')
    argparser.add_argument(
        '--positions',
        nargs='*',
        type=int,
        help='At which position should the simulation begin?')
    argparser.add_argument(
        '--force_left_and_right',
        action='store_true',
        help='For each position, both left and right turns need to be recorded')
    argparser.add_argument(
        '--fov',
        type=int,
        default=90,
        help='Field of view')
    argparser.add_argument(
        '--point_cloud',
        action='store_true',
        help='Save point clouds?')
    argparser.add_argument(
        '--lidar',
        action='store_true',
        help='Save lidar data?')

    args = argparser.parse_args()
    print("Args:", args)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:
            run_carla_client(args)
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
