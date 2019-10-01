"""
This file is used to infinitely running the CARLA server.
"""

import argparse
import os


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--town',
        default='Town01',
        type=str,
        help='Which Town to run (Town01/Town02)')
    argparser.add_argument(
        '--carla-dir',
        default='/usr/prakt/s0050/carla_online/CARLA_0.8.2/',
        type=str,
        help='Root dir of the carla simulator')

    args = argparser.parse_args()

    # set server path
    carla_server_path = args.carla_dir + 'CarlaUE4.sh ' + args.town + ' -carla-server -windowed -ResX=800 -ResY=600 -benchmark -fps=10 -WeatherId=8'

    while True:
        try:
            os.system(carla_server_path)
        except KeyboardInterrupt:
            print("You cancelled the simulation.")


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
