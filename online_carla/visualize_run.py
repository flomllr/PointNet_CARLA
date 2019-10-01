import visualize_ply as vp
import gifgen as gg
import argparse
import glob
import os

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
  '--path',
  type=str,
  #required=True,
  help='Root dir of a run to visualize')
argparser.add_argument(
  '--root',
  type=str,
  #required=True,
  help='Root dir of the runs to visualize')
argparser.add_argument(
  '--gif',
  action='store_true',
  help="Only gif")
argparser.add_argument(
  '--images',
  action='store_true',
  help="Only images")
argparser.add_argument(
  '--point_cloud',
  action='store_true',
  help="Only pointcloud")
argparser.add_argument(
  '--lidar',
  action='store_true',
  help="Only lidar")
argparser.add_argument(
  '--duration',
  type=float,
  default=0.02,
  #required=True,
  help='Duration for each frame')

args = argparser.parse_args()

run_paths = glob.glob(os.path.join(args.root, '*')) if args.root else [args.path]
print(run_paths)
failed_paths = []

for path in run_paths:
  if not args.gif and not args.point_cloud:
    args.gif = True
    args.point_cloud = True

  if args.point_cloud:
    print("*** VISUALIZING POINT CLOUDS ***")
    data_folder = os.path.join(path, "point_clouds/*.ply")
    out = os.path.join(path,"point_clouds_visual")
    vp.visualize_point_clouds(data_folder,out,'point_cloud')

  if args.lidar:
    print("*** VISUALIZING POINT CLOUDS ***")
    data_folder = os.path.join(path, "point_clouds/*.ply")
    out = os.path.join(path,"point_clouds_visual")
    vp.visualize_point_clouds(data_folder,out,'lidar')

  if args.gif:
    try:
      print("*** GENERATING GIF ***")
      if args.images:
        source = os.path.join(path,"images/")
        dest = path
        filename = "rgb"
        gg.generate_gif(source, dest, filename, args.duration)
      else:
        sources = (os.path.join(path, "images/"), os.path.join(path, "point_clouds_visual/"))
        dests = tuple([path] * 2)
        filenames = ('rgb', 'ply')
        gg.generate_syncronized_gif(sources, dests, filenames, args.duration)
    except:
      failed_paths.append(path)
    finally:
      print("Done with this run. Failed paths:", failed_paths)

print("### DONE ### - Failed:", failed_paths)



