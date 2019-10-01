import imageio
import glob
import sys
import os
from tqdm import tqdm

def generate_gif(source, dest=None, filename=None, duration=0.005):
  print(source, dest, filename)
  if not dest:
    dest = source
  if not filename:
    filename = 'gif'
  images = []
  files = sorted(
    glob.glob("{}*.png".format(source))
  )
  for file in tqdm(files):
    try:
      rgb = imageio.imread(file)
      images.append(rgb)
    except:
      pass
  imageio.mimsave('{}/{}.gif'.format(dest, filename), images, format='GIF', duration=duration)

def generate_gifs(sources, dests=None, filenames=None):
  if not dests or len(sources) != len(dests):
    print("Saving gifs in the source directories")
    dests = sources
  if not filenames:
    filenames = ['gif'] * len(sources)
  for source, dest, filename in tdqm(zip(sources, dests, filenames)):
    generate_gif(source, dest, filename)

def generate_syncronized_gif(source_tuple, dest_tuple, filename_tuple, duration=0.005):
  if len(source_tuple) != 2 or len(dest_tuple) != 2 or len(filename_tuple) != 2:
    sys.exit("Tuples must have dimension 2")

  files_0 = sorted(glob.glob(os.path.join(source_tuple[0], '*.png')))
  files_1 = sorted(glob.glob(os.path.join(source_tuple[1], '*.png')))
  images_0 = []
  images_1 = []

  for files in tqdm(zip(files_0, files_1)):
    try:
      ply = imageio.imread(files[1])
      images_1.append(ply)
      rgb = imageio.imread(files[0])
      images_0.append(rgb)
    except:
      pass
  imageio.mimsave(os.path.join(dest_tuple[0], filename_tuple[0]+".gif"), images_0, format='GIF', duration=duration)
  imageio.mimsave(os.path.join(dest_tuple[1], filename_tuple[1]+".gif"), images_1, format='GIF', duration=duration)
  

def generate_syncronized_gifs(sources_tuple, dests_tuple, filenames):
  if len(sources_tuple) != 2 or len(dests_tuple) != 2 or len(filenames) != 2:
    sys.exit("Tuples must have dimension 2")
  for source, dest in tqdm(zip(
      zip(sources_tuple[0], sources_tuple[1]), 
      zip(dests_tuple[0], dests_tuple[1]),
    )):
    generate_syncronized_gif(source, dest, filenames)


def main():
  if len(sys.argv) > 1:
    generate_gif(sys.argv[1])
  else:
    typ = "natural_nofeature"
    positions = [42,85]
    sources_rgb = [None] * len(positions)
    sources_ply = [None] * len(positions)

    sources_rgb = ["_capture_pos{}/pos{}/images/".format(typ, pos) for pos in positions]
    sources_ply = ["_capture_pos{}/pos{}/point_cloud_visual/".format(typ, pos) for pos in positions]

    sources = (sources_rgb, sources_ply)
    dests = tuple(["_capture_pos{}/pos{}/".format(typ, pos) for pos in positions] * 2)
    filenames = (['rgb'] * len(sources), ['ply'] * len(sources))

    generate_syncronized_gifs(sources, dests, filenames)

if __name__ == "__main__":
  main()