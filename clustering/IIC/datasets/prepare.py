import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import pdb
import imageio as io
from shutil import copyfile


def _flatten_gt(gt, color=False):
    if color:
        # turn rgb into flat indices
        colour_dict = {"[0, 0, 0]": 0,  # unlabeled
                        "[70, 70, 70]": 1,  # building
                        "[190, 153, 153]": 2,  # fence
                        "[250, 170, 160]": 3,  # other
                        "[72, 0, 90]": 3, #other
                        "[220, 20, 60]": 4,  # pedestrian
                        "[153, 153, 153]": 5,  # pole
                        "[157, 234, 50]": 6,  # road line
                        "[128, 64, 128]": 7,  # road
                        "[244, 35, 232]": 8,  # sidewalk
                        "[107, 142, 35]": 9,  # vegetation
                        "[0, 0, 142]": 10,  # car
                        "[0, 0, 255]": 10, #vehicles
                        "[102, 102, 156]": 11,  # wall
                        "[220, 220, 0]": 12,  # traffic sign
                        #"[160, 160, 160]": 3,  # other
                        #"[112, 148, 37]": 3, # other
                        }
    else:
        colour_dict = {"[0, 0, 0]": 0,  # unlabeled
                        "[1, 0, 0]": 1,  # building
                        "[2, 0, 0]": 2,  # fence
                        "[3, 0, 0]": 3,  # other
                        "[4, 0, 0]": 4,  # pedestrian
                        "[5, 0, 0]": 5,  # pole
                        "[6, 0, 0]": 6,  # road line
                        "[7, 0, 0]": 7,  # road
                        "[8, 0, 0]": 8,  # sidewalk
                        "[9, 0, 0]": 9,  # vegetation
                        "[10, 0, 0]": 10,  # car
                        "[11, 0, 0]": 11,  # wall
                        "[12, 0, 0]": 12,  # traffic sign
                        #"[160, 160, 160]": 3,  # other
                        #"[112, 148, 37]": 3, # other
                        }

    gt = gt.astype(np.uint8)
    h, w, c = gt.shape
    flat_gt = np.zeros((h, w), dtype=np.int32)
    assert (c == 3)
    for y in xrange(h):
        for x in xrange(w):
            colour = str(list(gt[y, x]))
            gt_c = colour_dict[colour]
            flat_gt[y, x] = gt_c
    return flat_gt

def prepare_images(root, out, gt=True):
    gt_path = os.path.join(out, 'gt')
    img_path = os.path.join(out, 'imgs')
    if not os.path.exists(out):
        os.makedirs(out)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    fns = sorted(
        glob.glob(os.path.join(root, "*.png"))
    )
    print(fns)
    if gt:
        for index, path in tqdm(enumerate(fns)):
            gt_img = io.imread(path)
            gt_flat = _flatten_gt(gt_img)
            assert (gt_flat.dtype == np.int32)
            np.save(os.path.join(gt_path, str(index) + '.npy'), gt_flat)
    else:
        files_txt = open(os.path.join(out, 'all.txt'), 'a')
        for index, path in tqdm(enumerate(fns)):
            copyfile(path, os.path.join(img_path, str(index) + '.png'))
            files_txt.write(str(index)+'\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--root', type=str, help='Root path of ground truth directory')
    parser.add_argument(
            '--out', type=str, help='Root path of output directory')
    opt = parser.parse_args()
    prepare_images(os.path.join(opt.root, 'CameraSemSeg0'), os.path.join(opt.out))
    prepare_images(os.path.join(opt.root, 'CameraRGB0'), os.path.join(opt.out), gt=False)

