from __future__ import print_function
import os
import argparse
import pdb
import glob
from tqdm import tqdm

def rename_file(src, dest):
    os.rename(src, dest)

def rename_files_for_pointnet(src, dry_run=False):
    ply_fns = sorted(glob.glob(os.path.join(src, '*.ply')))
    print(ply_fns)

    if dry_run:
        fns = ply_fns
    else:
        fns = tqdm(ply_fns)

    for fn in fns:
        pathname = os.path.dirname(fn)
        filename = os.path.basename(fn)
        filename_without_ext = os.path.splitext(filename)[0]
        new_filename = 'point_cloud_{}.ply'.format(filename_without_ext.zfill(5))
        print("Renaming {} to {}".format(filename, new_filename))
        if not dry_run:
            rename_file(fn, os.path.join(pathname, new_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Root path of dataset", required=True)
    opt = parser.parse_args()
    rename_files_for_pointnet(opt.root, dry_run=True)
    pdb.set_trace()
    rename_files_for_pointnet(opt.root)
