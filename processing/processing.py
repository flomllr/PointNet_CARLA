from __future__ import print_function
import argparse
import os
import glob
import imageio as io
import numpy as np
import sys
import pdb
import cv2
from tqdm import tqdm

# Clustering imports
clustering_root = os.path.join(os.path.dirname(__file__), '..', 'clustering/IIC')
sys.path.insert(0, clustering_root)
import torch
import pickle
from code.utils.segmentation.data import make_Coco_dataloaders, \
    make_Potsdam_dataloaders, make_Carla_dataloaders
import code.archs as archs
from code.utils.segmentation.segmentation_eval import \
    _segmentation_get_data, segmentation_eval
from code.utils.segmentation.render import render
from code.utils.segmentation.transforms import \
    custom_greyscale_numpy

# Carla point cloud processing import
clustering_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, clustering_root)
from carla_utils.image_converter import \
    depth_to_local_point_cloud, labels_to_cityscapes_palette, cityscape_labels

# load rgb image from root
# load depth image from root
# evaluate image using IIC model to get segmentation
# project segmentation and depth to get pointcloud
# filter point cloud using segmentation image
# save result in ply file


def get_rgb_depth_filenames(root, only_rgb=False):
    """Load rgb and depth images from given root directory
        and return tuple containing two lists. Image files
        must be in .png format
    
    Arguments:
        root {str} -- The root directory to load the filenames from. Must contain two subdirectories CameraDepth0 and CameraRGB0
    
    Returns:
        (rgb : list, depth : list) -- A tuple containing two lists with corresponding full paths of the images
    """
    if only_rgb:
        rgb_fns = sorted(glob.glob(os.path.join(root, '*.png')))
        return (rgb_fns, [])
    else:
        rgb_fns = sorted(glob.glob(os.path.join(root, 'CameraRGB0/*.png')))
        depth_fns = sorted(glob.glob(os.path.join(root, 'CameraDepth0/*.png')))
        return (rgb_fns, depth_fns)


def predict_segmentation(filename,
                         model_root="/usr/prakt/s0050/IIC_selfmade/1",
                         net_name="latest"):
    """Returns the predicted segmentation of an rgb image
    
    Arguments:
        filename {str} -- The full path to the image file
    
    Keyword Arguments:
        model_root {str} -- Full path to the root directory of the model. Directory must contain the .pytorch file containing the model and a config.pickle file containing the IIC config. (default: {"/usr/prakt/s0050/1"})
        net_name {str} -- Name of the .pytorch file inside of model_root (default: {"latest"})
    
    Returns:
        numpy array -- Numpy array with shape (h, w, 3) containing the segmentation. h and w correspond to height and width of the original image file
    """
    cuda_available = torch.cuda.is_available()
    print("cuda is available" if cuda_available else "cuda is NOT available")

    # Load config
    reloaded_config_path = os.path.join(model_root, "config.pickle")
    print("Loading restarting config from: %s" % reloaded_config_path)
    with open(reloaded_config_path, "rb") as config_f:
        config = pickle.load(config_f)
    assert (str(config.model_ind) == os.path.basename(model_root))

    if not hasattr(config, "use_doersch_datasets"):
        config.use_doersch_datasets = False

    dataloaders_train, mapping_assignment_dataloader, mapping_test_dataloader \
        = make_Carla_dataloaders(config)
    all_label_names = [
        "building", "fence", "other", "pedestrian", "pole", "road line",
        "road", "sidewalk", "vegetation", "car", "vehicles", "wall",
        "traffic sign"
    ]
    assert (len(all_label_names) == config.gt_k)

    # Load model
    net = archs.__dict__[config.arch](config)
    model_path = os.path.join(model_root, net_name + '.pytorch')
    print("getting model path %s " % model_path)
    loaded = torch.load(model_path, map_location=lambda storage, loc: storage)
    if "net" in loaded.keys():
        loaded = loaded["net"]
    net.load_state_dict(loaded)
    if cuda_available:
        net.cuda()
    net = torch.nn.DataParallel(net)
    net.module.eval()

    # Load image
    orig_img = io.imread(filename)
    orig_img = np.array([orig_img]) # Add one dimension
    
    # Preprocessing 
    if not config.no_sobel:
        print("Applying sobel filter")
        orig_img = custom_greyscale_numpy(orig_img, include_rgb=config.include_rgb)


    # Load to torch
    orig_img = orig_img.astype(np.float32) / 255.
    orig_img = torch.from_numpy(orig_img)
    if cuda_available:
        orig_img.cuda()

    # Swap axes - going from (1,512,512,3) to (1,3,512,512)
    orig_img = orig_img.permute(0,3,1,2)

    # Cast to float tensor
    

    # Evaluate
    with torch.no_grad():
        x_out = net(orig_img)
    x_out = x_out[0]
    if cuda_available:
        x_out = x_out.cpu()
    x_out = x_out.numpy()
    assert (x_out.max() < config.gt_k)

    # Flatten and unpack
    flat_preds = np.argmax(x_out, axis=1)
    return flat_preds[0]


def render_segmentation(predictions, out_path):
    """Render rgb image from flat prediction array
    
    Arguments:
        predictions {numpy array} -- numpy array with shape (h,w,1) containing the predictions to be rendered
        color_map {list} -- list containing colors (as numpy array with dtype np.uint8) for the segmentation classes
        out_path {str} -- Full path name of destination of rendered image
    """
    # Create folder to save if does not exist.
    folder = os.path.dirname(out_path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    

    color_map = [np.array(label, dtype=np.uint8) for key, label in cityscape_labels.iteritems()]
   
    render(predictions,
        mode="preds",
        name=(os.path.basename(out_path)),
        offset=0,
        colour_map=color_map,
        out_dir=os.path.dirname(out_path))

def generate_pointcloud(depth_fn, color=None, return_numpy=True):
    """Creates a pointcloud from a given depth image (filename) and filters the pointcloud using the given segmentation and filter class
    
    Arguments:
        depth_fn {string} -- The full path of the depth image 
        segmentation {numpy array} -- 2-d numpy array containing the segmentation information
        filter_class {int} -- Indicator of the class which should be part of the resulting point cloud
        out {string} -- Path of the output directory

    Returns:
        numpy array
    """
    depth = io.imread(depth_fn)
    grayscale = np.dot(depth[:, :, :3], [1.0, 256, 256.0 * 256.0])
    grayscale /= (256.0 * 256.0 * 256.0 - 1.0)
    return depth_to_local_point_cloud(
        grayscale, color=color, max_depth=0.05, return_numpy=return_numpy)

def filter_class(point_cloud, color_map, filter_class):
    """Filter point_cloud by filter_class and return the result as numpy array
    
    Arguments:
        point_cloud {numpy array} -- Point cloud to be filtered
        filter_class {int} -- Class indicator of the class to be filtered
    """

    class_color = cityscape_labels[filter_class]
    colored_point_cloud = np.append(point_cloud, color_map, axis=1)
    # Returs only the points in the specified rgb collor
    res = colored_point_cloud[(colored_point_cloud == class_color[0])[:,3]] # all rows which have pattern[0] at their 3rd posotion (red value)
    res = res[(res == class_color[1])[:,4]] # all rows which have pattern[1] at their 4th position (green value)
    res = res[(res == class_color[2])[:,5]] # all rows which have pattern[1] at their 4th position (green value)
    
    return res


# Function copied and modified from cara/sensor.py
def save_to_disk(array, path):
    """Save the given point cloud as a .ply file
    
    Arguments:
        array {numpy array} -- Point cloud as numpy array
        path {string} -- Full path including filename to save the point cloud at
    """

    # Create folder to save if does not exist.
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)

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
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    with open(path, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",
                        type=str,
                        help="path of dataset",
                        required=True)
    parser.add_argument("--out",
                        type=str,
                        help="path of destination",
                        required=True)
    parser.add_argument("--only_rgb",
                        help="Subdirectories for rgb and depth in root dir or single dir?",
                        action="store_true")
    parser.add_argument("--model_root",
                        type=str,
                        help="Path of trained IIC model")
    parser.add_argument("--net_name",
                        type=str,
                        help="IIC net name")

    opt = parser.parse_args()
    rgb_fns, depth_fns = get_rgb_depth_filenames(opt.root, only_rgb=opt.only_rgb)
    # print(rgb_fns)
    for index, rgb_fn in tqdm(enumerate(rgb_fns)):
        # Predict the segmentation
        pred = predict_segmentation(rgb_fn, opt.model_root, opt.net_name)

        # Upscale the 200x200 prediction to 512x512
        pred = pred.astype('uint8')
        pred = cv2.resize(pred, (512,512), interpolation=cv2.INTER_NEAREST)
        render_segmentation(pred, os.path.join(opt.out, 'seg_rgb', str(index)))

        # Get a color map
        colors = labels_to_cityscapes_palette(pred, flat=True)

        # Generate the point cloud
        point_cloud, color_map = generate_pointcloud(depth_fns[0], color=colors, return_numpy=True)

        # Filter the point cloud to extract only the class we want
        # In this case it's class 7
        filtered = filter_class(point_cloud, color_map, 7)

        # Save the resulting point cloud (without color)
        save_to_disk(
            filtered[:,0:3],
            path=os.path.join(
                opt.out,
                'point_clouds',
                str(index) + '.ply'
            )
        )

    ## Testing
    # pred = predict_segmentation(rgb_fns[0])

    # pred = pred.astype('uint8')
    # pred = cv2.resize(pred, (512,512), interpolation=cv2.INTER_NEAREST)
    # render_segmentation(pred, os.path.join(opt.out, 'seg_big_nearest'))

    # colors = labels_to_cityscapes_palette(pred, flat=True)

    # point_cloud, color_map = generate_pointcloud(depth_fns[0], color=colors, return_numpy=True)
    
    # for ind in range(13):
    #     filtered = filter_class(point_cloud, color_map, ind)
    #     save_to_disk(
    #         filtered[:,0:3],
    #         path=os.path.join(
    #             opt.out,
    #             str(ind) + '.ply'
    #         )
    #     )