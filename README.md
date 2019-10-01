### Hands on Deep Learning - Project "Extending PointNet" - Subtask control

> For some reasons, the tables are not displayed properly on GitHub. Please see README.html for a formatted version of this document
## Table of contents

- [Table of contents](#table-of-contents)
- [1. Introduction](#1-introduction)
- [2. Pretrained models and testing visualizations](#2-pretrained-models-and-testing-visualizations)
- [3. Installing required packages](#3-installing-required-packages)
- [4. Running the CARLA server](#4-running-the-carla-server)
- [5. Generating a dataset using the CARLA driving simulator](#5-generating-a-dataset-using-the-carla-driving-simulator)
- [6. Preprocessing the dataset (optional)](#6-preprocessing-the-dataset-optional)
  - [Extracting road edges and lines from a dense point cloud](#extracting-road-edges-and-lines-from-a-dense-point-cloud)
- [7. Training PointNet](#7-training-pointnet)
  - [Dataset](#dataset)
  - [Training script](#training-script)
  - [Examples](#examples)
- [8. Offline testing and visualizing](#8-offline-testing-and-visualizing)
  - [Example](#example)
- [9. Testing the trained model in the CARLA driving simulator](#9-testing-the-trained-model-in-the-carla-driving-simulator)
    - [Examples](#examples-1)
  - [Setting the steering indicator during testing](#setting-the-steering-indicator-during-testing)
  - [Visualizing test runs in the simulator](#visualizing-test-runs-in-the-simulator)
    - [Example](#example-1)
  - [Automated testing](#automated-testing)
- [10. Training IIC model for unsupervised segmentation](#10-training-iic-model-for-unsupervised-segmentation)
  - [Setting up the environment](#setting-up-the-environment)
  - [Preparing the dataset](#preparing-the-dataset)
  - [Training the unsupervised segmentation model](#training-the-unsupervised-segmentation-model)
  - [Visualize the segmentation predicted by the trained IIC model](#visualize-the-segmentation-predicted-by-the-trained-iic-model)
  - [Preprocessing point clouds using the trained IIC model](#preprocessing-point-clouds-using-the-trained-iic-model)
  - [Rename files to create a valid PointNet dataset](#rename-files-to-create-a-valid-pointnet-dataset)
  

## 1. Introduction
The goal of this project was to control a car in the CARLA driving simulator using a deep learning model trained on point clouds. This repository contains
* PointNet - a modified version of the [PointNet PyTorch implementation by Fei Xia](https://github.com/fxia22/pointnet.pytorch)
* Pretrained models for dense point clouds and for LiDAR point clouds
* scripts to automatically generate training data using the [CARLA driving simulator](http://carla.org/)
* scripts to preprocess point clouds using a segmentation image
* scripts to train and evaluate the PointNet model
* scripts to test a trained model in the [CARLA driving simulator](http://carla.org/)
* scripts to visualize test runs in the simulator using GIFs
* IIC - Invariant Information clustering model for unsupervised segmentation of images [(implementation by Xu Ji)](https://github.com/xu-ji/IIC)

## 2. Pretrained models and testing visualizations

Pretrained PointNet models can be found in `trained_models/PointNet`.
There is one LiDAR model, two dense point cloud models (one trained with two steering indicators and one trained with three steering indicators) and one model trained on point clouds filtered using the [predicted segmentation of a trained IIC model](#10-training-iic-model-for-unsupervised-segmentation).

Pretrained IIC models can be found in `trained_models/IIC`.

## 3. Installing required packages

```
pip install -r requirements.txt
```

## 4. Running the CARLA server
Before being able to generate a dataset or test a model in the simulator, you must make sure a CARLA server is running. Please [install CARLA 0.8.2 according to the official documentation](https://github.com/carla-simulator/carla/releases/tag/0.8.2).
After CARLA is installed, run the following script to start the CARLA server:
```bash
python online_carla/run_carla_server.py --carla-dir /path/to/your/CARLA_0.8.2/
```

## 5. Generating a dataset using the CARLA driving simulator

Make sure that the CARLA server is running in one terminal window.
Then run the following script to generate a custom dataset using the CARLA simulator. The data will be saved in online_carla/_out.

```bash
python online_carla/generate_data.py --capture
```

Available command line arguments:

| Argument | Type | Description | 
| --- |--- | --- |
| positions | int* | List of positions on CARLA map at which datageneration should start |
| levels_of_randomness | float* | Levels of randomness added to the autopilot steering (each in a separate episode) |
| frames | int* | Length of the simulation in frames corresponding to each level of randomness. Number of arguments must be equal or greater than number of arguments for levels_of_randomness |
| capture | store_true | Without this flag, the data won't be saved. (Useful for dry-run.) Without --point_cloud or --lidar flag, only driving data and rgb images will be saved. |
| point_cloud | store_true | Save dense point clouds (preprocessed to only contain points of road edges and road lines) |
| lidar | store_true | Save LiDAR point clouds |
| force_left_and_right | store_true|This flag causes two episodes for each position to be rendered: one where the car turns left and one where it turns right (useful for T-junctions) |
| ignore_red_lights | store_true | This flags fixes the throttle at 0.5 and disables all breaks (usefil to ignore red lights) |

Example:
```bash
python online_carla/generate_data.py --positions 65 78 --levels_of_randomness 0.1 0.2 0.3 --frames 200 200 200 --capture --point_cloud --ignore-red-lights
```
This example call would lead to a generation of 6 episodes of point clouds (2 positions with each 3 levels of randomness), each containing 200 frames.


If no position/level_or_randomness/frame argument is provided, the script falls back to the following values, which can be changed in the code:

```python
positions = args.positions or [65, 78, 49, 50, 2, 5, 10, 7, 23, 40, 20, 61]
levels_of_randomness = args.randomness or [0.0, 0.2, 0.4, 0.6]
frames_per_level = args.frames or [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]

```



## 6. Preprocessing the dataset (optional)
### Extracting road edges and lines from a dense point cloud
> Note: when using the provided generate_data.py script to generate the point cloud dataset, you can skip this step, since the point clouds will already be preprocessed by the data generation script and only contain points of road edges and road lines.

The following script extracts point clouds containing only points of road edges and road lines from a dataset of dense point clouds and segmentation images:

```bash
python road_edge_projection.py --datapath=/path/to/dataset/ --out=/path/to/output/dir/ --num_episodes=10
```

The script expects the dataset to be processed to have the following structure:

```
episode_000/
    CameraSemSeg0/
        image_00000.png
        image_00001.png
        ...
    CameraDepth0/
        image_00000.png
        image_00001.png
        ...
episode_001/
    ...
```
oIf the `num_episodes` argument is not provided, the script expects the datapath argument to be the root of an episode.

The generated point clouds are saved in the directory specified with the `out` argument. It will have the following structure:
```
out/
    episode_000/
        point_cloud_00000.ply
        point_cloud_00001.ply
        ...
    episode_001/
    ...
```



## 7. Training PointNet
### Dataset
The training script expects the dataset to have one of the following three structures:

1. Plain `steering`:
    ```
    dataset/
        driving_data.csv
        point_clouds/
            point_cloud_00000.ply
            point_cloud_00001.ply
            ...
    ```

2. Multiple episodes - `steering_episodes`:
    ```
    dataset/
        episode_000/
        episode_001/
        ...
    ```
3. Nested - `steering_nested`: (this is the format the dataset will have when generated using the provided data generation script)
    ``` 
    dataset/
        pos102_/                Format pos{POSITION}_{optional: left/right}
            randomness_0/       Format irrelevant
                point_clouds/
                    driving_data.csv
                    point_cloud_00000.ply
                    point_cloud_00001.ply
                    ...
            randomness_0.1/
            ...
        pos42_/
        ...
    ```

4. Combined - `steering_combined` 
   <br>Multiple `steering_nested` or `steering_episodes` datasets. The root directories of the individual datasets must be set directly in the script.
    ```python
    # train_steering.py line 106
    datasets = [
        ('/usr/prakt/s0050/PointCloudProcessed/lidar_town02', 2),
        ('/usr/prakt/s0050/PointCloudProcessed/lidar_town01', 2),
    ]
    ```
    The individual datasets are provided using a tuple, with the first element representing the root directory and the second element the levels of nesting.  ( `1` corresponds to a `steering_episodes` dataset, `2` corresponds to a `steering_nested` dataset, `3` could be a directory containing multiple `steering_nested` datasets.)


The type of dataset can be set with the `dataset_type` argument.

### Training script

Before starting the training script, make sure that a [visdom](https://github.com/facebookresearch/visdom) server is running in order to receive training plots. If you don't want to use visdom to plot the training progress, provide the `no_visdom` flag.

``` bash
python pointnet.pytorch/utils/train_steering.py
```

Available command line arguments:
|Argument|Type|Description|Default value|
|---|---|---|---|
batch_size|int|Batch size|64|
num_points|int|Number of points to be sampled from each point cloud|1000|
nepoch|int|Number of epochs|150|
outf|str|Output folder for the trained model. The basename will be used for Visdom as well.|cls|
model|str|(optional) Path to a .pth file, from which the training will continue||
dataset|str|Path to the root directory of the dataset||
dataset_type|str|Type of provided dataset (see previous section for details)| steering|
feature_transform|store_true|If flag is set, the feature transform network will be included in the training||
only_left|store_true|If flag is set, the balanced/weighted sampler will only return examples from straight or left steering situations||
sampler|str|Which sampler to use. Options: balanced, weighted, default| weighted|
use_whole_dataset|store_true|If this flag is set, the whole provided dataset will be used for training. Otherwise, 80% will be used for training and 20% for testing||
steering_indicator|store_true|If this flag is set, a steering indicator will be included during training, which tells the model in which direction to steer when there are multiple possible paths (recommended)||
num_steering_indicators|int|Number of steering indicators to be used during training (2 = left/right, 3 = left/straight/right)||
running_mean|int|Size of a running mean to be applied on the steering values (not recommended)||
data_augmentation|store_true|If this flag is set, the training data will be augmented using random noise and random rotations (not recommended)|2|

### Examples
Training with nested dataset
```
python pointnet.pytorch/utils/train_steering.py \
 --batch_size=64 \
 --num_points=1000 \
 --dataset=/usr/prakt/s0050/PointCloudProcessed/t-junctions \
 --dataset_type="steering_nested" \
 --outf=cls/steering_nested \
 --sampler=weighted \
 --nepoch=150 \
 --steering_indicator \
 --use_whole_dataset \
 --feature_transform
```

Training with combined dataset
```
python pointnet.pytorch/utils/train_steering.py \
--sampler=weighted \
--dataset_type=steering_combined \
--outf=cls/steering_combined \
--use_whole_dataset \
--batch_size=32 \
--num_points=2500 \
--nepoch=200 \
--feature_transform
```

## 8. Offline testing and visualizing

Predictions on a provided example set can be visualized using visdom by calling the following script. A carla server must be running before calling the script.

```
python pointnet.pytorch/utils/test_model.py
```

Available command line arguments

|Argument|Type|Description|
|---|---|---|
num_points|int|Number of points to be sampled from each point cloud|
feature_transform|store_true|If flag is set, the feature transform network will be included during testing|
model|str|Model path (.pth file)|
dataset|str|Path of the dataset|
dataset_type|str|Type of dataset. Currently only support for `steering` and `steering_episodes` datasets is implemented. (See [Dataset](#dataset) section for more details)|
visdom|str|Name for visdom plots|
running_mean|int|Size of the running mean applied to ground truth steering values (not recommended)|
steering_indicator|store_true|If this flag is set, a steering indicator will be included during training, which tells the model in which direction to steer when there are multiple possible paths (recommended)|
num_steering_indicators|int|Number of steering indicators to be used during training (2 = left/right, 3 = left/straight/right)|
data_augmentation|store_true|Set this flag, if data augmentation was used during training (if --data_augmentation flag was set for training)|

### Example

```
python pointnet.pytorch/utils/test_model.py --num_points 25000 --feature_transform --model trained_models/PointNet/iic3_natural_turns.pth --dataset /usr/prakt/s0050/PointCloudProcessed/iic/ --dataset_type steering_episodes --num_episodes 1 --batch_size 1 --visdom iic
```

## 9. Testing the trained model in the CARLA driving simulator

Before testing a trained PointNet model in the CARLA simulator environment,  make sure that a CARLA server is running. (Refer to serction ["Running the CARLA server"](#3-running-the-carla-server))

A CARLA client executing the trained model can be started using the following script:

```
python online_carla/pointnet_pilot.py
```
Available command line arguments:
|Argument|Type|Description|Default value
|---|---|---|---|
model|str|Path to the .pth file of the trained model which will be used to predict the steering values||
|feature_transform|store_true|If this flag is set, the feature transform network will be included in the model. This must be set accoring to the settings during training the model.||
|use_steering_indicator|store_true|If this flag is set, a steering indicator is given to the model during testing. This flag must be set accoring to the settings during training||
|position|int|Starting position of the simulation. ||
|steering_indicator|str|Fix the steering indicator to a specific value. (Options "left"/"right")||
|frames|int|Length of the simulation in frames||
|capture|store_true|If this flag is set, the point clouds used for predicting the steering angle will be saved to `_capture`. This flag must be set to be able to [visualize the test run](#9-visualizing-test-runs-in-the-simulator) afterwards.||
|key_control|store_true|If this flag is set, it is possible to dynamically [set the steering indicator during the simulation using the keyboard](#setting-the-steering-indicator-during-testing)||
|ignore_red_lights|store_true|If this flag is set, the car will ignore red lights (by fixing throttle to 0.5 and disabling the breaks)||
|lidar|store_true|If this flag is set, LiDAR point clouds will be generated and passed to the model instead of preprocessed dense point clouds.||
|lidar_pps|int|This flag specifies the points per second sampled by the LiDAR sensor if the `lidar` flag is  set.|100.000|
lidar_fov|int|Field of view of the LiDAR sensor. Options: 360, 180.||

#### Examples

Testing the pre-trained point cloud model
```
python online_carla/pointnet_pilot.py \
--model trained_models/PointNet/point_cloud_three_indicators.pth \
--feature_transform \
--use_steering_indicator \
--position 1 \
--frames 300 \
--key_control \
--ignore_red_lights
```

Testing the pre-trained lidar model
```
python online_carla/pointnet_pilot.py \
--model trained_models/PointNet/lidar_180_fov.pth \
--feature_transform \
--use_steering_indicator \
--position 1 \
--frames 300 \
--key_control \
--ignore_red_lights \
--lidar \
--lidar_fov 180
```

Testing the IIC model
```
python online_carla/pointnet_pilot.py \
--model trained_models/PointNet/iic3_natual_turns.pth \
--feature_transform \
--use_steering_indicator \
--position 1 \
--frames 300 \
--key_control \
--ignore_red_lights \
--iic \
--iic_net_name latest \
--iic_model /usr/prakt/s0050/ss19_extendingpointnet/trained_models/IIC/3
```

### Setting the steering indicator during testing

To set the steering indicator during testing, set the `key_control` flag when calling the `pointnet_pilot.py` script. Note that the focus must remain on the terminal window (not the simulation window) for the key control to work.

Key settings:
```
a = left / 10
d = right / 01
s = straight / 00
```

### Visualizing test runs in the simulator

After capturing a test run by setting the `capture` flag, the point clouds and RGB images can be converted to two synchronous GIFs in order to "see what the model saw".

```
python visualize_run.py
```
Available command line arguments:
|Argument|Type|Description|
|---|---|---|
path|str|Root directory of a single run to viszualize.|
root|str|Directory containing multiple runs. (E.g. `_capture` directory generated by the testing script)|
lidar|store_true|Set this flag, if the point clouds are generated by a LiDAR sensor.|
point_cloud|store_true|Set this flag, if the point clouds are dense point clouds|
gif|store_true|Set this flag to generate a GIF|
images|store_true|If this flag is set, the point clouds are ignored and a GIF is generated only from the RGB images|

#### Example
```
python visualize_run.py --root _capture --point_cloud --gif
```

### Automated testing

The `online_carla/runtest.sh` bash script provides a convenient way to test and a model in multiple situations and visualize the runs without manually having to start the scripts one by one.

Model path and settings can be changed directly in the file:
``` bash
# /carla_online/runtest.sh

Model=${args[0]}
Modelpath="/usr/prakt/s0050/ss19_extendingpointnet/pointnet.pytorch/cls_final/${Model}.pth" # Change this to the .pth file of your trained model

# Settings
FOV=360 # 360 or 180
CAPTURE="--capture" # Replace with "" for a dry run
LIDAR="--lidar" # Remove this flag, 
VISUALIZE=true  # Set to false to skip the visualization steps

# Define the desired start positions
LeftPositions=(42 67 85)
RightPositions=(42 67 85)
StraightPositions=(66 138 78)
```

After defining your settings in the file, you can simply call the script to start the testing.

``` console
sh runtest.sh
```

## 10. Training IIC model for unsupervised segmentation

In this project the [Invariant Information Clustering](https://github.com/xu-ji/IIC) was used as an attempt to replace the ground truth segmentation data currently used for preprocessing the dense point clouds.

>This part of the project is encapsulated from the rest of the project - it therefore has a different python virtual environment and the provided commands should be run from the `clustering/IIC/` directory.

### Setting up the environment

To get started, run the following command in `clustering/IIC/`:
```
conda create --name iic --file requirements.txt
```


### Preparing the dataset


### Training the unsupervised segmentation model

``` bash
export CUDA_VISIBLE_DEVICES=1 && \
nohup python -m code.scripts.segmentation.segmentation_twohead_custom \
--mode IID \
--dataset Carla \
--dataset_root /usr/prakt/s0050/ss19_extendingpointnet/clustering/IIC/datasets/Carla/ \
--model_ind 1 \
--arch SegmentationNet10aTwoHead \
--num_epochs 4800 \
--lr 0.000001 \
--lamb_A 1.0 \
--lamb_B 1.0 \
--num_sub_heads 1 \
--batch_sz 36 \
--num_dataloaders 1 \
--output_k_A 36 \
--output_k_B 13 \
--gt_k 13 \
--input_sz 200 \
--half_T_side_spar se_min 0 \
--half_T_side_sparse_max 0 \
--half_T_side_dense 5  \
--include_rgb \
--no_sobel \
--jitter_brightness 0.1 \
--jitter_contrast 0.1 \
--jitter_saturation 0.1 \
--jitter_hue 0.1  \
--use_uncollapsed_loss \
--batchnorm_track \
--out_root /usr/prakt/s0050/IIC_selfmade \
> train.out & 
```

### Visualize the segmentation predicted by the trained IIC model

Run the following script from `clustering/IIC` to render visualizations of the segmentation predicted by the IIC model:
```
python -m code.scripts.segmentation.analysis.render_general \
--model_inds 3 \
--net_name latest
```

### Preprocessing point clouds using the trained IIC model
Run the following script from the root of this repository (after adjusting the command line arguments) to extract the points corresponding to the segmentation class corresponding to road lines and sidewalk from a dense point cloud using the segmentation provided by the trained IIC model:

```
python -m processing.processing \
--root /storage/group/hodl4cv/lopc/control/episode_000/ \
--out /usr/prakt/s0050/carla_iic_processed/pointcloud \
--model_root /usr/prakt/s0050/IIC_selfmade/1 \
--net_name latest
```

### Rename files to create a valid PointNet dataset

Run the following script from the root of this repository (after adjusting the command line arguments) to rename the processed point clouds to create a valid dataset to train the PointNet model

```
python processing/rename_files.py --root /usr/prakt/s0050/PointCloudProcessed/iic/
```
