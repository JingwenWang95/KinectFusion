# KinectFusion implemented in Python with PyTorch

<img src="images/kinfu.gif" height=250px align="right"/>

This is a lightweight Python implementation of KinectFusion. All the core functions (TSDF volume, frame-to-model tracking, point-to-plane ICP, raycasting, TSDF fusion, etc.) are implemented using pure PyTorch, i.e. no CUDA kernels. 
Although without any custom CUDA functions, the system could still run at a fairly fast speed: The demo reconstructs the TUM fr1_desk sequence into a 225 x 171 x 111 TSDF volume with 2cm resolution at round 17 FPS with a single RTX-2080 GPU (~1.5 FPS in CPU mode) 

Note that this project is mainly for study purpose, and is not fully optimized for accurate camera tracking.

## Requirements
The core functionalities were implemented in PyTorch (1.10). Open3D (0.14.0) is used for visualisation. Other important dependancies include:

* numpy==1.21.2
* opencv-python==4.5.5
* imageio==2.14.1
* scikit-image==0.19.1
* trimesh==3.9.43

You can create an anaconda environment called `kinfu` with the required dependencies by running:
```
conda env create -f environment.yml
conda activate kinfu
```

## Data Preparation
The code was tested on [TUM dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download). After downloading the raw sequences, you will need to run the pre-processing script under `dataset/`. For example:

```
python dataset/preprocess.py --config configs/fr1_desk.yaml
```

There are some example config files under `configs/` which correspond to different sequences. You need to replace `data_root` to your own sequence directory before running the script. 
After running the script a new directory `processed/` will appear under your sequence directory. 

## Run
After obtaining the processed sequence, you can simply run

```
python kinfu.py --config configs/fr1_desk.yaml
```
Or

```
python kinfu_gui.py --config configs/fr1_desk.yaml
```

If you want to visualize the tracking and reconstruction process on-the-fly.

## Acknowledgement
Part of the tracking code was borrowed and modified from [DeepIC](https://github.com/lvzhaoyang/DeeperInverseCompositionalAlgorithm). Also thank [Binbin Xu](https://github.com/binbin-xu) for his PyTorch implementation of TSDF volume.
