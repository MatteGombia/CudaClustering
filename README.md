# Clustering Node

## Overview
ROS2 node for performing GPU-based point cloud clustering and optional filtering or segmentation.

## Features
- Voxel-based downsampling
- Dimension filtering
- Plane segmentation
- Configurable via YAML parameters

## Requirements
- CUDA toolkit
- PCL
- ROS2 Humble

## Configuration
- Enables or disables Z filtering, segmentation, voxel grid.
- Min/max cluster size, voxel sizes, and distance thresholds.

## Build
cd cuda_clustering
colcon build

## Usage
Edit parameters in config/clustering.yaml.

## Run
source install/setup.bash
ros2 launch clustering cuda_clustering_launch.py

## TO FIX 
1. The clustering stops working while the filter on Z is activated 
