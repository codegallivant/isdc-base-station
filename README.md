# isdc-base-station

## Description
This repository contains the software stack for completing the navigation and science missions of International Space Drone Competition 2025 (ISDC 2025) held at BITS Pilani, Goa Campus on 2nd February 2024. The team placed 4th overall and 1st amongst fully autonomous drones.
  
A Raspberry Pi 5 (8GB) and a Realsense D435 camera were mounted on the drone. The base station receives sensor data arriving from the Pi, which is stitched in real-time to create RGB and depth maps of the covered area. Depth estimation, terrain analysis and cone detection can also be conducted on the covered area. All configuration (such as order of packages, input & output directories, modes etc) can be altered easily by setting configuration files.

## Download
1. Input can be received from either a directory or a zmq subscription socket, depending on the mode set in ``config.yaml``. The [linked repository](https://github.com/codegallivant/realsense-pub-interface) contains the zmq publisher, responsible for compressing and transmitting sensor data from a Realsense D4xx camera.
```bash
# For publishing sensor data
git clone git@github.com:codegallivant/realsense-pub-interface.git
```
2. Clone 
```bash
# For receiving and processing sensor data
git clone --depth 1 git@github.com:codegallivant/isdc-base-station.git --recurse-submodules
```
3. Download models (if using image\_analysis or depth\_estimator packages)
```bash
./download_models.sh 
```

## Configuring and running
1. Set ``config.yaml`` and run
```bash
python3 configure.py
```
This will automatically write package config to each directory and create a run script.
2. Run
```bash
./run.sh
```

## Packages
1. ``stitcher``:
Rapidly stitches sequentially arriving RGB images and uses the transformation calculated to stitch corresponding depth images, resulting in an RGB map and depth map. Stitching is dependent on the [image-stitcher](https://github.com/codegallivant/image-stitcher) submodule.
2. ``depth_estimator``:
Uses [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) for estimating depth in RGB images. Used in case data from the depth camera is not available.
3. ``image_analysis``:
Predicts craters, river valleys and terrain from images.
4. ``cone_pixel``:
Detects pixel coordinates of cones, that are predicted using either a YOLO model or manually, depending on the mode set.
5. ``cone_gps``:
Determines GPS coordinates of cones. Note: GPS coordinates of the drone are deciphered from the image file name itself.
