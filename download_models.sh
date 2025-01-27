#!/bin/bash
mkdir -p models
cd models
mkdir -p cone_detector depth_estimator
cd cone_detector
wget -O cone_detector.pt https://drive.usercontent.google.com/download?id=1PbuyIlUsB-EsHLtAZPabe9UUYz6b9xes&export=download&authuser=0&confirm=t
cd ..
cd depth_estimator
wget -O depth_anything_v2_small.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true
wget -O depth_anything_v2_base.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true
wget -O depth_anything_v2_large.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true