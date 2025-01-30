#!/bin/bash
mkdir -p models
cd models
mkdir -p cone_detector depth_estimator image_analysis

cd cone_detector
echo "\nDownloading cone detector\n"
wget -O cone_detector.pt https://drive.usercontent.google.com/download?id=1PbuyIlUsB-EsHLtAZPabe9UUYz6b9xes&export=download&authuser=0&confirm=t

cd ..
cd depth_estimator
echo "\nDownloading depth estimator\n"
wget -O depth_anything_v2_small.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true
wget -O depth_anything_v2_base.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true
wget -O depth_anything_v2_large.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

cd ..
cd image_analysis
echo "\nDownloading image analysis models\n"
wget -O terrain.pt https://drive.usercontent.google.com/download?id=1-SMFYXmK0iDkQtOXPXmVT3ASZ0-gIJXL&export=download&authuser=0&confirm=t
wget -O river.pt https://drive.usercontent.google.com/download?id=1ZKKqNje5ZGoW-8_C6JP_SWrCrGkyfT2k&export=download&authuser=0&confirm=t
wget -O crater.pt https://drive.usercontent.google.com/download?id=1zjOpnloFEqX6ZuoeCkthVam4iSQOGagR&export=download&authuser=0&confirm=t
