import torch
import cv2
import os
from depth_anything_v2.dpt import DepthAnythingV2
import yaml
import numpy as np
import matplotlib as mpl
from PIL import Image


with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

INPUT_PATH = config["input_path"]
OUTPUT_PATH = config["output_path"]
MODEL_PATH = config["model_path"]

def get_depth(img):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model.infer_image(img)

def get_depth_image_from_matrix(depth_matrix):
    normalized_depth = cv2.normalize(depth_matrix, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = normalized_depth.astype(np.uint8)
    return depth_image

def color_map(img):
    cm_hot = mpl.colormaps.get_cmap('hot')
    img_src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_src = Image.fromarray(img).convert('L')
    img_src.thumbnail((512,512))
    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


files = [file for file in os.listdir(INPUT_PATH) if os.path.splitext(file)[1] in ['.jpg','.png']]
output_files = [os.path.join(OUTPUT_PATH,file) for file in files]
input_files = [os.path.join(INPUT_PATH,file) for file in files]
for ip, op in zip(input_files, output_files):
    print("Processing:",ip)
    depth_image = get_depth(cv2.imread(ip, cv2.IMREAD_COLOR))
    # depth_image = color_map(get_depth_image_from_matrix(depth_image))
    print(depth_image.shape)
    print(depth_image)
    cv2.imwrite(op, depth_image)
    print("Written to:",op)