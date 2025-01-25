import cv2
import matplotlib as mpl
import numpy as np
from PIL import Image
import os
import glob
import zmq
import zstandard as zstd


def color_map(img):
    cm_hot = mpl.cm.get_cmap('hot')
    img_src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_src = Image.fromarray(img).convert('L')
    img_src.thumbnail((512,512))
    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def get_depth_image_from_matrix(depth_matrix):
    normalized_depth = cv2.normalize(depth_matrix, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = normalized_depth.astype(np.uint8)
    return depth_image


def save_depth_matrix_as_png(depth_matrix, output_file):
    depth_image = get_depth_image_from_matrix(depth_matrix)    
    cv2.imwrite(output_file, depth_image)


def reset_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        delfiles = glob.glob(path+"/*")
        for f in delfiles:
            if os.path.isfile(f):
                os.remove(f)


def create_subscriber(address, port):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect (f"tcp://{address}:{port}")
    socket.subscribe("")
    return socket


def receive_mat(decompressor, socket, dtype, width, height, channels):
    original_size = width * height * channels * np.dtype(dtype).itemsize
    compressed_data = socket.recv()
    # return compressed_data
    decompressed_data = decompressor.decompress(compressed_data)
    if channels == 1:
        image = np.frombuffer(decompressed_data, dtype=dtype).reshape((height, width))
    else:
        image = np.frombuffer(decompressed_data, dtype=dtype).reshape((height, width, channels))
    return image