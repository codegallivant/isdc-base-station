import cv2
from image_stitcher.stitcher import ConsecutiveStitcher
import logging
import yaml
import os
import utils.utils as utils
import zmq
import numpy as np
import time
import zstandard as zstd
from threading import Thread, Lock
from collections import deque
from utils.convert_bin import convert_bin_files


# Read the configuration file
logging.info("Reading configuration file")
with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    MODE = cfg["mode"]
    INPUT_DIR = cfg["directory"]["input_path"]
    UNPACK = cfg["directory"]["unpack"]
    ADDRESS = cfg["socket"]["address"]
    RGB_PORT = cfg["socket"]["rgb_port"]
    DEPTH_PORT = cfg["socket"]["depth_port"]
    WIDTH = cfg["image"]["width"]
    HEIGHT = cfg["image"]["height"]
    OUTPUT_DIR = cfg["output_path_stitch"]
    SRC_PT_PATH = cfg["output_path_src_pt"]
    LOG_LEVEL = cfg["log_level"]
    BACKUP_INTERVAL = cfg["backup_interval"]

RGB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "rgb")
DEPTH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "depth")


def save_src_pt(src_pt):
    point_dict = {
        "source_point": src_pt
    }
    with open(SRC_PT_PATH, 'w') as file:
        yaml.dump(point_dict, file, default_flow_style=False)
    print(f"Saved source points to {SRC_PT_PATH}")


class SynchronizedStitcher:
    def __init__(self, rgb_args=None, depth_args=None, points=None):
        self.rgb_stitcher = ConsecutiveStitcher(**rgb_args)
        self.depth_stitcher = ConsecutiveStitcher(**depth_args)


    def stitch(self, rgb_image, depth_image):
        first_stitch = self.rgb_stitcher.stitch_count == 0
        start = time.time()
        tf = self.rgb_stitcher.stitch_consecutive(rgb_image)
        end = time.time()
        print("RGB Stitching time:", end-start)
        if not first_stitch:
            if self.rgb_stitcher.stitch_count > self.depth_stitcher.stitch_count:
                self.depth_stitcher.save_and_reset()
        start = time.time()
        self.depth_stitcher.stitch_consecutive(utils.get_depth_image_from_matrix(depth_image), tf)
        end = time.time()
        print("Depth Stitching time:", end-start)
        return tf


utils.reset_directory(OUTPUT_DIR)
utils.reset_directory(RGB_OUTPUT_DIR)
utils.reset_directory(DEPTH_OUTPUT_DIR)


sync_st = SynchronizedStitcher(
    rgb_args={
        "output_dir": RGB_OUTPUT_DIR,
        "matcher": 0,
        "algorithm": 2,
        "backup_interval": BACKUP_INTERVAL,
        "consecutive_range": 3,
        "blender": 1,
        "log_level": LOG_LEVEL
    },
    depth_args={
        "output_dir": DEPTH_OUTPUT_DIR,
        "blend_processor": lambda x: utils.color_map(x),
        "backup_interval": BACKUP_INTERVAL,
        "blender": 0,
        "ref_image_contrib": 0.5,
        "log_level": LOG_LEVEL
    }
)

class FrameReceiver:
    def __init__(self, callback, width, height, depth_port, rgb_port, address):
        self.context = zmq.Context()
        self.callback = callback
        self.running = False
        
        # Thread-safe queues for frame data
        self.depth_queue = deque()
        self.rgb_queue = deque()
        self.queue_lock = Lock()
        
        # Zstandard decompressor for depth
        self.dctx = zstd.ZstdDecompressor()

        self.width = width
        self.height = height
        self.depth_port = depth_port
        self.rgb_port = rgb_port
        self.address = address


    def _depth_receiver(self):
        socket = self.context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.address}:{self.depth_port}")
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        while self.running:
            try:
                data = socket.recv(zmq.NOBLOCK)
                with self.queue_lock:
                    self.depth_queue.append(data)
            except zmq.Again:
                time.sleep(0.001)  # Prevent busy wait

    def _rgb_receiver(self):
        socket = self.context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.address}:{self.rgb_port}")
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        while self.running:
            try:
                data = socket.recv(zmq.NOBLOCK)
                with self.queue_lock:
                    self.rgb_queue.append(data)
            except zmq.Again:
                time.sleep(0.001)

    def _process_frames(self):
        while self.running or self.depth_queue or self.rgb_queue:
            depth_frame = None
            rgb_frame = None
            
            # Get latest frames
            with self.queue_lock:
                if self.depth_queue and self.rgb_queue:
                    depth_data = self.depth_queue.popleft()
                    depth_frame = np.frombuffer(
                        self.dctx.decompress(depth_data),
                        dtype=np.uint16
                    ).reshape((self.height, self.width))
                
                    rgb_data = self.rgb_queue.popleft()
                    rgb_frame = cv2.imdecode(
                        np.frombuffer(rgb_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
            
            # Callback with any available frames
            if depth_frame is not None and rgb_frame is not None:
                self.callback(depth=depth_frame, rgb=rgb_frame)
                
            time.sleep(0.001)  # Prevent CPU overload

    def start(self):
        self.running = True
        # Start network receiver threads
        Thread(target=self._depth_receiver, daemon=True).start()
        Thread(target=self._rgb_receiver, daemon=True).start()
        # Start processing thread
        Thread(target=self._process_frames, daemon=True).start()

    def stop(self):
        self.running = False
        self.context.term()

processing_times = list()
src_pt = None
prev_shape = None
def handle_frames(depth, rgb):
    global src_pt, prev_shape
    processing_times.append(time.time())
    if(len(processing_times) > 10):
        processing_times.pop(0)
    print(f"Average processing time (last 10 frames): {np.mean(np.diff(processing_times))}")
    if not (depth is None) and not (rgb is None):
        print(f"Processing depth({depth.shape}) and RGB{rgb.shape} frame:")
        tf = sync_st.stitch(rgb, depth)
        if sync_st.rgb_stitcher.stitch_count == 0:
            if src_pt is None:
                src_pt = [rgb.shape[1]//2, rgb.shape[0]//2]
            else:
                src_pt = [src_pt[0] + (sync_st.rgb_stitcher.refs[0].shape[1] - prev_shape[1]), src_pt[1] + (sync_st.rgb_stitcher.refs[0].shape[0] - prev_shape[0])]
            prev_shape = sync_st.rgb_stitcher.refs[0].shape
            if sync_st.rgb_stitcher.image_count % BACKUP_INTERVAL == 0:
                save_src_pt(src_pt)
        else:
            save_src_pt(src_pt)


if MODE == 0: # live connection
    times = list()
    receiver = FrameReceiver(handle_frames, WIDTH, HEIGHT, DEPTH_PORT, RGB_PORT, ADDRESS)
    receiver.start()
    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except Exception as e:
        print("Exception:", str(e))
        receiver.stop()
        sync_st.depth_stitcher.save_last_stitch()
        sync_st.rgb_stitcher.save_last_stitch()
        save_src_pt(src_pt)


elif MODE == 1: # files:
    if UNPACK:
        convert_bin_files(INPUT_DIR, "unpacked_input", WIDTH, HEIGHT)
        INPUT_DIR = "unpacked_input"
    all_files = os.listdir(INPUT_DIR)
    print(all_files)    
    def sortkey(s):
        return int(s.split("_")[1])
    rgb_files = sorted([file for file in all_files if "color" in os.path.basename(file) or "rgb" in os.path.basename(file)], key = lambda x: sortkey(x))
    depth_files = sorted([file for file in all_files if "depth" in os.path.basename(file)], key= lambda x: sortkey(x))

    try:
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            rgb_file_path = os.path.join(INPUT_DIR, rgb_file)
            rgb_image = cv2.imread(rgb_file_path)

            depth_file_path = os.path.join(INPUT_DIR, depth_file)
            depth_matrix = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)        

            # sync_st.stitch(rgb_image, depth_matrix)
            handle_frames(depth_matrix, rgb_image)
    except Exception as e:
        print("Exception", str(e))
        sync_st.depth_stitcher.save_last_stitch()
        sync_st.rgb_stitcher.save_last_stitch()
        save_src_pt(src_pt)
