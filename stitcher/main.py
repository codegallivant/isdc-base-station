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
    INPUT_LOG_DIR = cfg["input_log_path"]
    ALGORITHM_CODE = int(cfg["algorithm"])
    MATCHER_CODE = int(cfg["matcher"])

RGB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "rgb")
DEPTH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "depth")
RGB_INPUT_LOG_DIR = os.path.join(INPUT_LOG_DIR, "rgb")
DEPTH_INPUT_LOG_DIR = os.path.join(INPUT_LOG_DIR, 'depth')

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


    def stitch(self, rgb_image, depth_image = None):
        first_stitch = self.rgb_stitcher.stitch_count == 0
        start = time.time()
        tf = self.rgb_stitcher.stitch_consecutive(rgb_image)
        end = time.time()
        print("RGB Stitching time:", end-start)
        if not (depth_image is None):
            if not first_stitch:
                if self.rgb_stitcher.stitch_count > self.depth_stitcher.stitch_count:
                    self.depth_stitcher.save_and_reset()
            start = time.time()
            self.depth_stitcher.stitch_consecutive(depth_image, tf)
            end = time.time()
            print("Depth Stitching time:", end-start)
        return tf


utils.reset_directory(OUTPUT_DIR)
utils.reset_directory(RGB_OUTPUT_DIR)
utils.reset_directory(DEPTH_OUTPUT_DIR)
utils.reset_directory(INPUT_LOG_DIR)
utils.reset_directory(RGB_INPUT_LOG_DIR)
utils.reset_directory(DEPTH_INPUT_LOG_DIR)


sync_st = SynchronizedStitcher(
    rgb_args={
        "output_dir": RGB_OUTPUT_DIR,
        "matcher": MATCHER_CODE,
        "algorithm": ALGORITHM_CODE,
        "backup_interval": BACKUP_INTERVAL,
        "consecutive_range": 2,
        "blender": 1,
        "log_level": LOG_LEVEL
    },
    depth_args={
        "output_dir": DEPTH_OUTPUT_DIR,
        # "blend_processor": lambda x: utils.color_map(x),
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
        
        self.depth_queue = deque()
        self.rgb_queue = deque()
        self.queue_lock = Lock()
        
        self.dctx = zstd.ZstdDecompressor()
        
        self.width = width
        self.height = height
        self.depth_port = depth_port
        self.rgb_port = rgb_port
        self.address = address

        # Create directories for saving frames
        self.raw_frames_dir = INPUT_LOG_DIR
        self.rgb_dir = RGB_INPUT_LOG_DIR
        self.depth_dir = DEPTH_INPUT_LOG_DIR
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        self.frame_counter = 0

    def _extract_gps_and_data(self, message):
        # Extract GPS data (3 doubles) from the start of the message
        gps_size = 3 * 8  # 3 doubles (latitude, longitude, altitude)
        gps_data = np.frombuffer(message[:gps_size], dtype=np.float64)
        latitude, longitude, altitude = gps_data
        
        # Extract the actual frame data after GPS data
        frame_data = message[gps_size:]
        
        return (latitude, longitude, altitude), frame_data

    def _save_frame(self, frame, gps, is_depth=False):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        frame_num = f"{self.frame_counter:06d}"
        
        if is_depth:
            save_dir = self.depth_dir
            filename = f"depth_{frame_num}__{str(gps[0]).replace('.', '_')}__{str(gps[1]).replace('.', '_')}__{str(gps[2]).replace('.', '_')}.png"
            # Normalize depth for visualization
            normalized = cv2.normalize(frame, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        else:
            save_dir = self.rgb_dir
            filename = f"rgb_{frame_num}__{str(gps[0]).replace('.', '_')}__{str(gps[1]).replace('.', '_')}__{str(gps[2]).replace('.', '_')}.png"
        
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def _depth_receiver(self):
        socket = self.context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.address}:{self.depth_port}")
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        while self.running:
            try:
                data = socket.recv(zmq.NOBLOCK)
                gps, frame_data = self._extract_gps_and_data(data)
                with self.queue_lock:
                    self.depth_queue.append((gps, frame_data))
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"Depth receiver error: {str(e)}")

    def _rgb_receiver(self):
        socket = self.context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.address}:{self.rgb_port}")
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        while self.running:
            try:
                data = socket.recv(zmq.NOBLOCK)
                gps, frame_data = self._extract_gps_and_data(data)
                with self.queue_lock:
                    self.rgb_queue.append((gps, frame_data))
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"RGB receiver error: {str(e)}")

    def _process_frames(self):
        while self.running or self.depth_queue or self.rgb_queue:
            try:
                depth_frame = None
                rgb_frame = None
                gps_data = None
                
                with self.queue_lock:
                    if self.depth_queue and self.rgb_queue:
                        depth_gps, depth_data = self.depth_queue.popleft()
                        rgb_gps, rgb_data = self.rgb_queue.popleft()
                        
                        try:
                            # Decompress depth frame using ZSTD
                            depth_frame = np.frombuffer(
                                self.dctx.decompress(depth_data),
                                dtype=np.uint16
                            ).reshape((self.height, self.width))
                            
                            # Decode RGB frame from WebP format
                            rgb_frame = cv2.imdecode(
                                np.frombuffer(rgb_data, dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )
                            
                            # Use GPS data from either frame (they should be the same)
                            gps_data = depth_gps

                            # Save frames
                            if depth_frame is not None:
                                self._save_frame(depth_frame, gps_data, is_depth=True)
                            if rgb_frame is not None:
                                self._save_frame(rgb_frame, gps_data, is_depth=False)
                            
                            self.frame_counter += 1
                            
                        except Exception as e:
                            print(f"Frame processing error: {str(e)}")
                            continue
                
                if depth_frame is not None and rgb_frame is not None:
                    self.callback(depth=depth_frame, rgb=rgb_frame, gps=gps_data)
                    
            except Exception as e:
                print(f"Processing loop error: {str(e)}")
                
            time.sleep(0.001)

    def start(self):
        """Start the frame receiver threads"""
        try:
            self.running = True
            # Start network receiver threads
            self.depth_thread = Thread(target=self._depth_receiver, daemon=True)
            self.rgb_thread = Thread(target=self._rgb_receiver, daemon=True)
            self.process_thread = Thread(target=self._process_frames, daemon=True)
            
            self.depth_thread.start()
            self.rgb_thread.start()
            self.process_thread.start()
            
            print("Frame receiver started successfully")
            
        except Exception as e:
            print(f"Error starting frame receiver: {str(e)}")
            self.stop()

    def stop(self):
        """Stop the frame receiver and clean up"""
        try:
            self.running = False
            
            # Wait for threads to finish
            if hasattr(self, 'depth_thread'):
                self.depth_thread.join(timeout=2.0)
            if hasattr(self, 'rgb_thread'):
                self.rgb_thread.join(timeout=2.0)
            if hasattr(self, 'process_thread'):
                self.process_thread.join(timeout=2.0)
                
            self.context.term()
            print("Frame receiver stopped successfully")
            
        except Exception as e:
            print(f"Error stopping frame receiver: {str(e)}")

            
processing_times = list()
src_pt = None
prev_shape = None
def handle_frames(depth, rgb, gps=None):
    global src_pt, prev_shape
    processing_times.append(time.time())
    if(len(processing_times) > 10):
        processing_times.pop(0)
    print(f"Average processing time (last 10 frames): {np.mean(np.diff(processing_times))}")
    if gps:
        print(f"GPS Data - Lat: {gps[0]}, Long: {gps[1]}, Alt: {gps[2]}")
    if not (rgb is None):
        if not (depth is None):
            depth = depth[:,90:]
        rgb = rgb[:,90:]
        print(f"Processing depth({depth.shape if not (depth is None) else depth}) and RGB{rgb.shape} frame:")
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
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print("Exception:", str(e))
    finally:
        receiver.stop()
        sync_st.depth_stitcher.save_last_stitch()
        sync_st.rgb_stitcher.save_last_stitch()
        save_src_pt(src_pt)


elif MODE == 1: # files:
    if UNPACK:
        convert_bin_files(INPUT_DIR, INPUT_LOG_DIR)
        INPUT_DIR = INPUT_LOG_DIR
        rgb_files = [os.path.join(RGB_INPUT_LOG_DIR, f) for f in os.listdir(RGB_INPUT_LOG_DIR)]
        depth_files = [os.path.join(DEPTH_INPUT_LOG_DIR, f) for f in os.listdir(DEPTH_INPUT_LOG_DIR)]
    else:
        all_files = os.listdir(INPUT_DIR)
        print(all_files)    
        def sortkey(s):
            print(s)
            return int(s.split("_")[1])
        rgb_files = sorted(all_files, key=lambda x: sortkey(x))
        rgb_files = [os.path.join(INPUT_DIR, f) for f in rgb_files]
        # rgb_files = sorted([file for file in all_files if "color" in os.path.basename(file) or "rgb" in os.path.basename(file)], key = lambda x: sortkey(x))
        depth_files = sorted([file for file in all_files if "depth" in os.path.basename(file)], key= lambda x: sortkey(x))
        depth_files = [os.path.join(INPUT_DIR, f) for f in depth_files]
    if len(depth_files) == 0:
        depth_files = None
        print("No depth files found. Not stitching depth.")
    try:
        for i in range(len(rgb_files)):
            rgb_image = cv2.imread(rgb_files[i])

            if not (depth_files is None):
                depth_matrix = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)        
            else:
                depth_matrix = None
            # sync_st.stitch(rgb_image, depth_matrix)
            handle_frames(depth_matrix, rgb_image)
    except Exception as e:
        print("Exception", str(e))
        sync_st.depth_stitcher.save_last_stitch()
        sync_st.rgb_stitcher.save_last_stitch()
        save_src_pt(src_pt)
