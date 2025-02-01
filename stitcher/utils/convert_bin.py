import os
import struct
import argparse
import numpy as np
import cv2
import zstandard as zstd
from tqdm import tqdm
from pathlib import Path

def read_frame_from_file(file):
    """Read a single frame and its GPS data from the binary file."""
    try:
        # Read GPS data
        gps_data = struct.unpack('ddd', file.read(24))  # 3 doubles for lat, lon, alt
        
        # Read frame size
        size = struct.unpack('I', file.read(4))[0]
        
        # Read frame data
        data = file.read(size)
        
        return gps_data, data
    except struct.error:
        return None, None

def decompress_bin_file(input_file, output_dir, is_depth=False):
    """Decompress a binary file containing frames and GPS data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize decompressor for depth frames
    if is_depth:
        dctx = zstd.ZstdDecompressor()
    
    with open(input_file, 'rb') as f:
        frame_count = 0
        
        # Get file size for progress bar
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        
        with tqdm(total=file_size, desc=f"Decompressing {Path(input_file).name}") as pbar:
            while f.tell() < file_size:
                result = read_frame_from_file(f)
                if result[0] is None:
                    break
                    
                gps_data, frame_data = result
                
                try:
                    # Create filename with GPS coordinates
                    filename = os.path.splitext(Path(input_file).name)[0]
                    
                    if is_depth:
                        # Decompress depth frame (ZSTD)
                        decompressed_data = dctx.decompress(frame_data)
                        depth_array = np.frombuffer(decompressed_data, dtype=np.uint16)
                        depth_array = depth_array.reshape(480, 848)  # Known dimensions
                        
                        # Save as 16-bit PNG
                        cv2.imwrite(os.path.join(output_dir, filename + ".png"), depth_array)
                    else:
                        # Decompress RGB frame (WebP)
                        nparr = np.frombuffer(frame_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            cv2.imwrite(os.path.join(output_dir, filename + ".png"), img)
                        else:
                            print(f"Warning: Could not decode frame {frame_count} in {input_file}")
                    
                    frame_count += 1
                    pbar.update(f.tell() - pbar.n)
                except Exception as e:
                    print(f"Error processing frame {frame_count} in {input_file}: {e}")
                    continue
    
    return frame_count

def convert_bin_files(input_dir, output_dir):
    """Process all bin files in the input directory."""
    input_path = Path(input_dir)
    
    # Create output directories
    depth_dir = Path(output_dir) / "depth"
    rgb_dir = Path(output_dir) / "rgb"
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    
    # Find all bin files
    depth_files = sorted(input_path.glob("depth_*.bin"))
    rgb_files = sorted(input_path.glob("rgb_*.bin"))
    
    total_depth_frames = 0
    total_rgb_frames = 0
    
    print("\nProcessing depth files...")
    for depth_file in depth_files:
        frames = decompress_bin_file(str(depth_file), str(depth_dir), is_depth=True)
        total_depth_frames += frames
    
    print("\nProcessing RGB files...")
    for rgb_file in rgb_files:
        frames = decompress_bin_file(str(rgb_file), str(rgb_dir), is_depth=False)
        total_rgb_frames += frames
    
    print(f"\nSummary:")
    print(f"Processed {len(depth_files)} depth files ({total_depth_frames} frames)")
    print(f"Processed {len(rgb_files)} RGB files ({total_rgb_frames} frames)")
    print(f"Output saved to:")
    print(f"  Depth images: {depth_dir}")
    print(f"  RGB images: {rgb_dir}")

def main():
    parser = argparse.ArgumentParser(description='Decompress binary files containing frames and GPS data')
    parser.add_argument('input_dir', help='Input directory containing bin files')
    parser.add_argument('output_dir', help='Output directory for decompressed frames')
    
    args = parser.parse_args()
    
    convert_bin_files(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()