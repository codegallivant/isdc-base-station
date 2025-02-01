import cv2
import os
import yaml
from ultralytics import YOLO


def get_images(src_dir):
    return [os.path.join(src_dir, image) for image in os.listdir(src_dir) if image.endswith(".png")]

def run_YOLO(model, imgs):
    results_dict = {}
    for img_path in imgs:
        image_name = os.path.basename(img_path)
        frame = cv2.imread(img_path)
        results = model(frame, conf=0.5)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append([x1, y1,x2, y2])
        results_dict[image_name] = boxes
    return results_dict

def save_to_yaml(data, output_path):
    with open(output_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=True)
