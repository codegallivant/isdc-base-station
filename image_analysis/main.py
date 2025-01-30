import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO



with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

input_dir = config["input_path"]
OUTPUT_PATH = config["output_path"]
MODEL_PATH_TERRAIN = config["model_path_terrain"]
MODEL_PATH_CRATER = config["model_path_crater"]
MODEL_PATH_RIVER = config["model_path_river"]


def segformer_terrain_pred(output_dir, model_path):
    model = torch.load(model_path)
    
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")

            img = cv2.imread(img_path)
            # img = img[:,80:]

            input_img = Image.fromarray(img).resize((512, 512))
            im_arr = np.array(input_img)
            im = torch.tensor(im_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            im = im / 255.0
            im = (im - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            logits = model(im)[0]

            upsampled_logits = F.interpolate(
                logits,
                size=(512, 512),
                mode="bilinear",
                align_corners=False
            )

            predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()  

            color_map = {
                0: (0, 0, 0, 100),  # Nothing=Black
                1: (255, 0, 0, 100),  # Sand=Red
                2: (0, 255, 0, 100),  # Soil=Green
                3: (0, 0, 255, 100)  # Rock=Blue
            }

            color_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 4), dtype=np.uint8)

            for cls, color in color_map.items():
                color_mask[predicted_mask == cls] = color

            mask_img = Image.fromarray(color_mask, mode='RGBA')

            input_img_rgba = input_img.convert("RGBA")

            overlay_img = Image.alpha_composite(input_img_rgba, mask_img)

            output_img_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            overlay_img.save(output_img_path)
            print(f"Saved result image as {output_img_path}")


def yolo_crater_pred(output_dir, model_path):
    model = YOLO(model_path)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            # image = image[:, 80:]

            results = model(image)

            for result in results:
                for box in result.boxes.data.tolist():  
                    cls, conf, x1, y1, x2, y2 = box[5], box[4], box[0], box[1], box[2], box[3]
                    print(f"Class: {int(cls)}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

            annotated_image = np.array(results[0].plot())  
            output_path = os.path.join(output_dir, filename)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)  
            cv2.imwrite(output_path, annotated_image)


def yolo_river_pred(output_dir, model_path):
    model = YOLO(model_path)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            # image = image[:, 80:]

            results = model(image)

            for result in results:
                for box in result.boxes.data.tolist():  
                    cls, conf, x1, y1, x2, y2 = box[5], box[4], box[0], box[1], box[2], box[3]
                    print(f"Class: {int(cls)}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

            annotated_image = np.array(results[0].plot())  
            output_path = os.path.join(output_dir, filename)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)  
            cv2.imwrite(output_path, annotated_image)



if __name__ == '__main__':
    paths = ["terrain", "crater", "river"]
    paths = [os.path.join(OUTPUT_PATH,path) for path in paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    segformer_terrain_pred(paths[0], MODEL_PATH_TERRAIN)
    yolo_crater_pred(paths[1], MODEL_PATH_CRATER)
    yolo_river_pred(paths[2], MODEL_PATH_RIVER)   