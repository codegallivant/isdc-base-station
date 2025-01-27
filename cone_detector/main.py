import yaml


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODE = config["mode"]
INPUT_PATH = config["input_path"]
OUTPUT_PATH = config["output_path"]
MODEL_PATH = config["model_path"]


if MODE == 0:
    import tkinter as tk
    from utils.manual import ImageViewerApp
    root = tk.Tk()
    root.geometry("800x1000")
    app = ImageViewerApp(root, INPUT_PATH)
    root.mainloop()
elif MODE == 1:
    from utils.yolo import YOLO, get_images, run_YOLO, save_to_yaml
    model = YOLO(MODEL_PATH)
    imgs = get_images(INPUT_PATH)
    results_dict = run_YOLO(model, imgs)
    save_to_yaml(results_dict, OUTPUT_PATH)
    print(f"Bounding box coordinates saved to {OUTPUT_PATH}")