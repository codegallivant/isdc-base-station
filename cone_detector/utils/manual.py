import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import yaml

class ImageViewerApp:
    def __init__(self, root, image_folder):
        self.root = root
        self.root.title("Image Viewer with Coordinate Capture")
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        
        self.canvas_scrollbar_x = ttk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL)
        self.canvas_scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        self.canvas_scrollbar_y = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL)
        self.canvas_scrollbar_y.grid(row=0, column=1, sticky="ns")
        
        self.canvas = tk.Canvas(self.canvas_frame, 
                                xscrollcommand=self.canvas_scrollbar_x.set,
                                yscrollcommand=self.canvas_scrollbar_y.set,
                                width=800, height=600)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        self.canvas_scrollbar_x.config(command=self.canvas.xview)
        self.canvas_scrollbar_y.config(command=self.canvas.yview)
        
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.coord_label = tk.Label(root, text="Click on image to record coordinates")
        self.coord_label.grid(row=1, column=0, pady=5)
        
        nav_frame = ttk.Frame(root)
        nav_frame.grid(row=2, column=0, pady=5)
        
        self.prev_button = ttk.Button(nav_frame, text="Previous", command=self.prev_image)
        self.prev_button.grid(row=0, column=0, padx=5)
        
        self.image_counter = ttk.Label(nav_frame, text="Image: 0/0")
        self.image_counter.grid(row=0, column=1, padx=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next", command=self.next_image)
        self.next_button.grid(row=0, column=2, padx=5)

        self.revert_button = ttk.Button(nav_frame, text= "Revert", command = self.revert)
        self.revert_button.grid(row = 1, column = 1, padx = 5)

        
        self.image_folder = image_folder
        self.image_paths = self.load_images()
        self.current_index = 0
        self.clicked_points = {}
        self.current_image_tk = None
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("q", lambda e: self.revert())
        
        self.update_display()

    def load_images(self):
        supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
        return sorted([
            os.path.join(self.image_folder, file) 
            for file in os.listdir(self.image_folder) 
            if file.lower().endswith(supported_formats)
        ])

    def update_display(self):
        if not self.image_paths:
            return
        
        self.image_counter.config(text=f"Image: {self.current_index + 1}/{len(self.image_paths)}")
        
        pil_image = Image.open(self.image_paths[self.current_index])
        
        self.current_image_tk = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image_tk)
        
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        self.prev_button.state(["!disabled"] if self.current_index > 0 else ["disabled"])
        self.next_button.state(["!disabled"] if self.current_index < len(self.image_paths) - 1 else ["disabled"])

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_display()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def on_click(self, event):
        x, y = event.x, event.y
        current_image_name = os.path.basename(self.image_paths[self.current_index])
        
        if current_image_name not in self.clicked_points:
            self.clicked_points[current_image_name] = []
        
        self.clicked_points[current_image_name].append([x, y])
        
        self.coord_label.config(text=f"Saved: X: {x}, Y: {y}")
        self.save_to_yaml()
    
    def revert(self):
        current_image_name = os.path.basename(self.image_paths[self.current_index])
        self.clicked_points[current_image_name].pop()
        self.coord_label.config(text = "Reverted")
        self.save_to_yaml()
    def save_to_yaml(self, output_file):
        with open(output_file, "w") as file:
            yaml.dump(self.clicked_points, file, default_flow_style= True)
