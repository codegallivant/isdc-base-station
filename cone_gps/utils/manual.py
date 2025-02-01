import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import yaml
import cv2
import numpy as np
from math import radians, degrees, sin, cos, asin, atan2, sqrt, pi
import random


def get_gps_data(image_path):
    print(image_path)
    gps_coords = image_path.split("/")[-1]
    gps_coords = os.path.splitext(image_path)[0].split("__")[1:]
    gps_coords = [c.replace("_",".") for c in gps_coords]
    gps_coords = [float(c) for c in gps_coords]
    return gps_coords[0], gps_coords[1], gps_coords[2]

class ImageViewerApp:
    def __init__(self, root, image_folder, output_path, annotated_output_folder):
        self.sensor_width = 3.984
        self.focal_len = 1.93
        self.img_width = 848
        self.img_height = 480
        self.yaw = 0
        
        self.root = root
        self.root.title("GPS Image Viewer with Coordinate Capture")
        
        # Configure main window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(self.main_frame, width=800, height=600)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        self.scrollbar_y = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self.main_frame, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)
        
        # Control panel frame
        control_frame = ttk.Frame(root)
        control_frame.grid(row=1, column=0, pady=5, sticky="ew")
        
        # Zoom controls
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.grid(row=0, column=0, padx=5)
        
        ttk.Label(zoom_frame, text="Zoom:").grid(row=0, column=0, padx=5)
        self.zoom_out = ttk.Button(zoom_frame, text="-", command=lambda: self.zoom_image("out"), width=3)
        self.zoom_out.grid(row=0, column=1, padx=2)
        self.zoom_reset = ttk.Button(zoom_frame, text="100%", command=self.reset_zoom, width=6)
        self.zoom_reset.grid(row=0, column=2, padx=2)
        self.zoom_in = ttk.Button(zoom_frame, text="+", command=lambda: self.zoom_image("in"), width=3)
        self.zoom_in.grid(row=0, column=3, padx=2)
        
        # Fit button
        self.fit_button = ttk.Button(zoom_frame, text="Fit to Window", command=self.fit_to_window)
        self.fit_button.grid(row=0, column=4, padx=5)
        
        # Coordinate display
        self.coord_label = ttk.Label(control_frame, text="Click on image to record GPS coordinates")
        self.coord_label.grid(row=0, column=1, padx=20)
        
        # Navigation controls
        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=0, column=2, padx=5)
        
        self.prev_button = ttk.Button(nav_frame, text="Previous (←)", command=self.prev_image)
        self.prev_button.grid(row=0, column=0, padx=5)
        
        self.image_counter = ttk.Label(nav_frame, text="Image: 0/0")
        self.image_counter.grid(row=0, column=1, padx=20)
        
        self.next_button = ttk.Button(nav_frame, text="Next (→)", command=self.next_image)
        self.next_button.grid(row=0, column=2, padx=5)
        
        self.revert_button = ttk.Button(nav_frame, text="Revert (Q)", command=self.revert)
        self.revert_button.grid(row=0, column=3, padx=5)
        
        # Initialize variables
        self.image_folder = image_folder
        self.output_path = output_path
        self.annotated_output_folder = annotated_output_folder
        os.makedirs(self.annotated_output_folder, exist_ok=True)
        self.image_paths = self.load_images()
        self.current_index = 0
        self.clicked_points = {}
        self.current_image = None
        self.current_image_tk = None
        self.zoom_factor = 1.0
        self.base_zoom = 1.0
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("q", lambda e: self.revert())
        self.root.bind("<Control-minus>", lambda e: self.zoom_image("out"))
        self.root.bind("<Control-plus>", lambda e: self.zoom_image("in"))
        self.root.bind("<Control-equal>", lambda e: self.zoom_image("in"))
        self.root.bind("<Control-0>", lambda e: self.reset_zoom())
        
        # Pan binding
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)
        
        # Wait for window to be ready before displaying first image
        self.root.update_idletasks()
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
        
        # Load GPS data
        self.lat, self.lon, self.alt = get_gps_data(self.image_paths[self.current_index])
        if self.lat is None or self.lon is None:
            print("Taking Default Values:")
            self.lat, self.lon = 28.3591, 75.5882
        
        # Load and display image
        self.current_image = Image.open(self.image_paths[self.current_index])
        self.original_size = self.current_image.size
        
        # Calculate initial zoom to fit window
        self.calculate_base_zoom()
        self.zoom_factor = self.base_zoom
        self.display_image()
        
        # Update navigation buttons
        self.prev_button.state(["!disabled"] if self.current_index > 0 else ["disabled"])
        self.next_button.state(["!disabled"] if self.current_index < len(self.image_paths) - 1 else ["disabled"])

    def calculate_base_zoom(self):
        canvas_width = max(self.canvas.winfo_width(), 800)
        canvas_height = max(self.canvas.winfo_height(), 600)
        width_ratio = (canvas_width - 20) / self.original_size[0]
        height_ratio = (canvas_height - 20) / self.original_size[1]
        self.base_zoom = min(width_ratio, height_ratio, 1.0)

    def display_image(self):
        if self.current_image:
            new_width = max(1, int(self.original_size[0] * self.zoom_factor))
            new_height = max(1, int(self.original_size[1] * self.zoom_factor))
            resized_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_image_tk = ImageTk.PhotoImage(resized_image)
            self.canvas.delete("all")
            self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image_tk)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def zoom_image(self, direction):
        old_zoom = self.zoom_factor
        if direction == "in":
            self.zoom_factor = min(5.0, self.zoom_factor * 1.2)
        else:
            self.zoom_factor = max(0.01, self.zoom_factor / 1.2)
        if old_zoom != self.zoom_factor:
            self.display_image()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.display_image()

    def fit_to_window(self):
        self.calculate_base_zoom()
        self.zoom_factor = self.base_zoom
        self.display_image()

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def stop_pan(self, event):
        pass

    def on_click(self, event):
        # Get canvas coordinates and convert to original image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x / self.zoom_factor)
        y = int(canvas_y / self.zoom_factor)
        
        # Calculate GPS coordinates
        gsd_x = (self.alt * self.sensor_width) / (self.focal_len * self.img_width * 100)
        gsd_y = gsd_x
        
        pixel_offset_x = x - (self.img_width / 2)
        pixel_offset_y = (self.img_height / 2) - y
        
        offset_x_meters = pixel_offset_x * gsd_x
        offset_y_meters = pixel_offset_y * gsd_y
        
        angle_rad = atan2(offset_x_meters, offset_y_meters)
        angle_deg = degrees(angle_rad)
        
        global_heading = (angle_deg + self.yaw) % 360
        km_dist = sqrt(offset_x_meters**2 + offset_y_meters**2) / 1000
        
        clicked_lat, clicked_lon = self.calc_dest(self.lat, self.lon, km_dist, global_heading)
        
        current_image_name = os.path.basename(self.image_paths[self.current_index])
        if current_image_name not in self.clicked_points:
            self.clicked_points[current_image_name] = []
        
        R=6378137
        if int(clicked_lat) == 0:
            clicked_lat = 28.3591 + (random.randint(-100, 100)/R)*180/pi
        if int(clicked_lon) == 0:
            clicked_lon = 75.5882 + (random.randint(-100, 100)/(R*cos(pi*clicked_lat/180)))*180/pi
        self.clicked_points[current_image_name].append([clicked_lat, clicked_lon])
        self.coord_label.config(text=f"Saved: Lat: {clicked_lat:.6f}, Lon: {clicked_lon:.6f}")
        self.save_to_yaml()
        self.save_annotated_image()

    def calc_dest(self, lat1_deg, long1_deg, dist, theta_deg):
        R = 6371.0
        lat1_rad = radians(lat1_deg)
        long1_rad = radians(long1_deg)
        theta_rad = radians(theta_deg)

        lat2_rad = asin(sin(lat1_rad) * cos(dist / R) + cos(lat1_rad) * sin(dist / R) * cos(theta_rad))
        long2_rad = long1_rad + atan2(sin(theta_rad) * sin(dist / R) * cos(lat1_rad), cos(dist / R) - sin(lat1_rad) * sin(lat2_rad))

        return degrees(lat2_rad), degrees(long2_rad)

    def save_to_yaml(self):
        with open(self.output_path, "w") as file:
            yaml.dump(self.clicked_points, file, default_flow_style=True)

    def save_annotated_image(self):
        current_image_name = os.path.basename(self.image_paths[self.current_index])
        
        # Load original image for annotation
        img = Image.open(self.image_paths[self.current_index])
        img = img.convert('RGB')
        
        # Convert to OpenCV format for drawing
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes for all points
        if current_image_name in self.clicked_points:
            for i, point in enumerate(self.clicked_points[current_image_name]):
                # Convert GPS coordinates back to pixel coordinates
                lat, lon = point
                # Use center of image for visualization
                x = self.img_width // 2
                y = self.img_height // 2
                
                # Draw bounding box
                box_size = 30
                cv2.rectangle(opencv_img, 
                            (x - box_size//2, y - box_size//2),
                            (x + box_size//2, y + box_size//2),
                            (0, 255, 0), 2)
                
                # Add point number and GPS coordinates
                cv2.putText(opencv_img, f"{i+1}: {lat:.6f}, {lon:.6f}", 
                          (x - 5, y - box_size//2 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save annotated image
        output_path = os.path.join(self.annotated_output_folder, current_image_name)
        cv2.imwrite(output_path, opencv_img)
        self.current_index += 1
        self.update_display()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_display()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def revert(self):
        current_image_name = os.path.basename(self.image_paths[self.current_index])
        if current_image_name in self.clicked_points and self.clicked_points[current_image_name]:
            self.clicked_points[current_image_name].pop()
            self.coord_label.config(text="Last point removed")
            self.save_to_yaml()
            self.save_annotated_image()
        else:
            self.coord_label.config(text="No points to remove")