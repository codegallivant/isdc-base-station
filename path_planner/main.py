import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import yaml
from queue import PriorityQueue
import os
import glob


def reset_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        delfiles = glob.glob(path+"/*")
        for f in delfiles:
            if os.path.isfile(f):
                os.remove(f)

def compute_gradient(depth_grid):
    """Optimized gradient computation"""
    grad_y, grad_x = np.gradient(depth_grid)
    return np.hypot(grad_x, grad_y)  # Faster than sqrt(x²+y²)

def compute_potential_field(depth_grid, goal, obstacle_weight=2.0, goal_weight=2.0, 
                          sigma_obstacle=1, sigma_goal=3, slope_weight=2.0,
                          max_slope_threshold=0.5):
    """Enhanced potential field computation prioritizing path quality"""
    depth_min, depth_max = depth_grid.min(), depth_grid.max()
    depth_normalized = (depth_grid - depth_min) / (depth_max - depth_min + 1e-6)
    
    # Stronger slope penalties
    slope_field = compute_gradient(depth_grid)
    slope_field = np.clip(slope_field, 0, max_slope_threshold)
    slope_field_normalized = slope_field / max_slope_threshold
    
    # Enhanced obstacle field with stronger penalties
    obstacle_field = gaussian_filter(1 - depth_normalized, sigma=sigma_obstacle)
    slope_penalty = gaussian_filter(slope_field_normalized, sigma=sigma_obstacle)
    
    # Increased weights for obstacles and slopes
    potential = (obstacle_weight * obstacle_field + 
                slope_weight * slope_penalty)
    
    # Reduced goal attraction for more flexibility in path selection
    goal_field = np.zeros_like(depth_grid)
    goal_field[goal] = 1
    goal_field = gaussian_filter(goal_field, sigma=sigma_goal)
    potential -= goal_weight * goal_field
    
    # Stronger penalties for steep areas
    potential[slope_field_normalized > 0.7] *= 3.0
    
    return potential

def is_valid_move(depth_grid, pos):
    """Simple bounds check"""
    rows, cols = depth_grid.shape
    return (0 <= pos[0] < rows and 0 <= pos[1] < cols)

def heuristic(node, goal):
    """Modified heuristic with reduced influence"""
    dx = node[0] - goal[0]
    dy = node[1] - goal[1]
    # Using Manhattan distance component to encourage more gradual paths
    return 0.5 * (np.hypot(dx, dy) + abs(dx) + abs(dy))

def a_star_with_potential(depth_grid, start, goal, max_iterations=50000000):
    """Modified A* implementation prioritizing path quality"""
    if not (is_valid_move(depth_grid, start) and is_valid_move(depth_grid, goal)):
        return None
    
    print(f"\nStarting A* search from {start} to {goal}")
    print(f"Depth grid shape: {depth_grid.shape}")
    print(f"Depth grid range: {depth_grid.min():.3f} to {depth_grid.max():.3f}")
    
    potential_field = compute_potential_field(depth_grid, goal)
    slope_field = compute_gradient(depth_grid)
    
    rows, cols = depth_grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    # Pre-compute movement patterns with diagonal cost adjustment
    movements = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                (1, 1), (1, -1), (-1, 1), (-1, -1)]
    movement_costs = [np.sqrt(dx*dx + dy*dy) * (1.2 if dx and dy else 1.0) 
                     for dx, dy in movements]
    
    explored = 0
    iterations = 0
    while not open_set.empty() and iterations < max_iterations:
        iterations += 1
        current_cost, current = open_set.get()
        explored += 1
        
        if iterations % 1000 == 0:
            print(f"Iteration {iterations}, Explored {explored} nodes")
        
        if current == goal:
            print(f"Goal reached after {iterations} iterations!")
            break
            
        current_row, current_col = current
        
        for (dx, dy), base_cost in zip(movements, movement_costs):
            new_row, new_col = current_row + dx, current_col + dy
            
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue
                
            neighbor = (new_row, new_col)
            
            # Enhanced cost calculations prioritizing terrain quality
            slope_val = slope_field[neighbor]
            depth_val = depth_grid[neighbor]
            potential_val = potential_field[neighbor]
            
            # Increased weights for terrain factors
            terrain_cost = (
                6.0 * slope_val +  # Increased slope penalty
                4.0 * (1.0 - depth_val) +  # Increased depth penalty
                3.0 * max(0, potential_val)  # Increased potential field penalty
            )
            
            # Combine base movement cost with terrain cost
            total_cost = base_cost * (1.0 + terrain_cost)
            
            new_cost = cost_so_far[current] + total_cost
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + 0.5 * heuristic(neighbor, goal)
                open_set.put((priority, neighbor))
                came_from[neighbor] = current
    
    if goal not in came_from:
        print(f"Failed to find path after {iterations} iterations")
        return None
        
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    
    print(f"Found path with {len(path)} points")
    return path

def process_image(rgb_path, depth_path, proc_depth_path, coords, output_path):
    """Image processing with coordinate handling"""
    print(f"\nProcessing image:")
    print(f"RGB path: {rgb_path}")
    print(f"Depth path: {depth_path}")
    print(f"Coordinates: {coords}")
    
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    proc_depth = cv2.imread(proc_depth_path)
    # print(proc_depth.shape)
    # print(depth.shape)
    
    if rgb is None or depth is None:
        print("Failed to load images")
        return
        
    print(f"RGB shape: {rgb.shape}")
    print(f"Depth shape: {depth.shape}")
    
    depth_normalized = (depth.astype(float) - np.min(depth)) / (np.max(depth) - np.min(depth))
    print(f"Depth range: {depth_normalized.min():.3f} to {depth_normalized.max():.3f}")
    
    complete_path = []
    if len(coords) > 1:
        for i in range(len(coords) - 1):
            # Swap x,y to row,col for internal processing
            start = (coords[i][1], coords[i][0])
            goal = (coords[i + 1][1], coords[i + 1][0])
            
            print(f"\nPlanning segment {i+1}/{len(coords)-1}")
            print(f"Start: {start}")
            print(f"Goal: {goal}")
            
            path_segment = a_star_with_potential(depth_normalized, start, goal)
            
            if path_segment:
                if not complete_path:
                    complete_path.extend(path_segment)
                else:
                    complete_path.extend(path_segment[1:])
                print(f"Added segment with {len(path_segment)} points")
            else:
                print("Failed to find path for this segment")
    
    if complete_path:
        print(f"\nVisualization:")
        print(f"Total path length: {len(complete_path)}")
        visualize_path(rgb, proc_depth, complete_path, output_path)
    else:
        print("\nNo path to visualize")
        cv2.imwrite(output_path, rgb)

def visualize_path(image, proc_depth, path, f_name):
    """Enhanced path visualization"""
    # output_img = (image.copy()*0.7) + (proc_depth.copy()*0.3)
    output_img = image.copy()
    if path:
        # Convert coordinates to numpy array for drawing
        path_points = np.array(path, dtype=np.int32)
        
        # Draw the path
        for i in range(len(path_points) - 1):
            pt1 = (path_points[i][1], path_points[i][0])  # Swap x,y for OpenCV
            pt2 = (path_points[i + 1][1], path_points[i + 1][0])
            cv2.line(output_img, pt1, pt2, (0, 0, 255), 2)
        
        # Draw points
        for i, point in enumerate(path_points):
            color = (0, 255, 0) if i == 0 else (255, 0, 0) if i == len(path_points)-1 else (0, 0, 255)
            cv2.circle(output_img, (point[1], point[0]), 5, color, -1)
    
    cv2.imwrite(f_name, output_img)
    print(f"Saved path visualization to {f_name}")

def main():
    # Load configuration
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    INPUT_PATH_RGB = config["input_path_rgb"]
    INPUT_PATH_DEPTH = config["input_path_depth"]
    INPUT_PATH_DEPTH_PROCESSED = os.path.join(INPUT_PATH_DEPTH, "processed")
    INPUT_PATH_CONES = config["input_path_cones"]
    OUTPUT_PATH = config["output_path"]

    reset_directory(OUTPUT_PATH)
    
    # Load cone coordinates
    with open(INPUT_PATH_CONES, 'r') as file:
        cones_output = yaml.safe_load(file)
    
    # Process each image
    for file_name, coords in cones_output.items():
        print(f"\nProcessing {file_name}")
        print(f"Found {len(coords)} coordinates")
        
        if len(coords) <= 1:
            print(f"Skipping {file_name}: insufficient coordinates")
            continue
            
        rgb_path = os.path.join(INPUT_PATH_RGB, file_name)
        depth_path = os.path.join(INPUT_PATH_DEPTH, file_name)
        proc_depth_path = os.path.join(INPUT_PATH_DEPTH, file_name)
        output_path = os.path.join(OUTPUT_PATH, file_name)
        
        process_image(rgb_path, depth_path, proc_depth_path, coords, output_path)

if __name__ == "__main__":
    main()