"""
Visualization utilities for AI Traffic Management
"""

import cv2
import numpy as np


# Class configuration (matches data.yaml: 0=ambulance, 1=police, 2=vehicle)
CLASS_NAMES = {0: "ambulance", 1: "police", 2: "vehicle"}
CLASS_COLORS = {
    0: (0, 0, 255),     # ambulance - Red (BGR)
    1: (255, 0, 0),     # police - Blue
    2: (0, 255, 0)      # vehicle - Green
}
EMERGENCY_CLASSES = {0, 1}  # ambulance, police


def draw_detection_box(frame, box, class_names=CLASS_NAMES, class_colors=CLASS_COLORS):
    """
    Draw a single detection box on frame.
    
    Args:
        frame: OpenCV image (BGR)
        box: YOLO detection box object
        class_names: dict mapping class_id to name
        class_colors: dict mapping class_id to BGR color tuple
    
    Returns:
        frame: Annotated frame
        cls_id: Class ID of the detection
        conf: Confidence score
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    
    cls_name = class_names.get(cls_id, f"class_{cls_id}")
    color = class_colors.get(cls_id, (255, 255, 255))
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label = f"{cls_name}: {conf:.2f}"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, cls_id, conf


def draw_all_detections(frame, results):
    """
    Draw all detections from YOLO results.
    
    Args:
        frame: OpenCV image
        results: YOLO results object (single frame)
    
    Returns:
        frame: Annotated frame
        stats: Dict with counts and emergency flag
    """
    stats = {"vehicle": 0, "ambulance": 0, "police": 0, "emergency": False}
    
    for box in results.boxes:
        frame, cls_id, _ = draw_detection_box(frame, box)
        
        if cls_id == 0:
            stats["ambulance"] += 1
            stats["emergency"] = True
        elif cls_id == 1:
            stats["police"] += 1
            stats["emergency"] = True
        elif cls_id == 2:
            stats["vehicle"] += 1
    
    return frame, stats


def draw_stats_overlay(frame, stats, fps=0, lane_id=None):
    """
    Draw statistics overlay on frame.
    
    Args:
        frame: OpenCV image
        stats: Dict with vehicle counts and emergency flag
        fps: Current FPS value
        lane_id: Optional lane identifier
    
    Returns:
        frame: Frame with overlay
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    y = 35
    if lane_id is not None:
        cv2.putText(frame, f"Lane: {lane_id}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y += 30
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30
    
    cv2.putText(frame, f"Vehicles: {stats.get('vehicle', 0)}", (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30
    
    cv2.putText(frame, f"Ambulances: {stats.get('ambulance', 0)}", (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y += 30
    
    cv2.putText(frame, f"Police: {stats.get('police', 0)}", (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    y += 30
    
    emergency_text = "🚨 EMERGENCY!" if stats.get('emergency', False) else "Normal"
    emergency_color = (0, 0, 255) if stats.get('emergency', False) else (0, 255, 0)
    cv2.putText(frame, emergency_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emergency_color, 2)
    
    return frame


def create_grid_view(frames, grid_size=(2, 2), cell_size=(640, 480)):
    """
    Create a grid view of multiple camera frames.
    
    Args:
        frames: List of frames (can contain None)
        grid_size: Tuple (rows, cols)
        cell_size: Tuple (width, height) for each cell
    
    Returns:
        grid: Combined grid image
    """
    rows, cols = grid_size
    cell_w, cell_h = cell_size
    
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for idx, frame in enumerate(frames):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w
        
        if frame is not None:
            resized = cv2.resize(frame, cell_size)
            grid[y1:y2, x1:x2] = resized
        else:
            # Draw "No Signal" placeholder
            cv2.putText(grid, f"Lane {idx}: No Signal", (x1 + 50, y1 + cell_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    return grid
