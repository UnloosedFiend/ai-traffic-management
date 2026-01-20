"""
Multi-Camera Real-time Inference Script (4 IP Webcams)
=======================================================
This script runs real-time object detection on 4 IP webcam streams
for a 4-lane traffic junction.

Usage:
    python scripts/run_multi_camera.py

Configure camera URLs in the CAMERA_SOURCES list below.
"""

import argparse
import cv2
import time
import sys
import threading
import queue
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO


# ============================================================
# CONFIGURATION - Edit these for your setup
# ============================================================

CAMERA_SOURCES = [
    "http://192.168.1.3:8080/video",   # Lane 0 - Camera 1
    "http://192.168.1.4:8080/video",   # Lane 1 - Camera 2
    "http://192.168.1.5:8080/video",   # Lane 2 - Camera 3
    "http://192.168.1.6:8080/video",   # Lane 3 - Camera 4
]

# Model path (your trained model)
MODEL_PATH = "runs/detect/traffic_v14/weights/best.pt"

# Detection settings
CONFIDENCE_THRESHOLD = 0.4
IMAGE_SIZE = 640

# FPS limit per camera (to prevent CPU/GPU overload)
TARGET_FPS = 15

# Display settings
GRID_CELL_SIZE = (640, 480)  # Each camera view size
SHOW_INDIVIDUAL_WINDOWS = False  # Set True to show each camera separately

# ============================================================

# Class configuration
CLASS_NAMES = {0: "ambulance", 1: "police", 2: "vehicle"}
CLASS_COLORS = {
    0: (0, 0, 255),     # ambulance - Red
    1: (255, 0, 0),     # police - Blue
    2: (0, 255, 0)      # vehicle - Green
}


class CameraThread:
    """Threaded camera capture to prevent blocking"""
    
    def __init__(self, source, camera_id, frame_queue, max_queue_size=2):
        self.source = source
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.running = False
        self.thread = None
        self.cap = None
        self.connected = False
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            
    def _capture_loop(self):
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.connected = self.cap.isOpened()
        
        if not self.connected:
            print(f"[WARN] Camera {self.camera_id}: Failed to connect to {self.source}")
            return
            
        print(f"[OK] Camera {self.camera_id}: Connected")
        
        frame_interval = 1.0 / TARGET_FPS
        last_frame_time = 0
        
        while self.running:
            current_time = time.time()
            
            # FPS limiting
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
                
            ret, frame = self.cap.read()
            
            if not ret:
                # Try to reconnect
                time.sleep(1)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                continue
            
            # Clear old frames and add new one
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
            except:
                pass
                
            self.frame_queue.put((self.camera_id, frame))
            last_frame_time = current_time


def draw_detections(frame, results, lane_id):
    """Draw detections and return stats"""
    stats = {"vehicle": 0, "ambulance": 0, "police": 0, "emergency": False, "total": 0}
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        stats["total"] += 1
        if cls_id == 0:
            stats["ambulance"] += 1
            stats["emergency"] = True
        elif cls_id == 1:
            stats["police"] += 1
            stats["emergency"] = True
        elif cls_id == 2:
            stats["vehicle"] += 1
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{cls_name}: {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame, stats


def draw_lane_overlay(frame, lane_id, stats, fps):
    """Draw lane info overlay"""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (200, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Lane title
    cv2.putText(frame, f"LANE {lane_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Stats
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Vehicles: {stats['vehicle']}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Ambulance: {stats['ambulance']}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"Police: {stats['police']}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Emergency indicator
    if stats["emergency"]:
        cv2.rectangle(frame, (w-120, 5), (w-5, 35), (0, 0, 255), -1)
        cv2.putText(frame, "EMERGENCY", (w-115, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


def create_grid(frames, grid_size=(2, 2), cell_size=GRID_CELL_SIZE):
    """Create 2x2 grid view of all cameras"""
    rows, cols = grid_size
    cell_w, cell_h = cell_size
    
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for lane_id in range(4):
        row, col = lane_id // cols, lane_id % cols
        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w
        
        if lane_id in frames and frames[lane_id] is not None:
            resized = cv2.resize(frames[lane_id], cell_size)
            grid[y1:y2, x1:x2] = resized
        else:
            # No signal placeholder
            cv2.putText(grid, f"Lane {lane_id}: Waiting...", (x1 + 50, y1 + cell_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    return grid


def main():
    print("=" * 60)
    print("  AI Traffic Management - Multi-Camera Inference")
    print("=" * 60)
    
    # Load model
    model_path = Path(project_root) / MODEL_PATH
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    print("[OK] Model loaded")
    
    # Import numpy here (after confirming script runs)
    global np
    import numpy as np
    
    # Create frame queues for each camera
    frame_queues = [queue.Queue(maxsize=2) for _ in range(4)]
    
    # Start camera threads
    camera_threads = []
    print(f"\n[INFO] Starting {len(CAMERA_SOURCES)} camera streams...")
    print(f"[INFO] Target FPS per camera: {TARGET_FPS}")
    
    for i, source in enumerate(CAMERA_SOURCES):
        print(f"[INFO] Camera {i}: {source}")
        cam_thread = CameraThread(source, i, frame_queues[i])
        cam_thread.start()
        camera_threads.append(cam_thread)
    
    # Wait for cameras to connect
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("  Press 'q' to quit | Press 's' for screenshot")
    print("=" * 60 + "\n")
    
    # Lane state for traffic logic
    lane_state = {i: {"count": 0, "emergency": False} for i in range(4)}
    
    # FPS tracking per lane
    fps_counters = {i: {"frames": 0, "last_time": time.time(), "fps": 0} for i in range(4)}
    
    # Latest processed frames
    processed_frames = {}
    
    try:
        while True:
            # Process frames from each camera
            for lane_id in range(4):
                try:
                    cam_id, frame = frame_queues[lane_id].get_nowait()
                    
                    # Run detection
                    results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE, verbose=False)[0]
                    
                    # Draw detections
                    frame, stats = draw_detections(frame, results, lane_id)
                    
                    # Update lane state
                    lane_state[lane_id]["count"] = stats["total"]
                    lane_state[lane_id]["emergency"] = stats["emergency"]
                    
                    # Calculate FPS
                    fps_counters[lane_id]["frames"] += 1
                    elapsed = time.time() - fps_counters[lane_id]["last_time"]
                    if elapsed > 1.0:
                        fps_counters[lane_id]["fps"] = fps_counters[lane_id]["frames"] / elapsed
                        fps_counters[lane_id]["frames"] = 0
                        fps_counters[lane_id]["last_time"] = time.time()
                    
                    # Draw overlay
                    frame = draw_lane_overlay(frame, lane_id, stats, fps_counters[lane_id]["fps"])
                    
                    processed_frames[lane_id] = frame
                    
                except queue.Empty:
                    pass
            
            # Create and show grid view
            if processed_frames:
                grid = create_grid(processed_frames)
                
                # Add traffic decision info
                cv2.putText(grid, "AI Traffic Control - 4 Lane View", (10, grid.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("AI Traffic Management - 4 Cameras (Press 'q' to quit)", grid)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                screenshot_path = project_root / f"multi_cam_screenshot_{int(time.time())}.jpg"
                if processed_frames:
                    cv2.imwrite(str(screenshot_path), create_grid(processed_frames))
                    print(f"[INFO] Screenshot saved: {screenshot_path}")
            
            # Print lane state periodically
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    
    finally:
        # Cleanup
        for cam in camera_threads:
            cam.stop()
        cv2.destroyAllWindows()
        
        print("\n[SUMMARY] Final Lane States:")
        for lane_id, state in lane_state.items():
            print(f"  Lane {lane_id}: {state['count']} vehicles, Emergency: {state['emergency']}")
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
