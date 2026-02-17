"""
Multi-Camera Real-time Traffic Inference Script
=================================================
This script runs real-time object detection on 4 IP cameras simultaneously
and sends signals to Raspberry Pi to control traffic light LEDs.

Each camera monitors one lane:
  - Camera 1 -> Lane 0
  - Camera 2 -> Lane 1
  - Camera 3 -> Lane 2
  - Camera 4 -> Lane 3

Usage:
    python scripts/run_multi_camera_inference.py
    
    With pre-configured IPs:
    python scripts/run_multi_camera_inference.py --pi-ip 10.230.39.152
"""

import cv2
import time
import sys
import threading
import numpy as np
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from src.comms.pi_client import PiClient

# Dashboard server URL
DASHBOARD_URL = "http://localhost:5001/api/update"


# Class names as per your data.yaml (traffic_v14 model)
CLASS_NAMES = {0: "ambulance", 1: "police", 2: "vehicle"}

# Colors for visualization (BGR format)
CLASS_COLORS = {
    0: (0, 0, 255),     # ambulance - Red
    1: (255, 0, 0),     # police - Blue  
    2: (0, 255, 0)      # vehicle - Green
}

# Lane colors for display
LANE_COLORS = [
    (0, 255, 255),   # Lane 0 - Yellow
    (255, 0, 255),   # Lane 1 - Magenta
    (255, 255, 0),   # Lane 2 - Cyan
    (0, 165, 255),   # Lane 3 - Orange
]


@dataclass
class CameraConfig:
    """Configuration for a single camera"""
    lane_id: int
    url: str
    name: str


@dataclass
class LaneStats:
    """Detection statistics for a lane"""
    vehicle_count: int = 0
    ambulance_count: int = 0
    police_count: int = 0
    emergency: bool = False
    fps: float = 0.0
    connected: bool = False


class CameraThread(threading.Thread):
    """Thread for capturing frames from a single camera"""
    
    def __init__(self, config: CameraConfig):
        super().__init__(daemon=True)
        self.config = config
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        
    def run(self):
        cap = None
        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    print(f"[CAM {self.config.lane_id}] Connecting to {self.config.url}...")
                    cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if cap.isOpened():
                        self.connected = True
                        print(f"[CAM {self.config.lane_id}] Connected!")
                    else:
                        self.connected = False
                        time.sleep(2)
                        continue
                
                ret, frame = cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame.copy()
                    self.connected = True
                else:
                    self.connected = False
                    cap.release()
                    cap = None
                    time.sleep(1)
                    
            except Exception as e:
                print(f"[CAM {self.config.lane_id}] Error: {e}")
                self.connected = False
                if cap:
                    cap.release()
                cap = None
                time.sleep(2)
        
        if cap:
            cap.release()
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False


def get_camera_configs() -> List[CameraConfig]:
    """Prompt user for 4 camera IP addresses"""
    print("\n" + "=" * 70)
    print("  4-CAMERA IP WEBCAM CONFIGURATION")
    print("=" * 70)
    print("\nYou need to configure 4 IP cameras, one for each traffic lane.")
    print("Each camera should be running IP Webcam app on an Android phone.")
    print("\nExample URL format: http://192.168.1.100:8080/video")
    
    configs = []
    
    for i in range(4):
        print(f"\n--- Camera {i+1} (Lane {i}) ---")
        
        while True:
            source_type = input(f"Camera {i+1} source type [1=IP Webcam, 2=USB/Local, 3=Skip]: ").strip()
            
            if source_type == '1':
                ip = input(f"  Enter IP address for Camera {i+1}: ").strip()
                port = input(f"  Enter port (default: 8080): ").strip() or "8080"
                url = f"http://{ip}:{port}/video"
                configs.append(CameraConfig(lane_id=i, url=url, name=f"Lane {i}"))
                print(f"  [OK] Camera {i+1} configured: {url}")
                break
                
            elif source_type == '2':
                cam_idx = input(f"  Enter USB camera index (default: {i}): ").strip() or str(i)
                configs.append(CameraConfig(lane_id=i, url=cam_idx, name=f"Lane {i}"))
                print(f"  [OK] Camera {i+1} configured: USB camera {cam_idx}")
                break
                
            elif source_type == '3':
                print(f"  [SKIP] Camera {i+1} will be disabled")
                break
            else:
                print("  Invalid choice. Enter 1, 2, or 3.")
    
    return configs


def get_pi_config() -> tuple:
    """Prompt user for Raspberry Pi configuration"""
    print("\n" + "=" * 70)
    print("  RASPBERRY PI SIGNAL SERVER CONFIGURATION")
    print("=" * 70)
    print("\nThe Raspberry Pi controls traffic light LEDs via GPIO pins.")
    print("Make sure signal_server.py is running on the Pi.")
    
    pi_ip = input("\nEnter Raspberry Pi IP address (e.g., 10.230.39.152): ").strip()
    pi_port = input("Enter server port (default: 5000): ").strip() or "5000"
    
    return pi_ip, int(pi_port)


def draw_detections(frame, results, lane_id):
    """Draw bounding boxes and return stats"""
    stats = LaneStats()
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        if cls_id == 0:  # ambulance
            stats.ambulance_count += 1
            stats.emergency = True
        elif cls_id == 1:  # police
            stats.police_count += 1
            stats.emergency = True
        elif cls_id == 2:  # vehicle
            stats.vehicle_count += 1
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 8), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame, stats


def draw_lane_overlay(frame, lane_id, stats, pi_connected):
    """Draw lane info overlay on frame"""
    h, w = frame.shape[:2]
    
    # Lane label
    lane_color = LANE_COLORS[lane_id]
    cv2.rectangle(frame, (0, 0), (w, 30), lane_color, -1)
    cv2.putText(frame, f"LANE {lane_id}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Stats background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 35), (180, 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Stats text
    y = 55
    cv2.putText(frame, f"FPS: {stats.fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y += 20
    cv2.putText(frame, f"Vehicles: {stats.vehicle_count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 20
    cv2.putText(frame, f"Ambulances: {stats.ambulance_count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    y += 20
    cv2.putText(frame, f"Police: {stats.police_count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    y += 20
    
    if stats.emergency:
        cv2.putText(frame, "EMERGENCY!", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame


def create_grid_display(frames: Dict[int, np.ndarray], target_size=(640, 480)) -> np.ndarray:
    """Create a 2x2 grid display of all camera feeds"""
    grid_frames = []
    
    for i in range(4):
        if i in frames and frames[i] is not None:
            frame = cv2.resize(frames[i], target_size)
        else:
            # Create placeholder for disconnected camera
            frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(frame, f"LANE {i}", (target_size[0]//2 - 60, target_size[1]//2 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, LANE_COLORS[i], 2)
            cv2.putText(frame, "NO SIGNAL", (target_size[0]//2 - 80, target_size[1]//2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        grid_frames.append(frame)
    
    # Create 2x2 grid
    top_row = np.hstack([grid_frames[0], grid_frames[1]])
    bottom_row = np.hstack([grid_frames[2], grid_frames[3]])
    grid = np.vstack([top_row, bottom_row])
    
    return grid


def main():
    print("=" * 70)
    print("  AI TRAFFIC MANAGEMENT - 4-CAMERA INFERENCE SYSTEM")
    print("=" * 70)
    
    # Check for command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-ip", type=str, default=None)
    parser.add_argument("--pi-port", type=int, default=5000)
    parser.add_argument("--cam1", type=str, default=None, help="Camera 1 URL")
    parser.add_argument("--cam2", type=str, default=None, help="Camera 2 URL")
    parser.add_argument("--cam3", type=str, default=None, help="Camera 3 URL")
    parser.add_argument("--cam4", type=str, default=None, help="Camera 4 URL")
    parser.add_argument("--model", type=str, default="runs/detect/traffic_v14/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    
    # Get camera configurations
    if args.cam1 or args.cam2 or args.cam3 or args.cam4:
        # Use command line args
        configs = []
        cams = [args.cam1, args.cam2, args.cam3, args.cam4]
        for i, cam in enumerate(cams):
            if cam:
                configs.append(CameraConfig(lane_id=i, url=cam, name=f"Lane {i}"))
    else:
        # Interactive mode
        configs = get_camera_configs()
    
    if not configs:
        print("[ERROR] No cameras configured. Exiting.")
        return
    
    # Get Pi configuration
    if args.pi_ip:
        pi_ip = args.pi_ip
        pi_port = args.pi_port
    else:
        pi_ip, pi_port = get_pi_config()
    
    # Initialize Raspberry Pi client
    print(f"\n[INFO] Connecting to Raspberry Pi at {pi_ip}:{pi_port}...")
    pi_client = PiClient(pi_ip, port=pi_port)
    
    if pi_client.check_connection():
        print("[OK] Raspberry Pi connected successfully!")
        print("[OK] GPIO signal control ENABLED")
    else:
        print("[WARN] Raspberry Pi not reachable!")
        print("[WARN] Will retry connection during operation...")
    
    # Load YOLO model
    model_path = Path(project_root) / args.model
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    print(f"\n[INFO] Loading YOLO model: {model_path}")
    model = YOLO(str(model_path))
    print("[OK] Model loaded successfully")
    
    # Start camera threads
    print("\n[INFO] Starting camera threads...")
    camera_threads = {}
    for config in configs:
        thread = CameraThread(config)
        thread.start()
        camera_threads[config.lane_id] = thread
        print(f"[OK] Camera thread started for Lane {config.lane_id}")
    
    time.sleep(2)  # Allow cameras to connect
    
    print("\n" + "=" * 70)
    print("  CONTROLS:")
    print("    'q' - Quit")
    print("    's' - Save screenshot")
    print("    '0-3' - Manually send signal to lane 0-3")
    print("    'r' - Send all-red signal")
    print("=" * 70 + "\n")
    
    # Main processing loop
    lane_stats = {i: LaneStats() for i in range(4)}
    fps_counters = {i: {"count": 0, "start": time.time()} for i in range(4)}
    last_dashboard_update = 0
    last_signal_update = 0
    
    # Traffic control settings
    SIGNAL_UPDATE_INTERVAL = 1.0  # Check and update signals every second
    DASHBOARD_UPDATE_INTERVAL = 0.5  # Update dashboard every 500ms
    MIN_GREEN_TIME = 5  # Minimum green time before switching (seconds)
    NORMAL_GREEN_TIME = 15  # Default green time for normal traffic
    
    # Current traffic state
    current_green_lane = -1  # Which lane is currently green (-1 = all red)
    green_start_time = 0  # When current green started
    emergency_active = False  # Is any emergency vehicle present
    emergency_lane = -1  # Which lane has emergency vehicle
    last_signal_sent = None  # Track last signal to avoid redundant sends
    
    def get_priority_lane(lane_stats):
        """
        Determine which lane should get green light based on:
        1. Emergency vehicles (highest priority)
        2. Vehicle count (if no emergency)
        
        Returns: (lane_id, is_emergency)
        """
        # Check for emergency vehicles first
        emergency_lanes = []
        for lane_id, stats in lane_stats.items():
            if stats.emergency:
                emergency_lanes.append((lane_id, stats.ambulance_count + stats.police_count))
        
        if emergency_lanes:
            # Return lane with most emergency vehicles (or first if tie)
            emergency_lanes.sort(key=lambda x: x[1], reverse=True)
            return emergency_lanes[0][0], True
        
        # No emergency - find lane with most vehicles
        vehicle_counts = [(lane_id, stats.vehicle_count) for lane_id, stats in lane_stats.items() if stats.connected]
        
        if not vehicle_counts:
            return -1, False  # No connected cameras
        
        # Sort by vehicle count (descending), then by lane_id (for consistency)
        vehicle_counts.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        
        # If highest count is 0, still give green to first connected lane
        return vehicle_counts[0][0], False
    
    def send_traffic_signal(pi_client, green_lane, is_emergency):
        """Send signal to Raspberry Pi to update traffic lights"""
        nonlocal last_signal_sent
        
        # Create signal state signature
        signal_state = (green_lane, is_emergency)
        
        # Avoid sending duplicate signals
        if signal_state == last_signal_sent:
            return True
        
        if green_lane < 0:
            # All red
            success = pi_client.send_all_red()
        else:
            success = pi_client.send(lane=green_lane, duration=30, emergency=is_emergency)
        
        if success:
            last_signal_sent = signal_state
        
        return success
    
    def send_dashboard_update(lane_stats, green_lane, emergency_active, pi_connected, green_start_time):
        """Send status update to web dashboard"""
        try:
            remaining = max(0, NORMAL_GREEN_TIME - int(time.time() - green_start_time)) if green_lane >= 0 else 0
            
            data = {
                "current_lane": green_lane,
                "remaining_time": remaining,
                "emergency_active": emergency_active,
                "pi_connected": pi_connected,
                "lanes": {
                    str(i): {
                        "vehicles": s.vehicle_count,
                        "ambulance": s.ambulance_count,
                        "police": s.police_count,
                        "signal": "green" if i == green_lane else "red"
                    }
                    for i, s in lane_stats.items()
                }
            }
            requests.post(DASHBOARD_URL, json=data, timeout=0.5)
        except:
            pass  # Dashboard might not be running
    
    print("\n[INFO] Starting dynamic traffic control...")
    
    try:
        while True:
            processed_frames = {}
            current_time = time.time()
            
            # Process each camera and update lane stats
            for lane_id, thread in camera_threads.items():
                frame = thread.get_frame()
                
                if frame is not None:
                    # Run inference
                    results = model(frame, conf=args.conf, imgsz=640, verbose=False)[0]
                    
                    # Draw detections
                    frame, stats = draw_detections(frame, results, lane_id)
                    
                    # Calculate FPS
                    fps_counters[lane_id]["count"] += 1
                    elapsed = current_time - fps_counters[lane_id]["start"]
                    if elapsed > 1.0:
                        stats.fps = fps_counters[lane_id]["count"] / elapsed
                        fps_counters[lane_id]["count"] = 0
                        fps_counters[lane_id]["start"] = current_time
                    else:
                        stats.fps = lane_stats[lane_id].fps
                    
                    stats.connected = True
                    lane_stats[lane_id] = stats
                    
                    # Draw overlay (mark green lane)
                    frame = draw_lane_overlay(frame, lane_id, stats, pi_client.is_healthy())
                    
                    # Mark current green lane on frame
                    if lane_id == current_green_lane:
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], 5), (0, 255, 0), -1)
                    
                    processed_frames[lane_id] = frame
                else:
                    lane_stats[lane_id].connected = False
            
            # Dynamic traffic signal control
            if current_time - last_signal_update > SIGNAL_UPDATE_INTERVAL:
                # Determine which lane should be green
                priority_lane, is_emergency = get_priority_lane(lane_stats)
                
                # Check if we need to switch signals
                should_switch = False
                time_since_green = current_time - green_start_time
                
                if is_emergency:
                    # Emergency always takes priority (immediate switch)
                    if priority_lane != emergency_lane:
                        should_switch = True
                        print(f"\n[EMERGENCY] Vehicle detected on Lane {priority_lane}!")
                elif emergency_active and not is_emergency:
                    # Emergency ended - switch back to normal
                    should_switch = True
                    print(f"\n[NORMAL] Emergency cleared, returning to normal operation")
                elif priority_lane != current_green_lane:
                    # Different lane has priority - check min green time
                    if time_since_green >= MIN_GREEN_TIME or current_green_lane < 0:
                        should_switch = True
                elif time_since_green >= NORMAL_GREEN_TIME:
                    # Time expired - find next lane with vehicles
                    should_switch = True
                
                if should_switch and priority_lane >= 0:
                    signal_type = "EMERGENCY" if is_emergency else "NORMAL"
                    print(f"[SIGNAL] Lane {priority_lane} -> GREEN ({signal_type})")
                    
                    if send_traffic_signal(pi_client, priority_lane, is_emergency):
                        current_green_lane = priority_lane
                        green_start_time = current_time
                        emergency_active = is_emergency
                        emergency_lane = priority_lane if is_emergency else -1
                    else:
                        print(f"[WARN] Failed to send signal to Pi")
                
                last_signal_update = current_time
            
            # Send update to web dashboard
            if current_time - last_dashboard_update > DASHBOARD_UPDATE_INTERVAL:
                send_dashboard_update(lane_stats, current_green_lane, emergency_active, pi_client.is_healthy(), green_start_time)
                last_dashboard_update = current_time
            
            # Create grid display
            grid = create_grid_display(processed_frames)
            
            # Draw status bar on grid
            status_y = 25
            # Pi status
            pi_status = "Pi: CONNECTED" if pi_client.is_healthy() else "Pi: DISCONNECTED"
            pi_color = (0, 255, 0) if pi_client.is_healthy() else (0, 0, 255)
            cv2.putText(grid, pi_status, (grid.shape[1] - 200, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, pi_color, 2)
            
            # Current mode
            if emergency_active:
                mode_text = f"EMERGENCY - Lane {current_green_lane}"
                cv2.putText(grid, mode_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif current_green_lane >= 0:
                remaining = max(0, int(NORMAL_GREEN_TIME - (current_time - green_start_time)))
                mode_text = f"Lane {current_green_lane} GREEN ({remaining}s)"
                cv2.putText(grid, mode_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("AI Traffic Management - Dynamic Control (Press 'q' to quit)", grid)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                screenshot_path = Path(project_root) / f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(str(screenshot_path), grid)
                print(f"[INFO] Screenshot saved: {screenshot_path}")
            elif key == ord('r'):
                print("[SIGNAL] Sending ALL-RED signal...")
                pi_client.send_all_red()
                current_green_lane = -1
                emergency_active = False
                last_signal_sent = None
            elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                lane = key - ord('0')
                print(f"[SIGNAL] Manual GREEN signal to Lane {lane}")
                if pi_client.send(lane=lane, duration=15, emergency=False):
                    current_green_lane = lane
                    green_start_time = time.time()
                    emergency_active = False
                    last_signal_sent = (lane, False)
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        print("[INFO] Stopping camera threads...")
        for thread in camera_threads.values():
            thread.stop()
        
        print("[INFO] Sending all-red safety signal...")
        pi_client.send_all_red()
        
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
