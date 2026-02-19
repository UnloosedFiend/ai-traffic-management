"""
AI Traffic Management - Web Launcher
======================================
A web-based interface for configuring and running the traffic management system.

Features:
- Web UI for entering IP webcam addresses
- Real-time video streaming in browser
- Live inference visualization
- Traffic signal control dashboard

Usage:
    python scripts/web_launcher.py
    
Then open: http://localhost:5000
"""

import cv2
import time
import sys
import threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from dataclasses import dataclass, field
from typing import Dict, Optional
from queue import Queue
import urllib.request

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

app = Flask(__name__, template_folder=str(project_root / "src" / "web" / "templates"))
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = {0: "ambulance", 1: "police", 2: "vehicle"}
CLASS_COLORS = {
    0: (0, 0, 255),     # ambulance - Red
    1: (255, 0, 0),     # police - Blue  
    2: (0, 255, 0)      # vehicle - Green
}
LANE_NAMES = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}

# ============================================================================
# GLOBAL STATE
# ============================================================================

@dataclass
class LaneStats:
    vehicle_count: int = 0
    ambulance_count: int = 0
    police_count: int = 0
    emergency: bool = False
    fps: float = 0.0
    connected: bool = False

@dataclass 
class SystemState:
    running: bool = False
    model_loaded: bool = False
    pi_ip: str = ""
    pi_port: int = 5000
    pi_connected: bool = False
    cameras: Dict[int, str] = field(default_factory=dict)
    lane_stats: Dict[int, LaneStats] = field(default_factory=lambda: {i: LaneStats() for i in range(4)})
    current_green_lane: int = -1
    emergency_active: bool = False
    green_start_time: float = 0

state = SystemState()
model = None
camera_threads: Dict[int, 'CameraProcessor'] = {}
pi_client = None

# ============================================================================
# CAMERA PROCESSOR - Efficient threaded capture + inference
# ============================================================================

class CameraProcessor(threading.Thread):
    """Efficient camera processor with frame capture and inference"""
    
    def __init__(self, lane_id: int, url: str):
        super().__init__(daemon=True)
        self.lane_id = lane_id
        self.url = url
        self.running = True
        self.connected = False
        
        # Frame buffers
        self.raw_frame = None
        self.processed_frame = None
        self.lock = threading.Lock()
        
        # Stats
        self.stats = LaneStats()
        self.fps_counter = 0
        self.fps_start = time.time()
        
        # JPEG output for streaming
        self.jpeg_frame = None
        self.jpeg_lock = threading.Lock()
        
        # Determine capture mode
        self.use_jpeg_mode = "http" in url.lower()
        if self.use_jpeg_mode:
            base = url.replace("/video", "").replace("/shot.jpg", "").rstrip("/")
            self.jpeg_url = f"{base}/shot.jpg"
        
    def run(self):
        global model
        
        while self.running:
            try:
                # Capture frame
                frame = self._capture_frame()
                
                if frame is None:
                    self.connected = False
                    time.sleep(0.1)
                    continue
                
                self.connected = True
                
                # Run inference if model is loaded
                if model is not None:
                    frame, stats = self._process_frame(frame)
                    self.stats = stats
                else:
                    stats = LaneStats()
                
                # Calculate FPS
                self.fps_counter += 1
                elapsed = time.time() - self.fps_start
                if elapsed > 1.0:
                    self.stats.fps = self.fps_counter / elapsed
                    self.fps_counter = 0
                    self.fps_start = time.time()
                
                self.stats.connected = True
                
                # Draw overlay
                frame = self._draw_overlay(frame)
                
                # Encode to JPEG for web streaming
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                with self.jpeg_lock:
                    self.jpeg_frame = jpeg.tobytes()
                
                with self.lock:
                    self.processed_frame = frame
                    
            except Exception as e:
                print(f"[CAM {self.lane_id}] Error: {e}")
                self.connected = False
                time.sleep(0.2)
    
    def _capture_frame(self):
        """Capture single frame with minimal latency"""
        if self.use_jpeg_mode:
            return self._capture_jpeg()
        else:
            return self._capture_stream()
    
    def _capture_jpeg(self):
        """Fetch JPEG snapshot - zero buffering"""
        try:
            req = urllib.request.Request(self.jpeg_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except:
            return None
    
    def _capture_stream(self):
        """Capture from video stream (USB cameras)"""
        if not hasattr(self, '_cap') or self._cap is None:
            self._cap = cv2.VideoCapture(self.url)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self._cap.isOpened():
            # Grab and discard to get latest frame
            self._cap.grab()
            ret, frame = self._cap.retrieve()
            if ret:
                return frame
        return None
    
    def _process_frame(self, frame):
        """Run YOLO inference on frame"""
        global model
        
        stats = LaneStats()
        
        # Run inference with optimized settings
        results = model(frame, conf=0.4, imgsz=480, verbose=False)[0]
        
        # Draw detections
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
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, stats
    
    def _draw_overlay(self, frame):
        """Draw lane info overlay"""
        h, w = frame.shape[:2]
        
        # Lane label bar
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]
        cv2.rectangle(frame, (0, 0), (w, 28), colors[self.lane_id], -1)
        cv2.putText(frame, f"LANE {self.lane_id} - {LANE_NAMES[self.lane_id]}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Stats box
        cv2.rectangle(frame, (5, 32), (160, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 32), (160, 120), (100, 100, 100), 1)
        
        y = 50
        cv2.putText(frame, f"FPS: {self.stats.fps:.1f}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 18
        cv2.putText(frame, f"Vehicles: {self.stats.vehicle_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        y += 18
        cv2.putText(frame, f"Ambulance: {self.stats.ambulance_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        y += 18
        cv2.putText(frame, f"Police: {self.stats.police_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        
        if self.stats.emergency:
            cv2.putText(frame, "EMERGENCY!", (w - 120, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Green lane indicator
        if state.current_green_lane == self.lane_id:
            cv2.rectangle(frame, (0, h-8), (w, h), (0, 255, 0), -1)
        
        return frame
    
    def get_jpeg(self):
        """Get latest JPEG frame for streaming"""
        with self.jpeg_lock:
            return self.jpeg_frame
    
    def stop(self):
        self.running = False
        if hasattr(self, '_cap') and self._cap:
            self._cap.release()


# ============================================================================
# TRAFFIC CONTROL THREAD
# ============================================================================

class TrafficController(threading.Thread):
    """Background thread for traffic signal control logic"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        
        # Timing settings
        self.min_green_time = 5
        self.normal_green_time = 15
        self.signal_update_interval = 1.0
        
        # Round-robin tracking for equal vehicle counts
        self.last_round_robin_lane = -1
        
    def run(self):
        global state, pi_client, camera_threads
        
        last_signal_update = 0
        
        while self.running:
            if not state.running:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # Update lane stats from camera threads
            for lane_id, cam in camera_threads.items():
                state.lane_stats[lane_id] = cam.stats
            
            # Signal control logic
            if current_time - last_signal_update > self.signal_update_interval:
                self._update_signals(current_time)
                last_signal_update = current_time
            
            time.sleep(0.1)
    
    def _update_signals(self, current_time):
        global state, pi_client
        
        # Find priority lane
        priority_lane, is_emergency = self._get_priority_lane()
        
        time_since_green = current_time - state.green_start_time
        
        # Handle no vehicles case - all red
        if priority_lane < 0:
            if state.current_green_lane >= 0:
                # Switch to all red
                state.current_green_lane = -1
                state.emergency_active = False
                if pi_client:
                    try:
                        pi_client.send_all_red()
                    except:
                        pass
            return
        
        # Check if we need to switch
        should_switch = False
        
        if is_emergency and priority_lane != state.current_green_lane:
            should_switch = True
        elif state.emergency_active and not is_emergency:
            should_switch = True
        elif priority_lane != state.current_green_lane:
            if time_since_green >= self.min_green_time or state.current_green_lane < 0:
                should_switch = True
        elif time_since_green >= self.normal_green_time:
            should_switch = True
        
        if should_switch:
            state.current_green_lane = priority_lane
            state.green_start_time = current_time
            state.emergency_active = is_emergency
            
            # Update round-robin tracker
            if not is_emergency:
                self.last_round_robin_lane = priority_lane
            
            # Send to Pi
            if pi_client:
                try:
                    pi_client.send(lane=priority_lane, duration=30, emergency=is_emergency)
                except:
                    pass
    
    def _get_priority_lane(self):
        """
        Determine which lane should be green based on priority:
        1. Ambulance (highest priority emergency)
        2. Police (lower priority emergency)
        3. Most vehicles (normal traffic)
        4. Round-robin for equal vehicle counts
        5. All red if no vehicles detected
        """
        connected_lanes = [
            (lane_id, stats) for lane_id, stats in state.lane_stats.items() 
            if stats.connected
        ]
        
        if not connected_lanes:
            return -1, False  # No connected cameras
        
        # Priority 1: Check for ambulances first (highest emergency priority)
        ambulance_lanes = [
            (lane_id, stats.ambulance_count) 
            for lane_id, stats in connected_lanes 
            if stats.ambulance_count > 0
        ]
        if ambulance_lanes:
            # Return lane with most ambulances
            ambulance_lanes.sort(key=lambda x: x[1], reverse=True)
            return ambulance_lanes[0][0], True
        
        # Priority 2: Check for police (lower emergency priority)
        police_lanes = [
            (lane_id, stats.police_count) 
            for lane_id, stats in connected_lanes 
            if stats.police_count > 0
        ]
        if police_lanes:
            # Return lane with most police vehicles
            police_lanes.sort(key=lambda x: x[1], reverse=True)
            return police_lanes[0][0], True
        
        # Priority 3: Normal traffic - count vehicles per lane
        vehicle_counts = [
            (lane_id, stats.vehicle_count) 
            for lane_id, stats in connected_lanes
        ]
        
        # Check if ALL lanes have zero vehicles
        total_vehicles = sum(count for _, count in vehicle_counts)
        if total_vehicles == 0:
            return -1, False  # All red - no vehicles
        
        # Find maximum vehicle count
        max_count = max(count for _, count in vehicle_counts)
        
        # Get all lanes with maximum count (for round-robin)
        max_lanes = [lane_id for lane_id, count in vehicle_counts if count == max_count]
        
        if len(max_lanes) == 1:
            # Single lane with most vehicles
            return max_lanes[0], False
        else:
            # Multiple lanes with equal count - use round-robin
            # Find next lane in sequence after last_round_robin_lane
            max_lanes.sort()  # Ensure consistent ordering
            
            # Find next lane in round-robin sequence
            next_lane = max_lanes[0]  # Default to first
            for lane_id in max_lanes:
                if lane_id > self.last_round_robin_lane:
                    next_lane = lane_id
                    break
            else:
                # Wrap around to first lane
                next_lane = max_lanes[0]
            
            return next_lane, False
    
    def stop(self):
        self.running = False


traffic_controller = None

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main launcher page"""
    return render_template('launcher.html')


@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the traffic management system"""
    global state, model, camera_threads, pi_client, traffic_controller
    
    data = request.json
    
    # Get configuration
    cameras = data.get('cameras', {})
    pi_ip = data.get('pi_ip', '')
    pi_port = data.get('pi_port', 5000)
    
    if not any(cameras.values()):
        return jsonify({'status': 'error', 'message': 'No cameras configured'}), 400
    
    # Store config
    state.cameras = {int(k): v for k, v in cameras.items() if v}
    state.pi_ip = pi_ip
    state.pi_port = pi_port
    
    # Load model if not loaded
    if not state.model_loaded:
        try:
            from ultralytics import YOLO
            model_path = project_root / "runs" / "detect" / "traffic_v14" / "weights" / "best.pt"
            if not model_path.exists():
                model_path = project_root / "yolov8n.pt"
            print(f"[INFO] Loading model: {model_path}")
            model = YOLO(str(model_path))
            state.model_loaded = True
            print("[INFO] Model loaded!")
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Failed to load model: {e}'}), 500
    
    # Initialize Pi client
    if pi_ip:
        try:
            from src.comms.pi_client import PiClient
            pi_client = PiClient(pi_ip, port=pi_port)
            state.pi_connected = pi_client.check_connection()
        except:
            state.pi_connected = False
    
    # Stop existing threads
    for cam in camera_threads.values():
        cam.stop()
    camera_threads.clear()
    
    # Start camera processors
    for lane_id, url in state.cameras.items():
        cam = CameraProcessor(lane_id, url)
        cam.start()
        camera_threads[lane_id] = cam
        print(f"[INFO] Started camera processor for Lane {lane_id}")
    
    # Start traffic controller
    if traffic_controller:
        traffic_controller.stop()
    traffic_controller = TrafficController()
    traffic_controller.start()
    
    state.running = True
    state.current_green_lane = -1
    state.green_start_time = time.time()
    
    return jsonify({
        'status': 'ok',
        'message': 'System started',
        'cameras': len(state.cameras),
        'pi_connected': state.pi_connected
    })


@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the traffic management system"""
    global state, camera_threads, traffic_controller
    
    state.running = False
    
    for cam in camera_threads.values():
        cam.stop()
    camera_threads.clear()
    
    if traffic_controller:
        traffic_controller.stop()
    
    return jsonify({'status': 'ok', 'message': 'System stopped'})


@app.route('/api/status')
def get_status():
    """Get current system status"""
    global state
    
    remaining = 0
    if state.current_green_lane >= 0:
        remaining = max(0, 15 - int(time.time() - state.green_start_time))
    
    return jsonify({
        'running': state.running,
        'pi_connected': state.pi_connected,
        'current_green_lane': state.current_green_lane,
        'emergency_active': state.emergency_active,
        'remaining_time': remaining,
        'lanes': {
            str(i): {
                'connected': s.connected,
                'vehicles': s.vehicle_count,
                'ambulance': s.ambulance_count,
                'police': s.police_count,
                'emergency': s.emergency,
                'fps': round(s.fps, 1)
            }
            for i, s in state.lane_stats.items()
        }
    })


@app.route('/video/<int:lane_id>')
def video_feed(lane_id):
    """MJPEG video stream for a specific lane"""
    def generate():
        while True:
            if lane_id in camera_threads:
                jpeg = camera_threads[lane_id].get_jpeg()
                if jpeg:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.033)  # ~30 FPS max
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/signal/<int:lane_id>', methods=['POST'])
def manual_signal(lane_id):
    """Manually set a lane to green"""
    global state, pi_client
    
    state.current_green_lane = lane_id
    state.green_start_time = time.time()
    state.emergency_active = False
    
    if pi_client:
        try:
            pi_client.send(lane=lane_id, duration=30, emergency=False)
        except:
            pass
    
    return jsonify({'status': 'ok', 'lane': lane_id})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  AI TRAFFIC MANAGEMENT - WEB LAUNCHER")
    print("=" * 60)
    print()
    print("  Open in browser: http://localhost:5000")
    print()
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
