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
import webbrowser
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

# Import the proper TrafficController from traffic_logic
from src.logic.traffic_logic import (
    TrafficController as LogicController, 
    TrafficMode
)

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
    mode: str = "AUTO"  # Current mode: AUTO, EMERGENCY, MANUAL, FAILURE
    blue_light_blinking: bool = False

state = SystemState()
model = None
camera_threads: Dict[int, 'CameraProcessor'] = {}
pi_client = None
logic_controller: Optional[LogicController] = None  # The proper traffic logic controller

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
                stats.emergency = True  # Only ambulance triggers emergency
            elif cls_id == 1:  # police
                stats.police_count += 1
                # Police does NOT trigger emergency override
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

class TrafficControlThread(threading.Thread):
    """
    Background thread for traffic signal control logic.
    
    Uses the proper TrafficController from src.logic.traffic_logic
    which implements the full specification:
    - Adaptive density-based control
    - Round robin fallback 
    - Starvation prevention
    - Emergency override with blue light blinking
    - Manual override
    - Gap-out logic
    - Min/Max green enforcement
    - Yellow transition
    """
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.signal_update_interval = 1.0
        self.current_phase_start = 0
        self.current_green_duration = 0
        self.blue_blink_thread = None
        
    def run(self):
        global state, pi_client, camera_threads, logic_controller
        
        last_decision_time = 0
        
        while self.running:
            if not state.running or logic_controller is None:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # Update lane stats from camera threads
            for lane_id, cam in camera_threads.items():
                stats = cam.stats
                state.lane_stats[lane_id] = stats
                
                # Update the logic controller with detection results
                logic_controller.update_lane(
                    lane_id,
                    vehicle_count=stats.vehicle_count,
                    ambulance_count=stats.ambulance_count,
                    police_count=stats.police_count,
                    camera_ok=stats.connected,
                    detection_ok=stats.connected
                )
            
            # Check for emergency preemption FIRST (interrupts any phase)
            emergency_detected = False
            emergency_lane_id = None
            for lane_id, cam in camera_threads.items():
                if cam.stats.ambulance_count > 0:
                    emergency_detected = True
                    emergency_lane_id = lane_id
                    break
            
            # If ambulance detected and we're NOT already in emergency for that lane
            if emergency_detected and emergency_lane_id is not None:
                if not state.emergency_active or state.current_green_lane != emergency_lane_id:
                    # Force immediate emergency override
                    lane_id, duration, mode = logic_controller.decide()
                    if mode == TrafficMode.EMERGENCY:
                        self._handle_emergency(lane_id, duration)
                        state.mode = "EMERGENCY"
                    else:
                        self._handle_emergency(emergency_lane_id, 30)
                        state.mode = "EMERGENCY"
                    time.sleep(0.1)
                    continue
            
            # If we're in emergency but ambulance is GONE → end emergency immediately
            if state.emergency_active and not emergency_detected:
                print("[CONTROL] Ambulance gone — ending emergency, returning to normal")
                lane_id, duration, mode = logic_controller.decide()
                if lane_id >= 0:
                    self._handle_normal(lane_id, duration, mode)
                else:
                    self._handle_all_red()
                state.mode = mode.value.upper()
                time.sleep(0.1)
                continue
            
            # Check if current phase has expired
            time_since_phase = current_time - self.current_phase_start
            
            if time_since_phase >= self.current_green_duration or state.current_green_lane < 0:
                # Make new decision
                lane_id, duration, mode = logic_controller.decide()
                
                # Handle mode/phase
                if mode == TrafficMode.EMERGENCY:
                    self._handle_emergency(lane_id, duration)
                elif mode == TrafficMode.MANUAL:
                    self._handle_manual(lane_id, duration)
                elif lane_id >= 0:
                    self._handle_normal(lane_id, duration, mode)
                else:
                    # All lanes inactive - keep all red
                    self._handle_all_red()
                
                state.mode = mode.value.upper()
            
            time.sleep(0.1)
    
    def _handle_normal(self, lane_id, duration, mode):
        """Handle normal AUTO/FAILSAFE signal phase"""
        global state, pi_client
        
        # Stop any blue light blinking
        self._stop_blue_blink()
        
        state.current_green_lane = lane_id
        state.green_start_time = time.time()
        state.emergency_active = False
        self.current_phase_start = time.time()
        self.current_green_duration = duration
        
        # Send to Pi
        if pi_client and lane_id >= 0:
            try:
                pi_client.send(lane=lane_id, duration=duration, emergency=False)
            except Exception as e:
                print(f"[PI] Error: {e}")
    
    def _handle_emergency(self, lane_id, duration):
        """
        Handle emergency phase with blue light blinking.
        
        Per specification:
        - Blue light shall BLINK (1 sec ON, 1 sec OFF)
        - All other lanes remain RED
        - Blue light turns OFF before returning to AUTO
        """
        global state, pi_client
        
        state.current_green_lane = lane_id
        state.green_start_time = time.time()
        state.emergency_active = True
        state.blue_light_blinking = True
        self.current_phase_start = time.time()
        self.current_green_duration = duration
        
        # Start blue light blinking in separate thread
        self._start_blue_blink(lane_id)
        
        # Send emergency signal to Pi
        if pi_client and lane_id >= 0:
            try:
                pi_client.send(lane=lane_id, duration=duration, emergency=True)
            except Exception as e:
                print(f"[PI] Error: {e}")
    
    def _handle_manual(self, lane_id, duration):
        """Handle manual mode"""
        global state, pi_client, logic_controller
        
        if logic_controller.manual_setting == "ALL_YELLOW_BLINK":
            # Yellow blink mode - handled by Pi
            self._stop_blue_blink()
            state.current_green_lane = -1
            state.emergency_active = False
            # TODO: Send yellow blink command to Pi
        elif logic_controller.manual_setting == "FORCE_LANE":
            self._handle_normal(lane_id, duration, TrafficMode.MANUAL)
        else:
            # NORMAL - switch back to auto (handled by controller.decide())
            pass
        
        self.current_phase_start = time.time()
        self.current_green_duration = duration if duration > 0 else 5
    
    def _handle_all_red(self):
        """Keep all lanes red when no vehicles"""
        global state, pi_client
        
        self._stop_blue_blink()
        
        state.current_green_lane = -1
        state.emergency_active = False
        self.current_phase_start = time.time()
        self.current_green_duration = 2  # Re-check after 2 seconds
        
        if pi_client:
            try:
                pi_client.send_all_red()
            except:
                pass
    
    def _start_blue_blink(self, lane_id):
        """Start blue light blinking in background"""
        if self.blue_blink_thread and self.blue_blink_thread.is_alive():
            return  # Already blinking
        
        def blink_loop():
            global pi_client
            while state.blue_light_blinking and state.emergency_active:
                # This is visual indicator only - actual GPIO handled by Pi
                time.sleep(1.0)  # ON duration
                if not state.blue_light_blinking:
                    break
                time.sleep(1.0)  # OFF duration
        
        self.blue_blink_thread = threading.Thread(target=blink_loop, daemon=True)
        self.blue_blink_thread.start()
    
    def _stop_blue_blink(self):
        """Stop blue light blinking"""
        state.blue_light_blinking = False
        state.emergency_active = False
    
    def stop(self):
        self._stop_blue_blink()
        self.running = False


traffic_controller: Optional[TrafficControlThread] = None

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the landing page"""
    return render_template('launcher.html')


@app.route('/dashboard')
def dashboard():
    """Serve the main dashboard (camera feed + signal lights)"""
    return render_template('dashboard_main.html')


@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the traffic management system"""
    global state, model, camera_threads, pi_client, traffic_controller, logic_controller
    
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
    
    # Initialize the proper traffic logic controller
    logic_controller = LogicController(
        num_lanes=4,
        min_green=5,
        max_green=30,
    )
    print("[INFO] Traffic logic controller initialized")
    
    # Start traffic controller thread
    if traffic_controller:
        traffic_controller.stop()
    traffic_controller = TrafficControlThread()
    traffic_controller.start()
    
    state.running = True
    state.current_green_lane = -1
    state.green_start_time = time.time()
    state.mode = "AUTO"
    
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
    global state, logic_controller
    
    remaining = 0
    green_duration = 15  # default
    if logic_controller and logic_controller.current_duration > 0:
        green_duration = logic_controller.current_duration
    if state.current_green_lane >= 0:
        remaining = max(0, green_duration - int(time.time() - state.green_start_time))
    
    # Get timing info from logic controller
    timing_info = {}
    if logic_controller:
        try:
            timing_info = logic_controller.get_timing_info()
        except Exception:
            timing_info = {}
    
    return jsonify({
        'running': state.running,
        'pi_connected': state.pi_connected,
        'current_green_lane': state.current_green_lane,
        'emergency_active': state.emergency_active,
        'blue_light_blinking': state.blue_light_blinking,
        'mode': state.mode,
        'remaining_time': remaining,
        'timing': timing_info,
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
    """Manually set a lane to green (sets MANUAL mode with FORCE_LANE)"""
    global state, pi_client, logic_controller
    
    if logic_controller:
        logic_controller.set_manual_setting("FORCE_LANE", forced_lane=lane_id)
    
    state.current_green_lane = lane_id
    state.green_start_time = time.time()
    state.emergency_active = False
    state.mode = "MANUAL"
    
    if pi_client:
        try:
            pi_client.send(lane=lane_id, duration=30, emergency=False)
        except:
            pass
    
    return jsonify({'status': 'ok', 'lane': lane_id, 'mode': 'MANUAL'})


@app.route('/api/mode', methods=['POST'])
def set_mode():
    """
    Set operating mode.
    
    POST JSON:
    {
        "mode": "AUTO" | "MANUAL" | "FAILURE",
        "setting": "FORCE_LANE" | "ALL_YELLOW_BLINK" | "NORMAL",  (for MANUAL mode)
        "lane": 0-3  (for FORCE_LANE)
    }
    """
    global state, logic_controller
    
    data = request.json
    mode = data.get('mode', 'AUTO').upper()
    setting = data.get('setting', 'NORMAL')
    forced_lane = data.get('lane', 0)
    
    if logic_controller is None:
        return jsonify({'status': 'error', 'message': 'System not started'}), 400
    
    if mode == 'AUTO':
        logic_controller.set_mode(TrafficMode.NORMAL)
        logic_controller.set_manual_setting("NORMAL")
        state.mode = "AUTO"
    elif mode == 'MANUAL':
        logic_controller.set_manual_setting(setting, forced_lane=forced_lane)
        state.mode = "MANUAL"
    elif mode == 'FAILURE':
        logic_controller.set_mode(TrafficMode.FAILURE)
        state.mode = "FAILURE"
    else:
        return jsonify({'status': 'error', 'message': f'Invalid mode: {mode}'}), 400
    
    return jsonify({
        'status': 'ok', 
        'mode': mode,
        'setting': setting if mode == 'MANUAL' else None
    })


@app.route('/api/timing', methods=['GET'])
def get_timing():
    """Get timing configuration"""
    global logic_controller
    
    if logic_controller:
        return jsonify(logic_controller.get_timing_info())
    return jsonify({})


# ============================================================================
# MAIN
# ============================================================================

def open_browser():
    """Open browser after a short delay to allow server to start"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 60)
    print("  AI TRAFFIC MANAGEMENT - WEB LAUNCHER")
    print("=" * 60)
    print()
    print("  Features:")
    print("  - Adaptive density-based control")
    print("  - Emergency vehicle priority with blue light blinking")
    print("  - Starvation prevention")
    print("  - Gap-out logic")
    print("  - Min/Max green enforcement")
    print()
    print("  Opening browser: http://localhost:5000")
    print()
    print("=" * 60)
    
    # Start browser opener thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
