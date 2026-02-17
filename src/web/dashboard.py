"""
Web Dashboard for AI Traffic Management System

Provides a real-time web interface showing:
- 4 camera feeds with detection overlays
- Vehicle counts per lane
- Current traffic signal state
- Emergency vehicle alerts
- System status and mode

Usage:
    python -m src.web.dashboard
    Then open http://localhost:5001 in browser
"""

import sys
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    CAMERA_SOURCES, NUM_LANES, PI_IP, PI_PORT,
    MODEL_PATH, DETECTION_CONFIDENCE,
    MIN_GREEN_TIME, MAX_GREEN_TIME, BASE_GREEN_TIME,
    EMERGENCY_GREEN_TIME, FAILSAFE_CYCLE_TIME, EMERGENCY_CONFIRM_FRAMES,
    EMERGENCY_COOLDOWN, TIME_SLICE_ENABLED, TIME_SLICE_DETECT_ALL
)
from src.cameras.camera_manager import CameraManager
from src.vision.yolo_detector import YOLODetector
from src.logic.traffic_logic import TrafficController, TrafficMode
from src.comms.pi_client import PiClient


# Flask app setup
app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static'))
app.config['SECRET_KEY'] = 'traffic-management-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


class TrafficDashboard:
    """
    Main dashboard controller that manages cameras, detection, and traffic logic.
    Implements time-sliced detection for optimal performance with 4 cameras.
    """
    
    def __init__(self):
        self.cameras = None
        self.detector = None
        self.controller = None
        self.pi_client = None
        
        self.running = False
        self.current_frames = {i: None for i in range(NUM_LANES)}
        self.current_results = {i: None for i in range(NUM_LANES)}
        self.current_lane = 0
        self.current_duration = 0
        self.current_mode = TrafficMode.FAILSAFE
        self.green_start_time = time.time()
        
        # Time-sliced detection state
        self.detection_lane_index = 0  # Which lane to detect this cycle
        
        self.lock = threading.Lock()
        self.detection_thread = None
        self.control_thread = None
        
    def initialize(self):
        """Initialize all components"""
        print("[DASHBOARD] Initializing components...")
        
        # Initialize cameras
        try:
            self.cameras = CameraManager(CAMERA_SOURCES, timeout=2.0, max_failures=5)
            healthy = self.cameras.healthy_count()
            print(f"[DASHBOARD] Cameras: {healthy}/{NUM_LANES} connected")
        except Exception as e:
            print(f"[DASHBOARD] Camera error: {e}")
            self.cameras = None
        
        # Initialize detector
        try:
            self.detector = YOLODetector(model_path=MODEL_PATH, conf=DETECTION_CONFIDENCE)
            if self.detector.is_ready():
                print("[DASHBOARD] YOLO detector ready (traffic_v14)")
            else:
                print("[DASHBOARD] YOLO detector not ready")
        except Exception as e:
            print(f"[DASHBOARD] Detector error: {e}")
            self.detector = None
        
        # Initialize controller
        self.controller = TrafficController(
            num_lanes=NUM_LANES,
            min_green=MIN_GREEN_TIME,
            max_green=MAX_GREEN_TIME,
            base_green=BASE_GREEN_TIME,
            emergency_green=EMERGENCY_GREEN_TIME,
            failsafe_cycle=FAILSAFE_CYCLE_TIME,
            emergency_confirm_required=EMERGENCY_CONFIRM_FRAMES,
            emergency_cooldown=EMERGENCY_COOLDOWN
        )
        print("[DASHBOARD] Traffic controller ready")
        
        # Initialize Pi client
        self.pi_client = PiClient(PI_IP, port=PI_PORT)
        print(f"[DASHBOARD] Pi client ready ({PI_IP}:{PI_PORT})")
        
        return True
    
    def start(self):
        """Start background processing threads"""
        self.running = True
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        print("[DASHBOARD] Background threads started")
    
    def stop(self):
        """Stop all threads"""
        self.running = False
        if self.cameras:
            self.cameras.release_all()
    
    def _detection_loop(self):
        """
        Background thread for continuous detection.
        
        OPTIMIZED FOR LOW LATENCY:
        - Threaded cameras provide latest frames instantly
        - Time-sliced detection reduces GPU load
        - Minimal processing per frame
        """
        last_emit_time = 0
        emit_interval = 0.2  # Emit status every 200ms max
        
        while self.running:
            if self.cameras is None or self.detector is None:
                time.sleep(0.1)
                continue
            
            cycle_start = time.time()
            
            # Read frames from ALL cameras (now instant with threaded cameras)
            frames = self.cameras.read_all()
            
            # Determine which lane(s) to run detection on this cycle
            if TIME_SLICE_ENABLED and not TIME_SLICE_DETECT_ALL:
                # Time-sliced: only detect one camera per cycle
                lanes_to_detect = [self.detection_lane_index]
                self.detection_lane_index = (self.detection_lane_index + 1) % NUM_LANES
            else:
                # Full detection: detect all cameras every cycle
                lanes_to_detect = list(range(NUM_LANES))
            
            for lane_id, frame in frames.items():
                if frame is None:
                    # Mark camera as failed
                    self.controller.update_lane(
                        lane_id, 0, 0, 0,
                        camera_ok=False, detection_ok=False
                    )
                    continue
                
                # Check if we should run detection on this lane this cycle
                if lane_id in lanes_to_detect:
                    # Run YOLO detection (GPU-intensive)
                    result = self.detector.detect_detailed(frame)
                    
                    # Annotate frame with detections
                    annotated = self.detector.annotate_frame(frame.copy(), result)
                    
                    with self.lock:
                        self.current_results[lane_id] = result
                    
                    # Update controller with detection results
                    self.controller.update_lane(
                        lane_id,
                        result.vehicle_count,
                        result.ambulance_count,
                        result.police_count,
                        camera_ok=True,
                        detection_ok=True
                    )
                else:
                    # No detection this cycle, use cached result for annotation
                    with self.lock:
                        result = self.current_results.get(lane_id)
                    
                    if result is not None:
                        annotated = self.detector.annotate_frame(frame.copy(), result)
                    else:
                        annotated = frame.copy()
                
                # Add lane info overlay (always update for live feed)
                if result is None:
                    # Create empty result for overlay
                    from src.vision.yolo_detector import DetectionResult
                    result = DetectionResult()
                
                annotated = self._add_lane_overlay(annotated, lane_id, result)
                
                with self.lock:
                    self.current_frames[lane_id] = annotated
            
            # Emit status update via WebSocket (throttled)
            now = time.time()
            if now - last_emit_time >= emit_interval:
                self._emit_status()
                last_emit_time = now
            
            # Minimal sleep - let threaded cameras do the work
            # Target ~30 FPS processing
            elapsed = time.time() - cycle_start
            sleep_time = max(0.001, 0.033 - elapsed)  # ~30 FPS
            time.sleep(sleep_time)
    
    def _control_loop(self):
        """Background thread for traffic control decisions"""
        while self.running:
            # Check if current green time has expired
            elapsed = time.time() - self.green_start_time
            
            if elapsed >= self.current_duration:
                # Make new decision
                lane_id, duration, mode = self.controller.decide()
                
                with self.lock:
                    self.current_lane = lane_id
                    self.current_duration = duration
                    self.current_mode = mode
                    self.green_start_time = time.time()
                
                # Send to Pi
                is_emergency = (mode == TrafficMode.EMERGENCY)
                self.pi_client.send(lane_id, duration, emergency=is_emergency)
                
                print(f"[CONTROL] Lane {lane_id} GREEN for {duration}s ({mode.value})")
            
            time.sleep(0.5)
    
    def _add_lane_overlay(self, frame, lane_id, result):
        """Add lane info overlay to frame"""
        h, w = frame.shape[:2]
        
        # Determine if this lane is green
        with self.lock:
            is_green = (lane_id == self.current_lane)
            remaining = max(0, self.current_duration - (time.time() - self.green_start_time))
        
        # Draw status bar at top
        bar_color = (0, 200, 0) if is_green else (0, 0, 200)
        cv2.rectangle(frame, (0, 0), (w, 40), bar_color, -1)
        
        # Lane label
        status_text = f"LANE {lane_id} - {'GREEN' if is_green else 'RED'}"
        if is_green:
            status_text += f" ({remaining:.0f}s)"
        cv2.putText(frame, status_text, (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Detection counts
        count_text = f"V:{result.vehicle_count} A:{result.ambulance_count} P:{result.police_count}"
        cv2.putText(frame, count_text, (w - 200, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Emergency alert
        if result.emergency_detected:
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 255), -1)
            cv2.putText(frame, "! EMERGENCY VEHICLE DETECTED !", (w//2 - 200, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def _emit_status(self):
        """Emit current status via WebSocket"""
        with self.lock:
            remaining = max(0, self.current_duration - (time.time() - self.green_start_time))
            
            lane_data = {}
            for lane_id in range(NUM_LANES):
                result = self.current_results.get(lane_id)
                cam_status = self.cameras.get_status(lane_id) if self.cameras else None
                
                lane_data[lane_id] = {
                    'vehicles': result.vehicle_count if result else 0,
                    'ambulances': result.ambulance_count if result else 0,
                    'police': result.police_count if result else 0,
                    'emergency': result.emergency_detected if result else False,
                    'camera_ok': cam_status.connected if cam_status else False,
                    'is_green': lane_id == self.current_lane
                }
            
            status = {
                'current_lane': self.current_lane,
                'duration': self.current_duration,
                'remaining': round(remaining, 1),
                'mode': self.current_mode.value,
                'lanes': lane_data,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
        
        socketio.emit('status_update', status)
    
    def get_frame(self, lane_id):
        """Get current frame for a lane (for MJPEG streaming)"""
        with self.lock:
            frame = self.current_frames.get(lane_id)
        
        if frame is None:
            # Return placeholder
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Lane {lane_id}: No Signal", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        
        return frame
    
    def get_status(self):
        """Get current system status"""
        with self.lock:
            remaining = max(0, self.current_duration - (time.time() - self.green_start_time))
            
            return {
                'current_lane': self.current_lane,
                'duration': self.current_duration,
                'remaining': round(remaining, 1),
                'mode': self.current_mode.value,
                'pi_connected': self.pi_client.is_healthy() if self.pi_client else False,
                'cameras_healthy': self.cameras.healthy_count() if self.cameras else 0,
                'detector_ready': self.detector.is_ready() if self.detector else False
            }


# Global dashboard instance
dashboard = TrafficDashboard()


def generate_frames(lane_id):
    """Generator for MJPEG video stream - OPTIMIZED FOR LOW LATENCY"""
    last_frame = None
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50]  # Lower quality = faster
    
    while True:
        frame = dashboard.get_frame(lane_id)
        
        # Skip if same frame (no new data)
        if frame is last_frame:
            time.sleep(0.01)
            continue
        
        last_frame = frame
        
        # Resize for faster encoding if needed
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, 480))
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS streaming


# Flask Routes
@app.route('/')
def index():
    """Main dashboard page - minimal view"""
    return render_template('dashboard_minimal.html')


@app.route('/full')
def index_full():
    """Full dashboard page with camera feeds"""
    return render_template('dashboard.html')


@app.route('/video/<int:lane_id>')
def video_feed(lane_id):
    """MJPEG video stream for a lane"""
    if lane_id < 0 or lane_id >= NUM_LANES:
        return "Invalid lane", 404
    return Response(generate_frames(lane_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    """API endpoint for current status"""
    return jsonify(dashboard.get_status())


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print("[WS] Client connected")
    dashboard._emit_status()


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print("[WS] Client disconnected")


def run_dashboard(host='0.0.0.0', port=5001, debug=False):
    """Run the dashboard server"""
    print("=" * 60)
    print("  AI TRAFFIC MANAGEMENT - WEB DASHBOARD")
    print("=" * 60)
    print()
    
    dashboard.initialize()
    dashboard.start()
    
    print()
    print(f"[SERVER] Starting web server on http://{host}:{port}")
    print("[SERVER] Open this URL in your browser to view the dashboard")
    print()
    
    try:
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    finally:
        dashboard.stop()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Traffic Management Dashboard")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    run_dashboard(host=args.host, port=args.port, debug=args.debug)
