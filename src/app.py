#!/usr/bin/env python3
"""
AI Traffic Management System - Main Application

This is the main entry point for the desktop/laptop controller.
It manages:
- 4 IP cameras for vehicle detection
- YOLO-based detection of vehicles, ambulances, police
- Priority-based lane selection with emergency preemption
- Fail-safe round-robin when detection fails
- Communication with Raspberry Pi for signal control

Usage:
    python -m src.app
    or
    python src/app.py
"""

import sys
import time
import signal
import argparse
import cv2
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, '.')

from src.config import (
    CAMERA_SOURCES, NUM_LANES, PI_IP, PI_PORT,
    MODEL_PATH, DETECTION_CONFIDENCE,
    DETECTION_INTERVAL, EMERGENCY_CONFIRM_FRAMES,
    DEBUG_MODE, SHOW_VISUALIZATION,
    MIN_GREEN_TIME, MAX_GREEN_TIME, BASE_GREEN_TIME,
    EMERGENCY_GREEN_TIME, FAILSAFE_CYCLE_TIME, EMERGENCY_COOLDOWN,
    TIME_SLICE_ENABLED, TIME_SLICE_DETECT_ALL
)
from src.cameras.camera_manager import CameraManager
from src.vision.yolo_detector import YOLODetector
from src.logic.traffic_logic import TrafficController, TrafficMode
from src.comms.pi_client import PiClient


class TrafficManagementApp:
    """
    Main application class for AI Traffic Management.
    
    Coordinates cameras, detection, traffic logic, and Pi communication.
    """
    
    def __init__(self, camera_sources=None, pi_ip=None, pi_port=None,
                 model_path=None, show_video=False, debug=False):
        """
        Initialize the traffic management application.
        
        Args:
            camera_sources: List of camera URLs (uses config if None)
            pi_ip: Raspberry Pi IP address
            pi_port: Raspberry Pi port
            model_path: Path to YOLO model
            show_video: Show detection visualization
            debug: Enable debug output
        """
        self.camera_sources = camera_sources or CAMERA_SOURCES
        self.pi_ip = pi_ip or PI_IP
        self.pi_port = pi_port or PI_PORT
        self.model_path = model_path or MODEL_PATH
        self.show_video = show_video or SHOW_VISUALIZATION
        self.debug = debug or DEBUG_MODE
        
        self.running = False
        self.cameras = None
        self.detector = None
        self.controller = None
        self.pi_client = None
        
        # Time-sliced detection state
        self.detection_lane_index = 0
        
        # Statistics
        self.stats = {
            "cycles": 0,
            "emergency_activations": 0,
            "failsafe_activations": 0,
            "start_time": None
        }
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful
        """
        print("=" * 60)
        print("  AI TRAFFIC MANAGEMENT SYSTEM")
        print("  4-Lane Priority Controller with Emergency Preemption")
        print("=" * 60)
        print()
        
        # Initialize cameras
        print("[INIT] Connecting to cameras...")
        try:
            self.cameras = CameraManager(
                self.camera_sources,
                timeout=2.0,
                max_failures=5
            )
            healthy = self.cameras.healthy_count()
            print(f"[OK] Cameras: {healthy}/{len(self.camera_sources)} connected")
        except Exception as e:
            print(f"[WARN] Camera init error: {e}")
            print("[WARN] Will operate in failsafe mode")
            self.cameras = None
        
        # Initialize YOLO detector
        print(f"[INIT] Loading YOLO model: {self.model_path}")
        try:
            self.detector = YOLODetector(
                model_path=self.model_path,
                conf=DETECTION_CONFIDENCE
            )
            if self.detector.is_ready():
                print("[OK] YOLO detector ready")
            else:
                print("[WARN] YOLO detector not ready, using failsafe")
        except Exception as e:
            print(f"[WARN] Detector init error: {e}")
            self.detector = None
        
        # Initialize traffic controller
        print("[INIT] Initializing traffic controller...")
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
        print("[OK] Traffic controller ready")
        
        # Initialize Pi client
        print(f"[INIT] Connecting to Raspberry Pi at {self.pi_ip}:{self.pi_port}")
        self.pi_client = PiClient(
            self.pi_ip,
            port=self.pi_port,
            timeout=2.0,
            max_retries=3
        )
        if self.pi_client.check_connection():
            print("[OK] Raspberry Pi connected")
        else:
            print("[WARN] Raspberry Pi not reachable (will retry during operation)")
        
        print()
        print("[INIT] Initialization complete")
        print("-" * 60)
        
        return True
    
    def detect_all_lanes(self):
        """
        Run detection on camera feeds and update controller.
        
        Uses time-sliced detection for performance optimization:
        - Reads all camera frames every cycle
        - Runs YOLO on only one camera per cycle (rotates through lanes)
        - Reduces GPU load by 4x while maintaining responsive detection
        """
        if self.cameras is None or self.detector is None:
            # No cameras or detector - mark all lanes as failed
            for lane_id in range(NUM_LANES):
                self.controller.update_lane(
                    lane_id, 0, 0, 0,
                    camera_ok=False,
                    detection_ok=False
                )
            return
        
        frames = self.cameras.read_all()
        
        # Determine which lane(s) to detect this cycle
        if TIME_SLICE_ENABLED and not TIME_SLICE_DETECT_ALL:
            lanes_to_detect = [self.detection_lane_index]
            self.detection_lane_index = (self.detection_lane_index + 1) % NUM_LANES
        else:
            lanes_to_detect = list(range(NUM_LANES))
        
        for lane_id, frame in frames.items():
            camera_ok = frame is not None
            
            if not camera_ok:
                self.controller.update_lane(
                    lane_id, 0, 0, 0,
                    camera_ok=False,
                    detection_ok=False
                )
                continue
            
            # Only run detection on selected lane(s) this cycle
            if lane_id in lanes_to_detect and self.detector is not None:
                try:
                    result = self.detector.detect_detailed(frame)
                    
                    if self.debug:
                        total = result.total_count
                        emerg = "🚨" if result.emergency_detected else ""
                        print(f"  Lane {lane_id}: {total} vehicles "
                              f"(A:{result.ambulance_count} P:{result.police_count} V:{result.vehicle_count}) {emerg}")
                    
                    # Show visualization if enabled
                    if self.show_video:
                        annotated = self.detector.annotate_frame(frame.copy(), result)
                        window_name = f"Lane {lane_id}"
                        cv2.imshow(window_name, annotated)
                    
                    # Update controller with detection results
                    self.controller.update_lane(
                        lane_id,
                        result.vehicle_count,
                        result.ambulance_count,
                        result.police_count,
                        camera_ok=True,
                        detection_ok=True
                    )
                        
                except Exception as e:
                    if self.debug:
                        print(f"  Lane {lane_id}: Detection error: {e}")
            else:
                # Just show the frame without new detection
                if self.show_video:
                    window_name = f"Lane {lane_id}"
                    cv2.imshow(window_name, frame)
    
    def run_cycle(self):
        """
        Run one traffic control cycle.
        """
        self.stats["cycles"] += 1
        
        if self.debug:
            print(f"\n[CYCLE {self.stats['cycles']}] "
                  f"{datetime.now().strftime('%H:%M:%S')}")
        
        # Run detection on all lanes
        self.detect_all_lanes()
        
        # Get traffic decision
        lane_id, duration, mode = self.controller.decide()
        
        # Track statistics
        if mode == TrafficMode.EMERGENCY:
            self.stats["emergency_activations"] += 1
        elif mode == TrafficMode.FAILSAFE:
            self.stats["failsafe_activations"] += 1
        
        # Print decision
        mode_str = mode.value.upper()
        mode_emoji = {"normal": "🟢", "emergency": "🚨", "failsafe": "⚠️"}.get(mode.value, "")
        print(f"[DECISION] {mode_emoji} Lane {lane_id} GREEN for {duration}s ({mode_str})")
        
        # Send command to Raspberry Pi
        is_emergency = (mode == TrafficMode.EMERGENCY)
        success = self.pi_client.send(lane_id, duration, emergency=is_emergency)
        
        if not success and self.debug:
            print("[WARN] Failed to send command to Pi")
        
        # Handle OpenCV window events
        if self.show_video:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
        
        return lane_id, duration, mode
    
    def run(self):
        """
        Main run loop.
        """
        self.running = True
        self.stats["start_time"] = time.time()
        
        print()
        print("[RUN] Starting traffic control loop...")
        print("[RUN] Press Ctrl+C to stop")
        print()
        
        try:
            while self.running:
                cycle_start = time.time()
                
                # Run one control cycle
                lane_id, duration, mode = self.run_cycle()
                
                # Wait for green duration (or check more frequently)
                # In real deployment, we wait the full duration
                # For testing, we can use shorter intervals
                wait_time = min(duration, DETECTION_INTERVAL)
                
                # Account for cycle processing time
                elapsed = time.time() - cycle_start
                sleep_time = max(0, wait_time - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n[STOP] Keyboard interrupt received")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """
        Clean shutdown of all components.
        """
        print()
        print("[SHUTDOWN] Stopping traffic management system...")
        
        self.running = False
        
        # Send all-red for safety
        if self.pi_client:
            self.pi_client.send_all_red()
        
        # Release cameras
        if self.cameras:
            self.cameras.release_all()
        
        # Close visualization windows
        if self.show_video:
            cv2.destroyAllWindows()
        
        # Print statistics
        if self.stats["start_time"]:
            runtime = time.time() - self.stats["start_time"]
            print()
            print("=" * 40)
            print("  SESSION STATISTICS")
            print("=" * 40)
            print(f"  Runtime: {runtime:.1f} seconds")
            print(f"  Cycles: {self.stats['cycles']}")
            print(f"  Emergency activations: {self.stats['emergency_activations']}")
            print(f"  Failsafe activations: {self.stats['failsafe_activations']}")
            print("=" * 40)
        
        print("[SHUTDOWN] Complete")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Traffic Management System"
    )
    parser.add_argument(
        "--cameras", "-c",
        nargs="+",
        help="Camera URLs (overrides config)"
    )
    parser.add_argument(
        "--pi-ip",
        default=None,
        help="Raspberry Pi IP address"
    )
    parser.add_argument(
        "--pi-port",
        type=int,
        default=None,
        help="Raspberry Pi port"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to YOLO model"
    )
    parser.add_argument(
        "--show-video", "-v",
        action="store_true",
        help="Show detection visualization"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug output"
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    app = TrafficManagementApp(
        camera_sources=args.cameras,
        pi_ip=args.pi_ip,
        pi_port=args.pi_port,
        model_path=args.model,
        show_video=args.show_video,
        debug=args.debug
    )
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n[SIGNAL] Received shutdown signal")
        app.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run
    if app.initialize():
        app.run()
    else:
        print("[FATAL] Initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

