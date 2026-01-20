"""
Multi-camera manager for 4-lane traffic monitoring.

OPTIMIZED FOR LOW LATENCY:
- Threaded frame grabbing (always grabs latest frame)
- Minimal buffering
- Non-blocking reads
"""

import cv2
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from queue import Queue, Empty


@dataclass
class CameraStatus:
    """Status of a single camera"""
    connected: bool = False
    last_frame_time: float = 0.0
    consecutive_failures: int = 0
    total_frames: int = 0
    error_message: str = ""


class ThreadedCamera:
    """
    Single threaded camera that continuously grabs frames.
    Always provides the LATEST frame with minimal latency.
    """
    
    def __init__(self, source: str, lane_id: int):
        self.source = source
        self.lane_id = lane_id
        self.cap = None
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.status = CameraStatus()
        self.last_grab_time = 0
        
    def start(self) -> bool:
        """Start the camera and grabbing thread"""
        try:
            # Use FFMPEG backend with low-latency options
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                self.status.connected = False
                self.status.error_message = "Failed to open stream"
                print(f"[CAM {self.lane_id}] Failed to connect to {self.source}")
                return False
            
            # CRITICAL: Set minimal buffer
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.status.connected = True
            self.status.error_message = ""
            self.running = True
            
            # Start background thread that continuously grabs frames
            self.thread = threading.Thread(target=self._grab_loop, daemon=True)
            self.thread.start()
            
            print(f"[CAM {self.lane_id}] Connected to {self.source}")
            return True
            
        except Exception as e:
            self.status.connected = False
            self.status.error_message = str(e)
            print(f"[CAM {self.lane_id}] Connection error: {e}")
            return False
    
    def _grab_loop(self):
        """
        Continuously grab frames in background.
        This is the KEY to low latency - always discarding old frames.
        """
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            try:
                # Grab frame (this also clears any buffer)
                grabbed = self.cap.grab()
                
                if grabbed:
                    # Decode the frame
                    ret, frame = self.cap.retrieve()
                    
                    if ret and frame is not None:
                        with self.lock:
                            self.frame = frame
                            self.grabbed = True
                            self.last_grab_time = time.time()
                            self.status.last_frame_time = self.last_grab_time
                            self.status.total_frames += 1
                            self.status.consecutive_failures = 0
                    else:
                        self.status.consecutive_failures += 1
                else:
                    self.status.consecutive_failures += 1
                    
            except Exception as e:
                self.status.consecutive_failures += 1
                self.status.error_message = str(e)
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
    
    def read(self) -> Optional[any]:
        """Get the latest frame (non-blocking)"""
        with self.lock:
            if self.grabbed and self.frame is not None:
                # Check frame age - reject frames older than 500ms
                age = time.time() - self.last_grab_time
                if age > 0.5:
                    return None
                return self.frame.copy()
            return None
    
    def stop(self):
        """Stop the camera"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status.connected = False


class CameraManager:
    """
    Manages multiple IP cameras for traffic monitoring.
    
    OPTIMIZED FEATURES:
    - Threaded frame grabbing per camera
    - Always returns latest frame (no buffering delay)
    - Non-blocking reads
    - Automatic reconnection
    """
    
    def __init__(self, sources: List[str], timeout: float = 2.0, 
                 max_failures: int = 5, reconnect_delay: float = 5.0):
        """
        Initialize camera manager.
        
        Args:
            sources: List of camera URLs
            timeout: Frame read timeout (not used with threaded cameras)
            max_failures: Max consecutive failures before reconnect
            reconnect_delay: Seconds to wait before reconnecting
        """
        self.sources = sources
        self.timeout = timeout
        self.max_failures = max_failures
        self.reconnect_delay = reconnect_delay
        
        self.num_cameras = len(sources)
        self.cameras: Dict[int, ThreadedCamera] = {}
        self.reconnect_threads: Dict[int, threading.Thread] = {}
        
        # Initialize all cameras
        for i, src in enumerate(sources):
            self.cameras[i] = ThreadedCamera(src, i)
            self.cameras[i].start()
    
    def read(self, lane_id: int) -> Optional[any]:
        """
        Read a frame from the specified camera (non-blocking).
        
        Args:
            lane_id: Camera/lane index
        
        Returns:
            Frame (numpy array) or None if unavailable
        """
        if lane_id not in self.cameras:
            return None
        
        camera = self.cameras[lane_id]
        frame = camera.read()
        
        # Check if we need to reconnect
        if frame is None and camera.status.consecutive_failures >= self.max_failures:
            self._schedule_reconnect(lane_id)
        
        return frame
    
    def _schedule_reconnect(self, lane_id: int):
        """Schedule a reconnection in a background thread"""
        if lane_id in self.reconnect_threads:
            if self.reconnect_threads[lane_id].is_alive():
                return  # Already reconnecting
        
        def reconnect():
            print(f"[CAM {lane_id}] Reconnecting...")
            time.sleep(self.reconnect_delay)
            camera = self.cameras[lane_id]
            camera.stop()
            camera.start()
        
        thread = threading.Thread(target=reconnect, daemon=True)
        thread.start()
        self.reconnect_threads[lane_id] = thread
    
    def read_all(self) -> Dict[int, Optional[any]]:
        """
        Read frames from all cameras (parallel, non-blocking).
        
        Returns:
            Dict mapping lane_id to frame (or None)
        """
        frames = {}
        for lane_id in range(self.num_cameras):
            frames[lane_id] = self.read(lane_id)
        return frames
    
    def get_status(self, lane_id: int) -> CameraStatus:
        """Get status for a specific camera"""
        if lane_id in self.cameras:
            return self.cameras[lane_id].status
        return CameraStatus()
    
    def get_all_status(self) -> Dict[int, CameraStatus]:
        """Get status for all cameras"""
        return {i: self.cameras[i].status for i in self.cameras}
    
    def is_healthy(self, lane_id: int) -> bool:
        """Check if a camera is healthy"""
        if lane_id not in self.cameras:
            return False
        status = self.cameras[lane_id].status
        return status.connected and status.consecutive_failures < self.max_failures
    
    def healthy_count(self) -> int:
        """Count number of healthy cameras"""
        return sum(1 for i in range(self.num_cameras) if self.is_healthy(i))
    
    def reconnect(self, lane_id: int) -> bool:
        """Manually trigger reconnection for a camera"""
        if lane_id in self.cameras:
            self.cameras[lane_id].stop()
            return self.cameras[lane_id].start()
        return False
    
    def reconnect_all(self):
        """Reconnect all cameras"""
        for lane_id in self.cameras:
            self.reconnect(lane_id)
    
    def release_all(self):
        """Release all camera connections"""
        for lane_id in self.cameras:
            self.cameras[lane_id].stop()
        self.cameras.clear()
        print("[CAM] All cameras released")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.release_all()

