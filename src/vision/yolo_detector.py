"""
YOLO-based vehicle detector for traffic management.

Detects vehicles, ambulances, and police vehicles from camera frames.
"""

from ultralytics import YOLO
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


# Class IDs (must match data.yaml)
# 0 = ambulance, 1 = police, 2 = vehicle
EMERGENCY_CLASSES = {0, 1}  # ambulance, police

CLASS_NAMES = {
    0: "ambulance",
    1: "police", 
    2: "vehicle"
}


@dataclass
class DetectionResult:
    """Results from a single frame detection"""
    vehicle_count: int = 0
    ambulance_count: int = 0
    police_count: int = 0
    total_count: int = 0
    emergency_detected: bool = False
    detections: List[dict] = None
    
    def __post_init__(self):
        if self.detections is None:
            self.detections = []


class YOLODetector:
    """
    YOLO-based vehicle detector with detailed counting.
    
    Features:
    - Separate counts for vehicles, ambulances, police
    - Configurable confidence threshold
    - Optional frame annotation
    - Error handling for robust operation
    """
    
    def __init__(self, model_path: str = "runs/detect/traffic_v14/weights/best.pt", 
                 conf: float = 0.5):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained YOLO model weights
            conf: Detection confidence threshold (0.0-1.0)
        """
        self.model_path = model_path
        self.conf = conf
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            self.model = YOLO(self.model_path)
            print(f"[YOLO] Model loaded: {self.model_path}")
        except Exception as e:
            print(f"[YOLO] ERROR loading model: {e}")
            # Try fallback to base model
            try:
                self.model = YOLO("yolov8n.pt")
                print("[YOLO] Fallback to base yolov8n.pt")
            except Exception as e2:
                print(f"[YOLO] FATAL: Could not load any model: {e2}")
                self.model = None
    
    def detect(self, frame) -> Tuple[int, bool]:
        """
        Legacy detection interface.
        
        Args:
            frame: OpenCV image (BGR)
        
        Returns:
            (vehicle_count, emergency_present)
        """
        result = self.detect_detailed(frame)
        return result.total_count, result.emergency_detected
    
    def detect_detailed(self, frame) -> DetectionResult:
        """
        Run detection with detailed counting.
        
        Args:
            frame: OpenCV image (BGR)
        
        Returns:
            DetectionResult with counts and detections
        """
        result = DetectionResult()
        
        if self.model is None:
            return result
        
        if frame is None or not isinstance(frame, np.ndarray):
            return result
        
        try:
            # Run inference
            outputs = self.model(frame, conf=self.conf, verbose=False)
            
            if not outputs or len(outputs) == 0:
                return result
            
            boxes = outputs[0].boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detection = {
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                }
                result.detections.append(detection)
                
                # Count by class
                if cls_id == 0:  # ambulance
                    result.ambulance_count += 1
                    result.emergency_detected = True
                elif cls_id == 1:  # police
                    result.police_count += 1
                    result.emergency_detected = True
                elif cls_id == 2:  # vehicle
                    result.vehicle_count += 1
            
            result.total_count = (result.vehicle_count + 
                                   result.ambulance_count + 
                                   result.police_count)
            
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
        
        return result
    
    def annotate_frame(self, frame, result: DetectionResult):
        """
        Draw detection boxes on frame.
        
        Args:
            frame: OpenCV image to annotate (modified in place)
            result: DetectionResult from detect_detailed()
        
        Returns:
            Annotated frame
        """
        import cv2
        
        COLORS = {
            0: (0, 0, 255),    # ambulance - red
            1: (255, 0, 0),    # police - blue
            2: (0, 255, 0)     # vehicle - green
        }
        
        for det in result.detections:
            cls_id = det["class_id"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            
            color = COLORS.get(cls_id, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def is_ready(self) -> bool:
        """Check if detector is ready for inference"""
        return self.model is not None

