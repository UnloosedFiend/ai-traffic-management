"""
Real-time IP Webcam Inference Script
=====================================
This script runs real-time object detection using your trained YOLO model
on video stream from an IP webcam (like IP Webcam app on Android).

Usage:
    python scripts/run_realtime_inference.py --source "http://YOUR_PHONE_IP:8080/video"
    
    Or use default webcam:
    python scripts/run_realtime_inference.py --source 0
"""

import argparse
import cv2
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO


# Class names as per your data.yaml (traffic_v14 model)
CLASS_NAMES = {0: "ambulance", 1: "police", 2: "vehicle"}

# Colors for visualization (BGR format)
CLASS_COLORS = {
    0: (0, 0, 255),     # ambulance - Red
    1: (255, 0, 0),     # police - Blue  
    2: (0, 255, 0)      # vehicle - Green
}


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time YOLO inference on IP webcam")
    parser.add_argument(
        "--source", 
        type=str, 
        default="http://192.168.1.3:8080/video",
        help="Video source: IP camera URL or webcam index (0, 1, etc.)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="runs/detect/train2/weights/best.pt",
        help="Path to trained YOLO model weights"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.4,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640,
        help="Inference image size"
    )
    parser.add_argument(
        "--save-video", 
        action="store_true",
        help="Save output video"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output_detection.mp4",
        help="Output video path (if --save-video is set)"
    )
    return parser.parse_args()


def draw_detections(frame, results, class_names, class_colors):
    """Draw bounding boxes and labels on frame"""
    vehicle_count = 0
    ambulance_count = 0
    police_count = 0
    emergency_present = False
    
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get class and confidence
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # DEBUG: Print raw detection info
        print(f"[DEBUG] Detected class_id={cls_id}, conf={conf:.2f}")
        
        # Get class name and color
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        color = class_colors.get(cls_id, (255, 255, 255))
        
        # Count by class
        if cls_id == 0:  # ambulance
            ambulance_count += 1
            emergency_present = True
        elif cls_id == 1:  # police
            police_count += 1
            emergency_present = True
        elif cls_id == 2:  # vehicle
            vehicle_count += 1
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{cls_name}: {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, {
        "vehicle": vehicle_count,
        "ambulance": ambulance_count,
        "police": police_count,
        "emergency": emergency_present
    }


def draw_stats(frame, stats, fps):
    """Draw statistics overlay on frame"""
    h, w = frame.shape[:2]
    
    # Draw semi-transparent background for stats
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Draw stats text
    y_offset = 35
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"Vehicles: {stats['vehicle']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    y_offset += 30
    cv2.putText(frame, f"Ambulances: {stats['ambulance']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"Police: {stats['police']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    y_offset += 30
    emergency_text = "EMERGENCY DETECTED!" if stats['emergency'] else "No Emergency"
    emergency_color = (0, 0, 255) if stats['emergency'] else (0, 255, 0)
    cv2.putText(frame, emergency_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emergency_color, 2)
    
    return frame


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  AI Traffic Management - Real-time Inference")
    print("=" * 60)
    
    # Resolve model path
    model_path = Path(project_root) / args.model
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Available trained models:")
        weights_dir = Path(project_root) / "runs" / "detect"
        if weights_dir.exists():
            for run_dir in weights_dir.iterdir():
                best_pt = run_dir / "weights" / "best.pt"
                if best_pt.exists():
                    print(f"  - {best_pt.relative_to(project_root)}")
        return
    
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    print("[OK] Model loaded successfully")
    
    # Parse video source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    print(f"[INFO] Connecting to video source: {source}")
    
    # Open video capture
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {source}")
        print("\nTroubleshooting tips:")
        print("1. Make sure IP Webcam app is running on your phone")
        print("2. Check that your PC and phone are on the same network")
        print("3. Verify the IP address and port in the URL")
        print("4. Try opening the URL in a web browser first")
        return
    
    print("[OK] Video stream connected")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution: {frame_width}x{frame_height}")
    
    # Setup video writer if saving
    video_writer = None
    if args.save_video:
        output_path = Path(project_root) / args.output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (frame_width, frame_height))
        print(f"[INFO] Saving output to: {output_path}")
    
    print("\n" + "=" * 60)
    print("  Press 'q' to quit | Press 's' to save screenshot")
    print("=" * 60 + "\n")
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[WARN] Failed to read frame, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                continue
            
            # Run inference
            results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            
            # Draw detections
            frame, stats = draw_detections(frame, results, CLASS_NAMES, CLASS_COLORS)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Draw stats overlay
            frame = draw_stats(frame, stats, fps)
            
            # Save video frame
            if video_writer:
                video_writer.write(frame)
            
            # Display frame
            cv2.imshow("AI Traffic Detection - Press 'q' to quit", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                screenshot_path = Path(project_root) / f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"[INFO] Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
