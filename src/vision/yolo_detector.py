from ultralytics import YOLO

# Class IDs for emergency vehicles (as per data.yaml)
EMERGENCY_CLASSES = {5, 6}  # ambulance, firetruck

class YOLODetector:
    def __init__(self, model_path="models/yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Runs YOLO on a frame.
        Returns:
          vehicle_count (int)
          emergency_present (bool)
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        vehicle_count = 0
        emergency = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            vehicle_count += 1
            if cls_id in EMERGENCY_CLASSES:
                emergency = True

        return vehicle_count, emergency
