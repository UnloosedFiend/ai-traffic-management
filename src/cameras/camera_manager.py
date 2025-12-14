import cv2

class CameraManager:
    def __init__(self, sources):
        """
        sources: list of camera sources
        Each source can be:
          - int (USB camera index)
          - str (IP camera URL)
        """
        self.cams = []

        for src in sources:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise RuntimeError(f"Camera source not accessible: {src}")
            self.cams.append(cap)

    def read(self, lane_id):
        if lane_id >= len(self.cams):
            return None

        ret, frame = self.cams[lane_id].read()
        if not ret:
            return None

        return frame

    def release_all(self):
        for cap in self.cams:
            cap.release()
