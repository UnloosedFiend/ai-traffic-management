import time

from src.cameras.camera_manager import CameraManager
from src.vision.yolo_detector import YOLODetector
from src.logic.traffic_logic import decide_lane
from src.comms.pi_client import PiClient


def main():
    print("=== AI Traffic Management : Laptop Controller ===")

    # ============================
    # CAMERA CONFIGURATION
    # ============================
    CAMERA_SOURCES = [
        "http://192.168.1.3:8080/video"   # Phone IP camera (Lane 0)
    ]

    PI_IP = "192.168.1.50"   # Raspberry Pi IP (OK if unreachable for now)

    print("[INIT] Initializing components...")

    # ============================
    # CAMERA INITIALIZATION
    # ============================
    try:
        cams = CameraManager(CAMERA_SOURCES)
        print("[OK] Camera stream connected")
    except Exception as e:
        print("[FATAL] Camera initialization failed:", e)
        return

    # ============================
    # YOLO INITIALIZATION
    # ============================
    detector = YOLODetector()
    print("[OK] YOLO detector loaded")

    # ============================
    # PI COMMUNICATION
    # ============================
    pi = PiClient(PI_IP)
    print("[OK] Pi client ready")

    # ============================
    # LANE STATE (dynamic)
    # ============================
    lane_state = {
        0: {"count": 0, "emergency": False}
    }

    print("[RUN] Entering control loop...")

    # ============================
    # MAIN LOOP
    # ============================
    while True:
        # ---- Camera capture (NO YOLO yet) ----
        frame = cams.read(0)

        if frame is None:
            print("[CAM] Lane 0: No frame received")
            time.sleep(1)
            continue

        h, w, _ = frame.shape
        print(f"[CAM] Lane 0: Frame OK ({w}x{h})")

        # ---- TEMPORARY values (next step: YOLO) ----
        lane_state[0]["count"] = 1
        lane_state[0]["emergency"] = False

        # ---- Decision logic ----
        lane, duration, emergency = decide_lane(lane_state)

        print(
            f"[DECISION] Lane={lane} | Green={duration}s | Emergency={emergency}"
        )

        # ---- Send to Pi ----
        pi.send(lane, duration, emergency)

        time.sleep(5)


if __name__ == "__main__":
    main()
