"""
Centralized configuration for AI Traffic Management System

CAMERA SETUP INSTRUCTIONS:
==========================
1. Install "IP Webcam" app on your Android phone(s)
2. Open the app and configure:
   - Video preferences > Video resolution: 640x480
   - Video preferences > FPS: 15 (recommended for 4 cameras)
3. Start the server in the app
4. Note the IP address shown (e.g., http://192.168.1.100:8080)
5. Update CAMERA_SOURCES below with your camera IPs

For your hardware (i5 12th gen + RTX 2050):
- 4 cameras at 640x480, 15 FPS each = smooth performance
- Time-sliced detection processes 1 camera per cycle for efficiency
"""

# ============================
# CAMERA CONFIGURATION
# ============================
# IP camera URLs for 4 lanes
# Replace with your actual phone IP addresses from IP Webcam app
# Format: "http://<PHONE_IP>:8080/video"
CAMERA_SOURCES = [
    "http://192.168.137.52:8080/video",   # Lane 0 (North) - Phone 1
    "http://192.168.137.191:8080/video",  # Lane 1 (East)  - Phone 2
    "http://192.168.137.69:8080/video",   # Lane 2 (South) - Phone 3
    "http://192.168.137.179:8080/video",  # Lane 3 (West)  - Phone 4
]

# Number of lanes
NUM_LANES = 4

# ============================
# CAMERA PERFORMANCE SETTINGS
# ============================
# CRITICAL: Configure IP Webcam app on phones for LOW LATENCY:
# - Resolution: 640x480 (MANDATORY - higher = more latency)
# - FPS: 10 (lower = less latency)
# - Quality: 30-40% (lower quality = faster transmission)
# - Audio: DISABLED (reduces bandwidth)
# - Buffer: DISABLED or minimum

CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 10  # Target FPS

# ============================
# LATENCY OPTIMIZATION SETTINGS
# ============================
# These settings are tuned for minimal latency
CAMERA_TIMEOUT = 1.0           # Fast camera timeout
CAMERA_BUFFER_SIZE = 1         # Minimum buffer
STREAM_QUALITY = 50            # JPEG quality (lower = faster)

# ============================
# TIME-SLICED DETECTION (Performance Optimization)
# ============================
# Instead of detecting all 4 cameras every frame, we rotate through them
# This reduces GPU load from 4x to 1x per cycle while still detecting frequently
#
# With TIME_SLICE_ENABLED = True:
#   - Cycle 1: Detect Lane 0, read all cameras
#   - Cycle 2: Detect Lane 1, read all cameras
#   - Cycle 3: Detect Lane 2, read all cameras
#   - Cycle 4: Detect Lane 3, read all cameras
#   - (repeat)
#
# Each lane gets detected every ~400ms at 10 FPS processing

TIME_SLICE_ENABLED = True       # Enable time-sliced detection
TIME_SLICE_DETECT_ALL = False   # Set True to detect all cameras every cycle (slower)

# ============================
# RASPBERRY PI CONFIGURATION
# ============================
PI_IP = "192.168.1.50"
PI_PORT = 5000

# ============================
# MODEL CONFIGURATION
# ============================
# Path to trained traffic detection model
MODEL_PATH = "runs/detect/traffic_v14/weights/best.pt"

# Detection confidence threshold
DETECTION_CONFIDENCE = 0.5

# ============================
# TRAFFIC TIMING (seconds)
# ============================
MIN_GREEN_TIME = 5          # Minimum green light duration
MAX_GREEN_TIME = 30         # Maximum green light duration
BASE_GREEN_TIME = 10        # Base green time before density adjustment
EMERGENCY_GREEN_TIME = 30   # Green time for emergency vehicles

# Yellow/transition time between signals
YELLOW_TIME = 3

# Round-robin cycle time (failsafe mode)
FAILSAFE_CYCLE_TIME = 15

# ============================
# TRAFFIC LIGHT CONFIGURATION
# ============================
# Each lane has 4 lights:
#   - RED: Stop
#   - YELLOW: Prepare to stop/go
#   - GREEN: Go
#   - BLUE: Emergency vehicle approaching (shown on RED lanes)
#
# When emergency mode is active:
#   - Emergency lane: GREEN (let emergency vehicle pass)
#   - Other lanes: RED + BLUE (stopped, showing why)

# ============================
# EMERGENCY VEHICLE CONFIRMATION
# ============================
# Number of consecutive detections required to confirm emergency vehicle
# This prevents false positives from triggering emergency mode
EMERGENCY_CONFIRM_FRAMES = 3

# Cooldown after emergency (seconds) - prevents rapid re-triggering
EMERGENCY_COOLDOWN = 10

# ============================
# DETECTION LOOP TIMING
# ============================
# Detection cycle interval (seconds) - not used with threaded cameras
DETECTION_INTERVAL = 0.033  # ~30 FPS target

# ============================
# FAILSAFE CONFIGURATION
# ============================
# Max consecutive camera failures before entering failsafe mode
MAX_CAMERA_FAILURES = 5

# Max consecutive detection failures before entering failsafe mode
MAX_DETECTION_FAILURES = 3

# ============================
# LOGGING / DEBUG
# ============================
DEBUG_MODE = True
SHOW_VISUALIZATION = True
SAVE_DETECTION_FRAMES = False
DETECTION_FRAMES_DIR = "debug_frames"

# ============================
# CLASS DEFINITIONS (must match data.yaml)
# ============================
CLASS_NAMES = {
    0: "ambulance",
    1: "police",
    2: "vehicle"
}

# Classes that trigger emergency mode
EMERGENCY_CLASSES = {0, 1}  # ambulance, police
