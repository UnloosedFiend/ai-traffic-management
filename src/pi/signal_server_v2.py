"""
Raspberry Pi Traffic Signal Server (Updated)
=============================================
This is the updated signal_server.py to run on your Raspberry Pi.

Features:
- Emergency mode: Green + blinking Blue on emergency lane, Red only on other lanes
- Normal mode: Green on priority lane, Red on others
- Full state control endpoint

Upload this to your Raspberry Pi and run:
    python3 signal_server_v2.py

Endpoints:
    POST /set_signal - Set single lane green
    POST /set_state  - Set full traffic state (all 4 lanes)
    GET /status      - Get current signal status
"""

from flask import Flask, request, jsonify
import RPi.GPIO as GPIO
import threading

app = Flask(__name__)

# Blue light blink control
_blink_thread = None
_blink_active = False
_blink_lock = threading.Lock()

GPIO.setmode(GPIO.BCM)

# GPIO Pin assignments for each lane
# R=Red, Y=Yellow, G=Green, B=Blue
LANES = {
    "NORTH": {"R": 17, "Y": 27, "G": 22, "B": 23},
    "EAST":  {"R": 5,  "Y": 6,  "G": 13, "B": 19},
    "SOUTH": {"R": 12, "Y": 16, "G": 20, "B": 21},
    "WEST":  {"R": 24, "Y": 25, "G": 18, "B": 26}
}

LANE_ORDER = ["NORTH", "EAST", "SOUTH", "WEST"]

# Initialize all pins
for lane in LANES.values():
    for pin in lane.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

# Current state tracking
current_state = {
    "green_lane": None,
    "emergency": False,
    "emergency_lane": None
}


def all_off():
    """Turn off all lights"""
    for lane in LANES.values():
        GPIO.output(lane["R"], GPIO.LOW)
        GPIO.output(lane["Y"], GPIO.LOW)
        GPIO.output(lane["G"], GPIO.LOW)
        GPIO.output(lane["B"], GPIO.LOW)


def all_red():
    """Set all lanes to red"""
    for lane in LANES.values():
        GPIO.output(lane["G"], GPIO.LOW)
        GPIO.output(lane["Y"], GPIO.LOW)
        GPIO.output(lane["B"], GPIO.LOW)
        GPIO.output(lane["R"], GPIO.HIGH)


def stop_blink():
    """Stop any active blue light blinking"""
    global _blink_active, _blink_thread
    _blink_active = False
    if _blink_thread and _blink_thread.is_alive():
        _blink_thread.join(timeout=1.0)
    _blink_thread = None
    # Ensure all blue LEDs are off after stopping
    for lane in LANES.values():
        GPIO.output(lane["B"], GPIO.LOW)


def start_blink(lane_name):
    """Start blinking blue LED on the specified lane"""
    global _blink_active, _blink_thread
    stop_blink()  # Stop any existing blink first
    
    if lane_name not in LANES:
        return
    
    _blink_active = True
    
    def blink_loop():
        pin = LANES[lane_name]["B"]
        while _blink_active:
            GPIO.output(pin, GPIO.HIGH)
            # Blink on for 0.3s
            for _ in range(6):  # 6 × 0.05s = 0.3s
                if not _blink_active:
                    GPIO.output(pin, GPIO.LOW)
                    return
                import time
                time.sleep(0.05)
            GPIO.output(pin, GPIO.LOW)
            # Blink off for 0.3s
            for _ in range(6):
                if not _blink_active:
                    return
                import time
                time.sleep(0.05)
    
    _blink_thread = threading.Thread(target=blink_loop, daemon=True)
    _blink_thread.start()


def set_lane_green(lane_name, with_blue=False):
    """Set a specific lane to green. If with_blue, starts blinking blue."""
    if lane_name not in LANES:
        return False
    
    lane = LANES[lane_name]
    GPIO.output(lane["R"], GPIO.LOW)
    GPIO.output(lane["G"], GPIO.HIGH)
    
    if with_blue:
        start_blink(lane_name)
    else:
        GPIO.output(lane["B"], GPIO.LOW)
    
    return True


def set_lane_red(lane_name, with_blue=False):
    """Set a specific lane to red"""
    if lane_name not in LANES:
        return False
    
    lane = LANES[lane_name]
    GPIO.output(lane["G"], GPIO.LOW)
    GPIO.output(lane["R"], GPIO.HIGH)
    
    if with_blue:
        GPIO.output(lane["B"], GPIO.HIGH)
    else:
        GPIO.output(lane["B"], GPIO.LOW)
    
    return True


@app.route("/set_signal", methods=["POST"])
def set_signal():
    """
    Set a single lane to green (original endpoint for compatibility).
    
    POST JSON:
    {
        "lane": "NORTH",        # Lane to set green
        "emergency": false      # If true, adds blue light
    }
    """
    data = request.json
    lane = data.get("lane")
    emergency = data.get("emergency", False)
    
    if lane not in LANES:
        return jsonify({"status": "error", "message": f"Invalid lane: {lane}"}), 400
    
    # Stop any existing blue blink
    stop_blink()
    
    # Set all lanes to red first
    all_red()
    
    # Set the requested lane to green
    # Emergency: GREEN + blinking BLUE on emergency lane, RED only on others
    set_lane_green(lane, with_blue=emergency)
    
    # Update state
    current_state["green_lane"] = lane
    current_state["emergency"] = emergency
    current_state["emergency_lane"] = lane if emergency else None
    
    return jsonify({"status": "OK", "lane": lane, "emergency": emergency})


@app.route("/set_state", methods=["POST"])
def set_state():
    """
    Set the full traffic state for all 4 lanes.
    
    POST JSON:
    {
        "green_lane": "NORTH",      # Which lane is green (null for all red)
        "emergency": false,         # Emergency mode active
        "emergency_lane": null      # Which lane has emergency vehicle
    }
    
    Behavior:
    - If emergency=true and emergency_lane is set:
        - emergency_lane: GREEN + blinking BLUE
        - other lanes: RED only
    - If emergency=false and green_lane is set:
        - green_lane: GREEN
        - other lanes: RED
    - If green_lane is null:
        - all lanes: RED
    """
    data = request.json
    green_lane = data.get("green_lane")
    emergency = data.get("emergency", False)
    emergency_lane = data.get("emergency_lane")
    
    # Stop any existing blue blink
    stop_blink()
    
    # Start with all red
    all_red()
    
    if emergency and emergency_lane and emergency_lane in LANES:
        # Emergency mode: emergency lane gets GREEN + blinking BLUE
        # Other lanes get RED only (no blue)
        set_lane_green(emergency_lane, with_blue=True)
        
        current_state["green_lane"] = emergency_lane
        current_state["emergency"] = True
        current_state["emergency_lane"] = emergency_lane
        
    elif green_lane and green_lane in LANES:
        # Normal mode: green lane gets green, others stay red
        set_lane_green(green_lane, with_blue=False)
        
        current_state["green_lane"] = green_lane
        current_state["emergency"] = False
        current_state["emergency_lane"] = None
        
    else:
        # All red
        current_state["green_lane"] = None
        current_state["emergency"] = False
        current_state["emergency_lane"] = None
    
    return jsonify({"status": "OK", "state": current_state})


@app.route("/all_red", methods=["POST"])
def endpoint_all_red():
    """Set all lanes to red"""
    stop_blink()
    all_red()
    current_state["green_lane"] = None
    current_state["emergency"] = False
    current_state["emergency_lane"] = None
    return jsonify({"status": "OK", "message": "All lanes set to red"})


@app.route("/status", methods=["GET"])
def status():
    """Get current signal status"""
    return jsonify({
        "status": "OK",
        "current_state": current_state,
        "lanes": LANE_ORDER
    })


@app.route("/", methods=["GET"])
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "OK",
        "server": "Traffic Signal Controller",
        "version": "2.0"
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  Traffic Signal Server v2.0")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /set_signal  - Set single lane green")
    print("  POST /set_state   - Set full traffic state")
    print("  POST /all_red     - Set all lanes to red")
    print("  GET  /status      - Get current status")
    print("=" * 50)
    
    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        stop_blink()
        GPIO.cleanup()
