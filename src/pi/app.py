# src/pi/app.py
from flask import Flask, Response, render_template, request, jsonify
import cv2, threading, time, os

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
except ImportError:
    GPIO = None

from detect_tflite import TFLiteDetector

MODEL_PATH = os.getenv('MODEL_PATH','src/pi/model/best.tflite')
CAM_ID = int(os.getenv('CAM_ID','0'))
INPUT_SIZE = int(os.getenv('INPUT_SIZE','320'))

detector = TFLiteDetector(MODEL_PATH, input_size=INPUT_SIZE)

# Pin mapping from signal_server_v2.py (authoritative source)
LANES = {
    "NORTH": {"R": 17, "Y": 27, "G": 22, "B": 23},
    "EAST":  {"R": 5,  "Y": 6,  "G": 13, "B": 19},
    "SOUTH": {"R": 12, "Y": 16, "G": 20, "B": 21},
    "WEST":  {"R": 24, "Y": 25, "G": 18, "B": 26},
}
LANE_ORDER = ["NORTH", "EAST", "SOUTH", "WEST"]

# Initialize GPIO pins
if GPIO:
    for lane in LANES.values():
        for pin in lane.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)


def gpio_set(pin, value):
    """Set a GPIO pin HIGH or LOW."""
    if GPIO:
        GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)

app = Flask(__name__)
video_lock = threading.Lock()
global_frame = None
global_detections = []

def camera_loop():
    global global_frame, global_detections
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print('Camera open failed for id', CAM_ID)
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        detections, inf_time = detector.infer(frame)
        disp = frame.copy()
        h, w = frame.shape[:2]
        # draw detections and convert normalized coords if needed
        for d in detections:
            x1,y1,x2,y2 = d['box']
            # if coordinates look normalized (<=1), scale them
            if max(x1,x2,y1,y2) <= 1.01:
                x1,y1,x2,y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            else:
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            cls = d.get('class', -1)
            score = d.get('score', 0.0)
            cv2.rectangle(disp, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(disp, f'{cls}:{score:.2f}', (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.putText(disp, f'Inf: {inf_time*1000:.1f}ms', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        with video_lock:
            global_frame = disp
            global_detections = detections

t = threading.Thread(target=camera_loop, daemon=True)
t.start()

def gen_frames():
    global global_frame
    while True:
        with video_lock:
            if global_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', global_frame)
            frame = jpeg.tobytes()
        yield (b'--frame\\r\\n' b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/set_light', methods=['POST'])
def set_light():
    data = request.json or {}
    state = data.get('state', 'red')
    lane_name = data.get('lane', 'NORTH').upper()
    if lane_name not in LANES:
        return jsonify({'ok': False, 'error': f'Invalid lane: {lane_name}'}), 400
    lane = LANES[lane_name]
    gpio_set(lane['G'], state == 'green')
    gpio_set(lane['Y'], state == 'yellow')
    gpio_set(lane['R'], state == 'red')
    gpio_set(lane['B'], False)
    return jsonify({'ok': True})

@app.route('/set_signal', methods=['POST'])
def set_signal():
    """
    Set traffic signal for a specific lane.
    
    Expected JSON payload:
        {
            "lane": "NORTH" | "EAST" | "SOUTH" | "WEST",
            "emergency": true | false
        }
    
    Returns:
        JSON response with status
    """
    data = request.json or {}
    lane = data.get('lane', 'NORTH').upper()
    emergency = data.get('emergency', False)
    
    if lane not in LANES:
        return jsonify({'ok': False, 'error': f'Invalid lane: {lane}'}), 400
    
    # Set all lanes to red first
    for name in LANE_ORDER:
        l = LANES[name]
        gpio_set(l['G'], False)
        gpio_set(l['Y'], False)
        gpio_set(l['B'], False)
        gpio_set(l['R'], True)
    
    # Set the requested lane to green
    target = LANES[lane]
    gpio_set(target['R'], False)
    gpio_set(target['G'], True)
    if emergency:
        gpio_set(target['B'], True)
        for name in LANE_ORDER:
            if name != lane:
                gpio_set(LANES[name]['B'], True)
    
    return jsonify({
        'ok': True,
        'lane': lane,
        'emergency': emergency
    })

@app.route('/api/detections')
def get_detections():
    return jsonify(global_detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
