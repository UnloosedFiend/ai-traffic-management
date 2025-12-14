# src/pi/app.py
from flask import Flask, Response, render_template, request, jsonify
import cv2, threading, time, os
from detect_tflite import TFLiteDetector
from gpio_ctrl import GPIOController

MODEL_PATH = os.getenv('MODEL_PATH','src/pi/model/best.tflite')
CAM_ID = int(os.getenv('CAM_ID','0'))
INPUT_SIZE = int(os.getenv('INPUT_SIZE','320'))

detector = TFLiteDetector(MODEL_PATH, input_size=INPUT_SIZE)
PIN_MAP = {'lane_red':17, 'lane_yellow':27, 'lane_green':22}
gpio = GPIOController(PIN_MAP)

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
    state = data.get('state','red')
    gpio.set('lane_green', state=='green')
    gpio.set('lane_yellow', state=='yellow')
    gpio.set('lane_red', state=='red')
    return jsonify({'ok':True})

@app.route('/api/detections')
def get_detections():
    return jsonify(global_detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
