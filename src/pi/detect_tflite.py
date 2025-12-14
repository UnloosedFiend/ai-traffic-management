# src/pi/detect_tflite.py
# Lightweight TFLite wrapper with flexible output parsing and simple NMS.
import numpy as np
import cv2
import time

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

def non_max_suppression(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

class TFLiteDetector:
    def __init__(self, model_path: str, input_size: int = 320, score_thresh: float = 0.3, iou_thresh: float = 0.45):
        self.model_path = model_path
        self.input_size = input_size
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # debug print shapes
        # print('INPUT', self.input_details)
        # print('OUTPUT', self.output_details)

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0).astype(np.float32)

    def infer(self, frame):
        inp = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        t0 = time.time()
        self.interpreter.invoke()
        t1 = time.time()
        outputs = [self.interpreter.get_tensor(o['index']) for o in self.output_details]

        # Flexible parsing: handle common Ultralytics TFLite formats:
        # Case A: single output shape (1, N, 6) where each row is (x1,y1,x2,y2,score,class)
        # Case B: three outputs: boxes (1,N,4), scores (1,N), classes (1,N)
        detections = []
        try:
            if len(outputs) == 1:
                out = outputs[0]
                if out.ndim == 3 and out.shape[0] == 1 and out.shape[2] >= 6:
                    arr = out[0]  # (N, >=6)
                    # ensure we have 6 columns: x1,y1,x2,y2,score,class
                    for row in arr:
                        score = float(row[4])
                        if score < self.score_thresh:
                            continue
                        x1,y1,x2,y2 = map(float, row[0:4])
                        cls = int(row[5])
                        detections.append({'box':[x1,y1,x2,y2], 'score':score, 'class':cls})
                else:
                    # unknown single-output layout: attempt to flatten rows of 6
                    flat = out.reshape(-1, out.shape[-1])
                    for row in flat:
                        if row.size >= 6:
                            score = float(row[4])
                            if score < self.score_thresh: continue
                            x1,y1,x2,y2 = map(float, row[0:4])
                            cls = int(row[5])
                            detections.append({'box':[x1,y1,x2,y2], 'score':score, 'class':cls})
            elif len(outputs) >= 2:
                # try to find boxes, scores and classes heuristically
                boxes = None; scores = None; classes = None
                for out in outputs:
                    if out.ndim == 3 and out.shape[2] == 4:
                        boxes = out[0]
                    elif out.ndim == 2 and out.shape[1] >= 1:
                        # maybe (1,N) scores or (1,N,1)
                        a = out.reshape(-1)
                        if scores is None:
                            scores = a
                        elif classes is None:
                            classes = a
                if boxes is None:
                    # fallback: flatten first output as boxes
                    boxes = outputs[0].reshape(-1,4)
                if scores is None:
                    # fallback: use 0.5 for all
                    scores = np.ones(boxes.shape[0]) * 0.5
                if classes is None:
                    classes = np.zeros(boxes.shape[0], dtype=np.int32)
                for i in range(boxes.shape[0]):
                    score = float(scores[i])
                    if score < self.score_thresh: continue
                    x1,y1,x2,y2 = boxes[i].astype(float).tolist()
                    cls = int(classes[i])
                    detections.append({'box':[x1,y1,x2,y2], 'score':score, 'class':cls})
        except Exception as e:
            # if parsing fails, return empty with timing info
            print('Parsing error in TFLite outputs:', e)

        # Convert box coords from model space (if normalized 0..1) to pixel later in app using frame shape.
        # Apply NMS in model coordinate space
        if len(detections) > 0:
            boxes = np.array([d['box'] for d in detections], dtype=np.float32)
            scores = np.array([d['score'] for d in detections], dtype=np.float32)
            keep = non_max_suppression(boxes, scores, self.iou_thresh)
            detections = [detections[i] for i in keep]

        return detections, (t1 - t0)
