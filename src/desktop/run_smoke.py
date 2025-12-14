# src/desktop/run_smoke.py
# Downloads a sample image and runs a single inference with YOLOv8n to verify setup.
from ultralytics import YOLO
import os, urllib.request, sys

img = 'dataset/images/test/sample.jpg'
os.makedirs(os.path.dirname(img), exist_ok=True)
if not os.path.exists(img):
    print('Downloading sample image...')
    urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', img)

print('Using image:', img)
model = YOLO('yolov8n.pt')   # downloads automatically if missing
results = model.predict(img, imgsz=640, save=True)
print('Prediction completed. Saved images in runs/predict or runs/detect.')
