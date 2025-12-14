# src/desktop/export_model.py
# Export a trained Ultralytics weights file to ONNX and TFLite
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='runs/train/exp/weights/best.pt')
parser.add_argument('--imgsz', type=int, default=320)
args = parser.parse_args()

model = YOLO(args.weights)
print('Exporting to ONNX...')
model.export(format='onnx', imgsz=args.imgsz, device='cpu')
print('Exporting to TFLite...')
model.export(format='tflite', imgsz=args.imgsz, device='cpu')
print('Export complete. Move the tflite to src/pi/model/')
