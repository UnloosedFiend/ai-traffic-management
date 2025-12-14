# src/desktop/train_yolo.py
# Minimal trainer wrapper for Ultralytics YOLOv8
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='dataset/data.yaml')
parser.add_argument('--model', default='yolov8n.pt')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--project', default='runs/train')
parser.add_argument('--name', default='exp')
args = parser.parse_args()

model = YOLO(args.model)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, project=args.project, name=args.name, exist_ok=True)
print('Training finished. Check', args.project, '/', args.name)
