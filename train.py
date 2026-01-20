from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # pretrained base

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        name="traffic_v1",
        project="runs/detect"
    )

if __name__ == "__main__":
    main()
