# This script will evaluate the trained YOLOv5 model on the valiidation dataset.
from yolov5 import val

def evaluate():
    val.run(
        data='data/coco.yaml',
        weights='models/traffic_detection/best.pt',
        batch=16,
        imgsz=640,
    )

if __name__ == "__main__":
    evaluate()
