# This script will run inference on a given image using a pre-trained YOLOv5 model.
import torch
import cv2
from scripts.utils import draw_bboxes, show_image

def run_inference(weights_path, image_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    img = cv2.imread(image_path)

    results = model(img)
    detections = results.xyxy[0].cpu().numpy()

    bboxes = []
    labels = []
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        bboxes.append([x1, y1, x2, y2])
        labels.append(model.names[int(cls)] + f' {conf:.2f}')

    img_with_bboxes = draw_bboxes(img, bboxes, labels)
    show_image(img_with_bboxes)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/traffic_detection/best.pt')
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    run_inference(args.weights, args.image)
