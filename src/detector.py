import cv2
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)
        boxes = []

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                boxes.append((x1, y1, x2, y2))

        return boxes
