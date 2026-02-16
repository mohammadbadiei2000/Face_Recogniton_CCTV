import cv2
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.utils import draw_boxes

def main():
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    image = cv2.imread("data/test.jpg")

    boxes = detector.detect(image)

    labels = []

    for (x1, y1, x2, y2) in boxes:
        face_crop = image[y1:y2, x1:x2]
        embedding = recognizer.get_embedding(face_crop)

        if embedding is not None:
            name, score = recognizer.recognize(embedding)
            labels.append(f"{name} ({score:.2f})")
        else:
            labels.append("No Face")

    output = draw_boxes(image, boxes, labels)

    cv2.imshow("Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
