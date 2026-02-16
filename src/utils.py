import cv2

def draw_boxes(image, boxes, labels=None):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if labels:
            cv2.putText(
                image,
                labels[i],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    return image
