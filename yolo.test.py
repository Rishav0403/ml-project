from ultralytics import YOLO
import cv2
import numpy as np

def yoloTest(imgPath):
    model = YOLO(imgPath)  
    image_path = 'data/train/images/0006_jpg.rf.d39364141e90b456b6bcd6acfaad1e6e.jpg'

    image = cv2.imread(image_path)

    results = model(image)

    with open('detection_results.txt', 'w') as f:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                class_name = model.names[cls]
                
                f.write(f"{class_name} {conf:.2f} {x1} {y1} {x2} {y2}\n")
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite('yolo_output_image.jpg', image)


def main():
    yoloTest('runs/detect/train5/weights/best.pt')


if __name__ == "__main__":
    main()