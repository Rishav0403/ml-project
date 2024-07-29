import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Define class names
CLASS_NAMES = ['background', 'helmet', 'person', 'head', 'no_helmet', 'both']

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def process_prediction(prediction, image, threshold=0.5):
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()

    # Filter out low-scoring boxes
    mask = scores > threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    return boxes, scores, labels

def draw_boxes(image, boxes, scores, labels):
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        class_name = CLASS_NAMES[label]  # Get class name from label index
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), f"{class_name}: {score:.2f}", fill="red")
    return image

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    num_classes = len(CLASS_NAMES)  # Number of classes including background
    model = get_model(num_classes)
    model.load_state_dict(torch.load('faster_rcnn_model', map_location=device))
    model.to(device)
    model.eval()

    # Load and process image
    image_path = "/home/annonymous/Documents/projects/python/helmet_detection/data_c/test/0357_jpg.rf.08688d4a95421c4d3a1bfa09644b0201.jpg"
    image = load_image(image_path)
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Process prediction
    boxes, scores, labels = process_prediction(prediction, image)

    # Draw boxes on image
    result_image = draw_boxes(image, boxes, scores, labels)

    # Save or display the result
    result_image.save("result.jpg")
    result_image.show()

if __name__ == "__main__":
    main()