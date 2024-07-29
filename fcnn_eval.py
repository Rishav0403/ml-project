import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import json
from tqdm import tqdm

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def calculate_iou(box1, box2):
    # Calculate intersection over union
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def evaluate_model(model, test_dir, annotation_file, device, iou_threshold=0.5, conf_threshold=0.5):
    model.eval()
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    total_gt = 0
    total_pred = 0
    correct_pred = 0

    for img_info in tqdm(annotations['images']):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(test_dir, img_filename)
        
        # Load and process image
        image = load_image(img_path)
        image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).to(device)

        # Get ground truth annotations for this image
        gt_boxes = []
        gt_labels = []
        for ann in annotations['annotations']:
            if ann['image_id'] == img_id:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x+w, y+h])
                gt_labels.append(ann['category_id'])
        
        total_gt += len(gt_boxes)

        # Run inference
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        # Process prediction
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()

        # Filter predictions based on confidence threshold
        mask = pred_scores > conf_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        total_pred += len(pred_boxes)

        # Compare predictions with ground truth
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            for pred_box, pred_label in zip(pred_boxes, pred_labels):
                if calculate_iou(gt_box, pred_box) > iou_threshold and gt_label == pred_label:
                    correct_pred += 1
                    break

    precision = correct_pred / total_pred if total_pred > 0 else 0
    recall = correct_pred / total_gt if total_gt > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    num_classes = 6  # 5 classes + background
    model = get_model(num_classes)
    model.load_state_dict(torch.load('faster_rcnn_model', map_location=device))
    model.to(device)

    # Paths
    test_dir = "/home/annonymous/Documents/projects/python/helmet_detection/data_c/test"
    annotation_file = "/home/annonymous/Documents/projects/python/helmet_detection/data_c/test/_annotations.coco.json"

    # Evaluate model
    results = evaluate_model(model, test_dir, annotation_file, device)

    # Print results
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()