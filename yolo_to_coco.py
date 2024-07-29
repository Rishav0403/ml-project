import os
import json
from PIL import Image
import yaml

def yolo_to_coco(data_yaml_path, output_file):
    # Load data.yaml
    with open(data_yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Use the categories from data.yaml
    categories = data_yaml['names']
    for i, category in enumerate(categories):
        coco_format["categories"].append({"id": i, "name": category})
    
    annotation_id = 1
    image_id = 1

    # Process train, val, and test sets
    for subset in ['train', 'val', 'test']:
        image_dir = data_yaml[subset]
        label_dir = os.path.join(os.path.dirname(image_dir), 'labels')

        for img_file in os.listdir(image_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(image_dir, img_file)
                img = Image.open(img_path)
                width, height = img.size
                
                coco_format["images"].append({
                    "id": image_id,
                    "file_name": img_file,
                    "width": width,
                    "height": height
                })
                
                # Corresponding YOLO annotation file
                txt_file = os.path.splitext(img_file)[0] + '.txt'
                txt_path = os.path.join(label_dir, txt_file)
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        for line in f.readlines():
                            try:
                                class_id, x1, y1, x2, y2 = map(float, line.strip().split())
                                
                                # Convert to COCO format (x, y, width, height)
                                x = min(x1, x2)
                                y = min(y1, y2)
                                w = abs(x2 - x1)
                                h = abs(y2 - y1)
                                
                                coco_format["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": int(class_id),
                                    "bbox": [x, y, w, h],
                                    "area": w * h,
                                    "iscrowd": 0
                                })
                                annotation_id += 1
                            except ValueError as e:
                                print(f"Error processing line in {txt_file}: {line.strip()}")
                                print(f"Error message: {str(e)}")
                
                image_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_format, f)

# Usage
yolo_to_coco('/home/annonymous/Documents/projects/python/helmet_detection/data/data.yaml', '/home/annonymous/Documents/projects/python/helmet_detection/data/output_coco.json')