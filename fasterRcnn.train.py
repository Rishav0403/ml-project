import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torch.optim import SGD
from PIL import Image
import os

torch.cuda.empty_cache()


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Available GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        
        num_objs = len(coco_annotation)

        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# Data transforms
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

# Create the model
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Training function
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Loss: {losses.item()}")

# Main training loop
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Dataset paths
    train_data_dir = "/home/annonymous/Documents/projects/python/helmet_detection/data_c/train"
    train_coco_annotation = "/home/annonymous/Documents/projects/python/helmet_detection/data_c/train/_annotations.coco.json"
    
    
    # Create dataset and dataloader
    dataset = COCODataset(root=train_data_dir, annotation=train_coco_annotation, transforms=get_transform())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Number of classes (including background)
    num_classes = 6  # 5 classes + background

    # Get the model
    model = get_model(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, data_loader, device)

    # Save the model
    torch.save(model.state_dict(), 'faster_rcnn_model')

if __name__ == "__main__":
    main()