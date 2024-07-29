import os
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.transforms import transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# Custom dataset class
class COCODataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        
        num_objs = len(anns)
        boxes = []
        labels = []
        
        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

# Data transforms
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the dataset
dataset = COCODataset(root='data_c/train', 
                      annotation='data_c/train/_annotations.coco.json', 
                      transforms=get_transform())

# Split the dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load a pre-trained model
model = retinanet_resnet50_fpn(pretrained=True)

# Modify the classification head for our number of classes
# num_classes = 6  # 5 classes + background
# in_features = model.head.classification_head.num_anchors * 4
# model.head = RetinaNetHead(in_features, num_classes)
num_anchors = model.head.classification_head.num_anchors

# Get the number of features from the FPN
in_channels = model.backbone.out_channels

# Create a new RetinaNet head
num_classes = 6  # 5 classes + background
model.head = RetinaNetHead(
    in_channels,
    num_anchors,
    num_classes,
    norm_layer=torch.nn.BatchNorm2d  # Add this line
)
# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    # Update the learning rate
    lr_scheduler.step()
    
    # Print the losses
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'retinanet_custom.pth')