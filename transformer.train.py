# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from transformers import ViTForImageClassification, ViTFeatureExtractor
# from pycocotools.coco import COCO
# import os
# from PIL import Image

# # Custom Dataset for COCO
# class CocoDataset(Dataset):
#     def __init__(self, annotation_file, img_dir, transform=None):
#         self.coco = COCO(annotation_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.ids = list(self.coco.imgs.keys())

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         img_id = self.ids[idx]
#         img_info = self.coco.imgs[img_id]
#         path = os.path.join(self.img_dir, img_info['file_name'])
#         image = Image.open(path).convert("RGB")
        
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         anns = self.coco.loadAnns(ann_ids)
        
#         labels = [ann['category_id'] for ann in anns]
#         labels = torch.tensor(labels).unique()  # Ensure unique labels
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, labels

# # Custom collate function to handle variable-length labels
# def collate_fn(batch):
#     images, labels = zip(*batch)
#     images = torch.stack(images, dim=0)
#     # Pad labels to the maximum length in the batch
#     max_len = max(len(label) for label in labels)
#     padded_labels = torch.zeros((len(labels), max_len), dtype=torch.long)
#     for i, label in enumerate(labels):
#         padded_labels[i, :len(label)] = label
#     return images, padded_labels

# # Transformations and Data Loading
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# train_dataset = CocoDataset(annotation_file='data_c/train/_annotations.coco.json', img_dir='data_c/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# # Define the Model
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=5, ignore_mismatched_sizes=True)  # Adjust num_labels

# # Training Setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Training Loop
# num_epochs = 50

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(images).logits
        
#         # Compute loss considering multi-label classification
#         loss = sum(criterion(outputs, labels[:, i]) for i in range(labels.size(1)))
        
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# # Save the model
# torch.save(model.state_dict(), 'vit_model.pth')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification
from pycocotools.coco import COCO
import os
from PIL import Image

# Custom Dataset for COCO
class CocoDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(path).convert("RGB")
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        labels = [ann['category_id'] for ann in anns]
        labels = torch.tensor(labels).unique()  # Ensure unique labels
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels

# Custom collate function to handle variable-length labels
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    # Pad labels to the maximum length in the batch
    max_len = max(len(label) for label in labels)
    padded_labels = torch.zeros((len(labels), max_len), dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    return images, padded_labels

# Transformations and Data Loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CocoDataset(annotation_file='data_c/train/_annotations.coco.json', img_dir='data_c/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Define the Model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=5, ignore_mismatched_sizes=True)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Reduced learning rate

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).logits
        
        loss = 0
        for i in range(labels.size(1)):
            loss += criterion(outputs, labels[:, i])
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), 'vit_model.pth')
