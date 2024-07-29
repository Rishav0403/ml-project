import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import os
import seaborn as sns
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)  # Assuming input size is 224x224

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Custom dataset class for COCO format
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.root_dir, img_info['file_name'])).convert('RGB')
        
        # Assume we're doing classification based on the first category
        category_id = anns[0]['category_id']
        
        if self.transform:
            image = self.transform(image)
        
        return image, category_id

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the validation dataset
val_dataset = CocoDataset(root_dir='data_c/valid', 
                          annotation_file='data_c/valid/_annotations.coco.json', 
                          transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the trained model
num_classes = len(val_dataset.coco.cats)
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load('coco_cnn_model.pth'))
model.eval()

# Lists to store predictions and true labels
all_predictions = []
all_labels = []

# Evaluation loop
with torch.no_grad():
    for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Calculate and print per-class metrics
class_names = list(val_dataset.coco.cats.values())
for i, class_name in enumerate(class_names):
    class_precision = precision_score(all_labels, all_predictions, labels=[i], average='macro', zero_division=0)
    class_recall = recall_score(all_labels, all_predictions, labels=[i], average='macro', zero_division=0)
    class_f1 = f1_score(all_labels, all_predictions, labels=[i], average='macro', zero_division=0)
    print(f"\nMetrics for class '{class_name['name']}':")
    print(f"Precision: {class_precision:.4f}")
    print(f"Recall: {class_recall:.4f}")
    print(f"F1-score: {class_f1:.4f}")

# Calculate and print confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[cat['name'] for cat in class_names], 
            yticklabels=[cat['name'] for cat in class_names])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("\nConfusion matrix saved as 'confusion_matrix.png'")
