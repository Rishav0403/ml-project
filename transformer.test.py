import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=5, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load('vit_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_and_draw_bboxes(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image = input_image.to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(input_image).logits
    predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for label in predicted_labels:
        bbox = [50, 50, 200, 200]
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], bbox[1] - 10), f'Class: {label}', fill="red", font=font)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

image_path = 'data_c/test/0024_jpg.rf.50fce639ea2d13798db45d2205384978.jpg'
predict_and_draw_bboxes(image_path, model, transform)
