from ultralytics import YOLO
import torch
import os

torch.cuda.empty_cache()

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

print(torch.cuda.is_available())
torch.device("cuda")
model = YOLO("yolo/yolo-weights/yolov8l.pt")
model.train(data="data/data.yaml", imgsz=320, batch=1, epochs=20, workers=0)

# results = model.predict(source='/content/data/val/images', save=True)
