from ultralytics import YOLO

# Load your model
model = YOLO('runs/detect/train5/weights/best.pt')

# Run validation
results = model.val(data='data/data.yaml')
# Access metrics
mAP50 = results.box.map50
mAP50_95 = results.box.map
precision = results.box.mp
recall = results.box.recall
# print(results.box)
# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall)

# Print percentages
print(f"mAP50: {mAP50*100:.2f}%")
print(f"mAP50-95: {mAP50_95*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-score: {f1_score*100:.2f}%")