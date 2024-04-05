from ultralytics import YOLO

# Create a new YOLOv8n-OBB model from scratch
model = YOLO('yolov8n-obb.yaml')

# Train the model on the DOTAv2 dataset
results = model.train(data='DOTAv1.yaml', epochs=100, imgsz=640)
