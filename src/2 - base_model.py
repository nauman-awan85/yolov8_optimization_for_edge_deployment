import os
from ultralytics import YOLO
import torch

model_weights = 'yolov8s.pt'
data_yaml = r'C:/Users/Muhammad Nauman/Desktop/new_thesis/data/data.yaml'
run_dir = r'C:/Users/Muhammad Nauman/Desktop/new_thesis/runs'
if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not available.")
device = 0  
model = YOLO(model_weights)
results = model.train(data=data_yaml, epochs=40, imgsz=640, batch=8, device=device, workers=4, name='base_model_40eps', project=run_dir, save=True, save_period=1, exist_ok=True, patience=0)
val_results = model.val()
print(val_results)
