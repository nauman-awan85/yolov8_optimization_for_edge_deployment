import os
from ultralytics import YOLO
import torch

base_model_path = r"C:/Users/Muhammad Nauman/Desktop/new_thesis/runs/base_model_40eps/weights/best.pt"
data_yaml = r"C:/Users/Muhammad Nauman/Desktop/new_thesis/data/data.yaml"
run_dir = r"C:/Users/Muhammad Nauman/Desktop/new_thesis/runs"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not available.")
device = 0  
model = YOLO(base_model_path)
results = model.train(data=data_yaml, epochs=50, imgsz=640, batch=8, device=device, optimizer="SGD",lr0=0.0015, lrf=0.01, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.3, translate=0.2, workers=4, name="fine_tune_50eps",project=run_dir, save=True, exist_ok=True, patience=0)
val_results = model.val()
print(val_results)
