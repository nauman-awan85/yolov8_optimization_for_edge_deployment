import os
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

MODEL_PATH = r"C:\Users\Muhammad Nauman\Desktop\new_thesis\models\finetune.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = YOLO(MODEL_PATH)
model.model.to(DEVICE).eval()

all_weights = []
layerwise = []

with torch.no_grad():
    for name, module in model.model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
            w = module.weight.data
            if w.numel() == 0:
                continue
            abs_w = w.abs().detach().cpu().flatten()
            all_weights.append(abs_w)
            num_params = w.numel()
            num_zeros = (w == 0).sum().item()
            layerwise.append((name, num_params, num_zeros))

if not all_weights:
    raise RuntimeError("No Conv2d weights found.")

flat_weights = torch.cat(all_weights)
np_weights = flat_weights.numpy()

total = np_weights.size
zeros = int((np_weights == 0).sum())
sparsity = zeros * 100.0 / total
mean_val = np_weights.mean()
std_val = np_weights.std()

percentiles = [5, 10, 15, 20, 25, 30, 40, 50]

print(f"Total weights: {total:,}")
print(f"Zeros: {zeros:,} | Sparsity: {sparsity:.2f}%")
print(f"Mean abs(weight): {mean_val:.6f} | Std: {std_val:.5f}\n")

print(f"{'Percentile':>12} | {'Threshold':>12} | {'Count ≤ Thr.':>15} | {'% of Total':>10}")
print("-" * 60)
for p in percentiles:
    thr = np.percentile(np_weights, p)
    cnt = int((np_weights <= thr).sum())
    pct = cnt * 100.0 / total
    print(f"{p:10.1f}% | {thr:12.6f} | {cnt:15,} | {pct:9.2f}%")

if PRINT_LAYERWISE:
    print("\nLayer wise parameter sparsity:")
    print(f"{'Layer':<48} {'Params':>10} {'Zeros':>10} {'Sparsity%':>10}")
    print("-" * 80)
    for name, n, z in layerwise:
        sp = z * 100.0 / n if n else 0.0
        print(f"{name:<48} {n:10,} {z:10,} {sp:10.2f}")

plt.figure(figsize=(10, 5))
plt.hist(np_weights, bins=100, log=True)
plt.title(f"Weight Distribution in Finetuned Model")
plt.xlabel("Absolute Weight Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
