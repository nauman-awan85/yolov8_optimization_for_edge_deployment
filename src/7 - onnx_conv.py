import torch
from pathlib import Path
from ultralytics import YOLO

model_path = Path(r"C:\Users\Muhammad Nauman\Desktop\new_thesis\final\pruned.pt")
state_path = Path(r"C:\Users\Muhammad Nauman\Desktop\new_thesis\final\distilled.pt")
output_path = os.path.join(r"C:\Users\Muhammad Nauman\Desktop\new_thesis\final","distilled_static.onnx")

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO(str(model_path)).to(device)
sd = torch.load(str(state_path), map_location="cpu")
yolo.model.load_state_dict(sd, strict=False)
yolo.fuse()
yolo.model.eval().float()
exported = yolo.export(format="onnx",imgsz=512, opset=12, dynamic=False, simplify=True, half=False, batch=1, device=device, verbose=False)
exported_path = Path(exported if isinstance(exported, str) else exported[0])
if exported_path.resolve() != out_onnx.resolve():
    out_onnx.write_bytes(exported_path.read_bytes())
print(f"ONNX: {out_onnx}")
