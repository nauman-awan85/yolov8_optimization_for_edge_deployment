import os, glob, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

class ImageFolderFlat(Dataset):
    def __init__(self, root_dir, image_size):
        extensions = ('*.jpg', '*.jpeg')
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
        random.shuffle(self.image_paths)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

teacher_path = r"C:\Users\Muhammad Nauman\Desktop\new_thesis\models\new model\finetune.pt"
student_path = r"C:\Users\Muhammad Nauman\Desktop\new_thesis\models\new model\pruned.pt"
train_dir = r"C:\Users\Muhammad Nauman\Desktop\new_thesis\data\images\train"
output_path = os.path.join(r"C:\Users\Muhammad Nauman\Desktop\new_thesis\models\new model","distilled_model.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = (device.type == "cuda")
torch.backends.cudnn.benchmark = True

teacher = YOLO(teacher_path).model.to(device).eval()
student = YOLO(student_path).model.to(device).train()

for p in student.parameters():
    p.requires_grad_(True)
for p in teacher.parameters():
    p.requires_grad_(False)

conv_masks = {
    n: (m.weight.data != 0).to(m.weight.dtype).to(device)
    for n, m in student.named_modules()
    if isinstance(m, nn.Conv2d) and m.weight is not None
}

def reapply_masks():
    for n, m in student.named_modules():
        if isinstance(m, nn.Conv2d) and m.weight is not None and n in conv_masks:
            m.weight.mul_(conv_masks[n])

def sparsity(model):
    z = t = 0
    with torch.no_grad():
        for p in model.parameters():
            t += p.numel()
            z += (p == 0).sum().item()
    return 100.0 * z / max(1, t)

reapply_masks()
print(f"sparsity start: {sparsity(student):.2f}%")

tc = [m for m in teacher.modules() if isinstance(m, nn.Conv2d)]
sc = [m for m in student.modules() if isinstance(m, nn.Conv2d)]
L = min(len(tc), len(sc))
idxs = sorted({max(0, int(L * r)) for r in (0.3, 0.6, 0.9)})
print("Hook indices:", idxs)

t_feats, s_feats = [], []
def thook(_, __, o): t_feats.append(o.detach())
def shook(_, __, o): s_feats.append(o)

for i in idxs:
    tc[i].register_forward_hook(thook)
    sc[i].register_forward_hook(shook)

ds = ImageFolderFlat(train_dir, 512)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2,
                    pin_memory=(device.type == "cuda"), persistent_workers=True)

mse = nn.MSELoss()
opt = optim.AdamW(student.parameters(), lr=5e-4, weight_decay=5e-4)
sched = CosineAnnealingLR(opt, T_max=12, eta_min=5e-5)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

best, noimp = float('inf'), 0
print("Distill for 12 epochs")

for ep in range(1, 13):
    student.train()
    total = cnt = processed = 0
    for imgs in tqdm(loader, desc=f"Epoch {ep}/12"):
        processed += imgs.size(0)
        imgs = imgs.to(device)

        t_feats.clear()
        s_feats.clear()
        with torch.inference_mode():
            teacher(imgs)
        with torch.cuda.amp.autocast(enabled=use_amp):
            student(imgs)
            if not t_feats or not s_feats:
                continue
            loss = sum(mse(sf, torch.nn.functional.normalize(tf, dim=1))
                       for tf, sf in zip(t_feats, s_feats)) / len(t_feats)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        reapply_masks()

        total += loss.item()
        cnt += 1
        tqdm.write(f"loss {total / cnt:.4f}", end='\r')

    sched.step()
    avg = total / max(1, cnt)
    print(f"Epoch {ep}: avg_loss={avg:.6f}, lr={sched.get_last_lr()[0]:.2e}, sparsity={sparsity(student):.2f}%")
    if avg < best:
        best, noimp = avg, 0
        torch.save({k: v.float().cpu() for k, v in student.state_dict().items()}, output_path)
    else:
        noimp += 1
        if noimp >= 3:
            print("Early stopping.")
            break

reapply_masks()
print(f"sparsity end: {sparsity(student):.2f}%")

if os.path.isfile(output_path):
    print(f"Saved distilled model: {output_path}")
