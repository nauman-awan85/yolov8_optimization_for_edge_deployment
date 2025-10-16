import os
import json
from tqdm import tqdm
COCO_JSON = r"C:\Users\Muhammad Nauman\Desktop\new_thesis\data\annotations\annotations.json"

IMAGE_DIRS = {
    "train": r"C:\Users\Muhammad Nauman\Desktop\new_thesis\data\images\train",
    "val": r"C:\Users\Muhammad Nauman\Desktop\new_thesis\data\images\val"
}
LABEL_DIRS = {
    "train": r"C:\Users\Muhammad Nauman\Desktop\new_thesis\data\labels\train",
    "val": r"C:\Users\Muhammad Nauman\Desktop\new_thesis\data\labels\val"
}
MAX_CATEGORY_ID = 40 
with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)
images = {img["id"]: img for img in coco["images"]}
annotations = coco["annotations"]
split_lookup = {}
for split, folder in IMAGE_DIRS.items():
    image_files = set(os.listdir(folder))
    for img_id, img in images.items():
        if img["file_name"] in image_files:
            split_lookup[img_id] = split
labels_per_image = {}
invalid_categories = set()
for ann in annotations:
    img_id = ann["image_id"]
    if img_id not in images:
        continue
    category_id = ann["category_id"]
    if category_id > MAX_CATEGORY_ID:
        invalid_categories.add(category_id)
        continue
    class_id = category_id - 1
    img_data = images[img_id]
    img_w, img_h = img_data.get("width"), img_data.get("height")
    if not img_w or not img_h:
        continue
    x_min, y_min, width, height = ann["bbox"]
    x_center = (x_min + width / 2) / img_w
    y_center = (y_min + height / 2) / img_h
    norm_width = width / img_w
    norm_height = height / img_h
    line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
    labels_per_image.setdefault(img_id, []).append(line)
for img_id, label_lines in tqdm(labels_per_image.items(), desc="Generating YOLO labels"):
    split = split_lookup.get(img_id)
    if not split:
        continue
    img_file = images[img_id]["file_name"]
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(LABEL_DIRS[split], label_file)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))
if invalid_categories:
    print(f"Skipped annotations: {sorted(invalid_categories)}")
print("Completed.")
