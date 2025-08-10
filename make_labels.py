# make_labels.py
import json, os
from torchvision.datasets import ImageFolder
train_dir = r"C:\Users\mwang\ai_derm\dataset_categorized_final_split\train"
ds = ImageFolder(train_dir)
# ImageFolder maps class names -> indices in ds.class_to_idx (alphabetical by folder name)
idx2label = {v: k for k, v in ds.class_to_idx.items()}
labels = [idx2label[i] for i in range(len(idx2label))]
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)
print("Wrote labels.json with", len(labels), "classes")
