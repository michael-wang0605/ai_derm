import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def main():
    # === Paths ===
    test_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split\\test"
    model_path = "best_model.pth"

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Data Transform ===
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # === Dataset & Loader ===
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # === Load Model Architecture ===
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Sequential(
        nn.Linear(model.classifier[2].in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, len(test_dataset.classes))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # === Get Predictions ===
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Compute F1 Scores ===
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # === Save to CSV ===
    df = pd.DataFrame({
        'Metric': ['F1 Macro', 'F1 Micro', 'F1 Weighted'],
        'Score': [f1_macro, f1_micro, f1_weighted]
    })

    for i, score in enumerate(f1_per_class):
        class_name = test_dataset.classes[i]
        df = pd.concat([df, pd.DataFrame({'Metric': [f'F1 Class {class_name}'], 'Score': [score]})], ignore_index=True)

    df.to_csv("f1_scores.csv", index=False)
    print("✅ Saved F1 scores to f1_scores.csv")

# === Windows Multiprocessing Guard ===
if __name__ == '__main__':
    main()
