import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score
)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def main():
    # === Paths ===
    model_path = "best_model.pth"
    test_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split\\test"

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Data ===
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # === Model ===
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[2].in_features, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(512, len(test_dataset.classes))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # === Predictions ===
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # === Metrics ===
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    specificity = TN / (TN + FP + 1e-8)

    # === Compile Report ===
    report = {
        "F1 Macro": f1_macro,
        "F1 Micro": f1_micro,
        "F1 Weighted": f1_weighted,
        "Precision Macro": precision_macro,
        "Recall Macro": recall_macro,
        "Balanced Accuracy": balanced_acc,
        "Cohen Kappa": kappa,
    }

    for i, class_name in enumerate(test_dataset.classes):
        report[f"F1 Class {class_name}"] = f1_per_class[i]
        report[f"Precision Class {class_name}"] = precision_per_class[i]
        report[f"Recall Class {class_name}"] = recall_per_class[i]
        report[f"Specificity Class {class_name}"] = specificity[i]

    # === Save CSV ===
    df = pd.DataFrame(list(report.items()), columns=["Metric", "Score"])
    df.to_csv("full_metrics_report.csv", index=False)
    print("✅ Saved metrics report to full_metrics_report.csv")

if __name__ == "__main__":
    main()
