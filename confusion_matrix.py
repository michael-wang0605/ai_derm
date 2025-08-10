import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # === Paths ===
    model_path = "best_model.pth"
    test_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split\\test"

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Transform & Dataset ===
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # === Load Model ===
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
    model.to(device).eval()

    # === Gather Predictions & Labels ===
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Generating Confusion Matrix"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Generate Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    class_names = test_dataset.classes

    # Save as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv("confusion_matrix.csv")
    print("✅ Saved confusion matrix to confusion_matrix.csv")

    # Save as Heatmap PNG
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("📸 Saved heatmap to confusion_matrix.png")

if __name__ == '__main__':
    main()
