import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# === Paths ===
data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
test_dir = os.path.join(data_dir, "test")

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# === Load model ===
model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Sequential(
    nn.Linear(model.classifier[2].in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(512, len(os.listdir(os.path.join(data_dir, "train"))))
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# === Transforms and Dataset ===
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
class_names = test_dataset.classes

# === Loss Function ===
criterion = nn.CrossEntropyLoss()

# === Evaluation and Plotting ===
def evaluate_and_plot_metrics(model, test_loader, criterion, device, class_names):
    y_true = []
    y_pred = []
    y_scores = []

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(test_loader)
    val_acc = correct / total

    # CLASSIFICATION REPORT
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # CONFUSION MATRIX
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ROC CURVE (only for binary classification)
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, [s[1] for s in y_scores])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig("roc_curve.png")
        plt.close()

    print(f"\nFinal Loss: {val_loss:.4f} | Final Accuracy: {val_acc:.4f}")

# === Run Evaluation ===
evaluate_and_plot_metrics(model, test_loader, criterion, device, class_names)
