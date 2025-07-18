import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.cuda.amp import autocast, GradScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from tqdm import tqdm

# === Mixed Precision Scaler ===
scaler = GradScaler()

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return running_loss / len(test_loader), correct / total

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=10):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        print(f"Train Loss: {running_loss / len(train_loader):.4f} | Accuracy: {correct / total:.4f}")

        # Validate every 3 epochs or on final epoch
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print("Best model saved!")

if __name__ == '__main__':
    data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Sequential(
        nn.Linear(model.classifier[2].in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, len(os.listdir(train_dir)))
    )
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    weights = compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=np.array(train_dataset.targets))
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)

    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=50)


