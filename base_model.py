import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from tqdm import tqdm

# === Scaler for mixed precision ===
scaler = GradScaler()

# === TTA Evaluation ===
def evaluate_model(model, test_loader, criterion, device, tta_times=5, tta_transform=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating with TTA", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if tta_transform:
                outputs = []
                for _ in range(tta_times):
                    to_pil = transforms.ToPILImage()
                    aug_inputs = torch.stack([tta_transform(to_pil(img.cpu())) for img in inputs])
                    aug_inputs = aug_inputs.to(device)
                    output = model(aug_inputs)
                    outputs.append(torch.softmax(output, dim=1))
                outputs = torch.stack(outputs).mean(dim=0)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(test_loader), correct / total

# === Training Loop ===
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, tta_transform, epochs=50):
    best_val_acc = 0.0

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
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Train Loss: {running_loss / len(train_loader):.4f} | Train Acc: {correct / total:.4f}")

        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device,
                                           tta_times=5, tta_transform=tta_transform)
        print(f"Val Loss: {val_loss:.4f} | Val Acc (TTA): {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!")

# === MAIN ===
if __name__ == '__main__':
    data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("✅ GPU Detected:", torch.cuda.get_device_name(0))

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
    model = model.to(device).to(memory_format=torch.channels_last)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    tta_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(train_dataset.targets),
                                         y=np.array(train_dataset.targets))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=2, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                             num_workers=6, pin_memory=True)

    train_model(model, train_loader, test_loader, criterion, optimizer,
                scheduler, device, tta_transform=tta_transform, epochs=50)
