import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import RandAugment
from tqdm import tqdm  # For progress bars

# Initialize Gradient Scaler for Mixed Precision
scaler = GradScaler()

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_loss = running_loss / len(test_loader)
    val_acc = correct / total
    return val_loss, val_acc

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=10):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{epochs} starting...")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

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
        print(f"CUDA is selected. Using GPU: {torch.cuda.get_device_name(0)}")

    # Load ConvNeXt-Tiny model
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Sequential(
        nn.Linear(model.classifier[2].in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),  # Reduced dropout
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Reduced dropout
        nn.Linear(512, len(os.listdir(train_dir)))
    )

    model = model.to(device)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    # Compute class weights
    unique_classes = np.unique(train_dataset.targets)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=np.array(train_dataset.targets)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Define the loss function with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=50)
