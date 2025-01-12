import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from torch.cuda.amp import autocast, GradScaler  # Import for mixed precision training

# Initialize Gradient Scaler for Mixed Precision
scaler = GradScaler()

def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Track accuracy
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    # Calculate average loss and accuracy
    val_loss = running_loss / len(test_loader)
    val_acc = correct / total
    return val_loss, val_acc

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{epochs} starting...")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():  # Automatically cast operations to mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Step optimizer with scaled gradients
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # Log batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Calculate and log epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


# Evaluation function
def final_evaluation(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Main script guard
if __name__ == '__main__':
    data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    num_classes = 23
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet-50 model
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(20),
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

    # Compute class weights for weighted loss
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=train_dataset.targets
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Define the loss function with weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Compute sample weights for WeightedRandomSampler
    class_sample_counts = [train_dataset.targets.count(i) for i in range(num_classes)]
    class_weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for label in train_dataset.targets]

    # Create a sampler for the DataLoader
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoaders with sampler
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Train and evaluate
    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=30)
    final_evaluation(model, test_loader, device, test_dataset.classes)

    torch.save(model.state_dict(), "resnet50_dermnet.pth")
    print("Model saved as 'resnet50_dermnet.pth'")

