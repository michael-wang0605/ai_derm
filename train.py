import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from preprocessing import get_dataloaders
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np

def train_model(data_dir, num_classes, epochs=10, batch_size=32, learning_rate=0.001):
    # Load data
    train_loader, test_loader, classes = get_dataloaders(data_dir, batch_size)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model with weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1  # Specify pretrained weights
    base_model = efficientnet_b0(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False  # Freeze base model

    # Modify the classifier
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(base_model.classifier[1].in_features, num_classes)
    )
    base_model = base_model.to(device)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_loader.dataset.targets),
        y=train_loader.dataset.targets
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(epochs):
        base_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{epochs} starting...")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = base_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Calculate and log epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1} complete! Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Evaluate on validation set
        validate_model(base_model, test_loader, criterion, device)

    # Save the model
    torch.save(base_model.state_dict(), "dermnet_classifier.pth")
    print("Model saved as 'dermnet_classifier.pth'")

    # Evaluate class-wise performance
    evaluate_model(base_model, test_loader, device, classes)

def validate_model(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Track accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Calculate validation metrics
    val_loss = running_loss / len(test_loader)
    val_acc = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
    num_classes = 23  # Replace with the actual number of classes in your dataset
    train_model(data_dir, num_classes)
