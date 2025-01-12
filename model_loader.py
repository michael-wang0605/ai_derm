import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

def load_model(model_path, num_classes, device):
    """
    Load the ResNet-50 model with the specified number of classes and weights.

    Args:
        model_path (str): Path to the saved model weights file (.pth).
        num_classes (int): Number of classes for the classification task.
        device (torch.device): Device to load the model on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Load the ResNet-50 architecture
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Replace the fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, num_classes)
    )

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move the model to the specified device
    return model.to(device)
