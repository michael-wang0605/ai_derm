import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torch import nn

def load_model(model_path, num_classes, device):
    """
    Load the ConvNeXt model with the specified number of classes and weights.

    Args:
        model_path (str): Path to the saved model weights file (.pth).
        num_classes (int): Number of classes for the classification task.
        device (torch.device): Device to load the model on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Load the ConvNeXt architecture with pretrained weights
    model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

    # Replace the classification head
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move the model to the specified device
    return model.to(device)

if __name__ == "__main__":
    model_path = "convnext_base_dermnet.pth"
    num_classes = 23  # Adjust this based on your dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(model_path, num_classes, device)
    model.eval()

    # Example of using the model for inference
    # Ensure you have an image path and class names defined
    image_path = "example_image.jpg"
    class_names = ["Class1", "Class2", ...]  # Replace with your actual class names

    from userImage_preprocessing import preprocess_image

    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()

    print(f"Predicted Class: {class_names[predicted_class_idx]}, Confidence: {probabilities[predicted_class_idx]:.2f}")