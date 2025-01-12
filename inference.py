from model_loader import load_model
from userImage_preprocessing import preprocess_image
import torch

# Load class names from a file
def load_class_names(file_path="class_names.txt"):
    """
    Load class names from a file where each line corresponds to a class name.

    Args:
        file_path (str): Path to the file containing class names.

    Returns:
        list: A list of class names.
    """
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def predict(image_path, model, class_names, device):
    """
    Predict the class of an input image.

    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): The trained model.
        class_names (list): List of class names.
        device (torch.device): Device to run the model on.

    Returns:
        str, float: Predicted class name and confidence score.
    """
    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()

    if predicted_class_idx >= len(class_names):
        raise IndexError(f"Predicted class index {predicted_class_idx} is out of range for class names.")
    
    return class_names[predicted_class_idx], probabilities[predicted_class_idx].item()

if __name__ == "__main__":
    model_path = "resnet50_dermnet.pth"
    class_names_file = "class_names.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically load class names
    class_names = load_class_names(class_names_file)
    num_classes = len(class_names)  # Ensure num_classes matches the number of class names

    # Load the trained model
    model = load_model(model_path, num_classes, device)
    model.eval()

    # Hardcoded image path
    image_path = "C:\\Users\\mwang\\ai_derm\\example.jpg"

    # Predict the class
    try:
        predicted_class, confidence = predict(image_path, model, class_names, device)
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
    except IndexError as e:
        print(f"Error: {e}")
