import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import os

# Function to perform prediction with optional TTA
def predict_with_tta(model, image_path, device, tta_times=5, tta_transform=None, class_labels=None):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    outputs = []
    with torch.no_grad():
        for _ in range(tta_times):
            aug_image = tta_transform(image)
            aug_image = aug_image.unsqueeze(0).to(device)
            output = model(aug_image)
            outputs.append(torch.softmax(output, dim=1))
        outputs = torch.stack(outputs).mean(dim=0)
        predicted_class = torch.argmax(outputs, dim=1).item()
        predicted_label = class_labels[predicted_class]
        confidence = outputs[0, predicted_class].item()
    return predicted_label, confidence

# Main script
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths
    data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
    train_dir = os.path.join(data_dir, "train")

    # Get class labels
    class_labels = sorted(os.listdir(train_dir))
    num_classes = len(class_labels)

    # Initialize model architecture to match training
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Sequential(
        nn.Linear(model.classifier[2].in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    # Load the .pth file
    pth_path = "best_model.pth"  # Replace with your .pth file path if not in current directory
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Define TTA transform (matching training code)
    tta_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # List of image paths
    image_paths = [
        "C:\\Users\\mwang\\ai_derm\\image_1.jpg",  # Replace with your image paths
        "C:\\Users\mwang\\ai_derm\\image_2.jpg",
        "C:\\Users\mwang\\ai_derm\\image_3.webp"
    ]

    # Process and classify each image with TTA
    for img_path in image_paths:
        try:
            predicted_label, confidence = predict_with_tta(model, img_path, device, tta_times=5, tta_transform=tta_transform, class_labels=class_labels)
            print(f"Image: {img_path}")
            print(f"Predicted class: {predicted_label} (Confidence: {confidence:.4f})")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    main()