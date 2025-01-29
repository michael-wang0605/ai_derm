import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import torch

def visualize_augmentations(data_dir, num_samples=5):
    """
    Visualize original and augmented images side by side.

    Args:
        data_dir (str): Path to the dataset directory.
        num_samples (int): Number of images to display.
    """
    # Define transformations
    original_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])
    augmented_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=None)

    # Get samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    for i in range(num_samples):
        # Select a random image
        image, _ = dataset[i]  # Get PIL Image and label

        # Apply original and augmented transforms
        original_image = original_transform(image)
        augmented_image = augmented_transform(original_image)

        # Undo normalization for visualization
        def unnormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return tensor * std + mean

        augmented_image = unnormalize(augmented_image)

        # Convert tensors to numpy for plotting
        original_image = transforms.ToTensor()(original_image).permute(1, 2, 0).numpy()
        augmented_image = augmented_image.permute(1, 2, 0).numpy()

        # Plot original and augmented images
        axes[i, 0].imshow(original_image.clip(0, 1))
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(augmented_image.clip(0, 1))
        axes[i, 1].set_title("Augmented Image")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split\\train"
    visualize_augmentations(data_dir, num_samples=5)
