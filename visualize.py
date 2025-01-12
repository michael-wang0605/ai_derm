import matplotlib.pyplot as plt
import torchvision
from preprocessing import get_dataloaders
import torch

def visualize_samples(loader, classes):
    # Get one batch of data
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Create a grid of images
    img_grid = torchvision.utils.make_grid(images[:8], nrow=4)  # Show 8 samples
    img_grid = img_grid.permute(1, 2, 0)  # Rearrange dimensions for matplotlib

    # Normalize the image grid back to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_grid = img_grid * std + mean
    img_grid = img_grid.clamp(0, 1)  # Clamp values to valid range

    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(img_grid)
    plt.title("Sample Images")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
    train_loader, _, classes = get_dataloaders(data_dir)
    visualize_samples(train_loader, classes)
