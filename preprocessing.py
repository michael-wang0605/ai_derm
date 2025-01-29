import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import torch
import numpy as np
from torchvision.transforms import RandAugment

def get_dataloaders(batch_size=32, input_size=224):
    # Define the dataset directory
    data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"

    # Enhanced data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define datasets
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Improved Weighted Sampling
    class_counts = Counter(train_dataset.targets)
    sample_weights = [1.0 / (class_counts[label] + 1) for label in train_dataset.targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_dataset.classes
