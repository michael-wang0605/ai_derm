import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def lr_finder(model, train_loader, criterion, optimizer, device,
              init_value=1e-5, final_value=1e-2, beta=0.98):
    """
    Run a learning rate finder test with a progress bar.
    """
    num_batches = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num_batches)
    lr = init_value
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    avg_loss = 0.0
    best_loss = float('inf')
    losses = []
    log_lrs = []

    model.train()
    progress_bar = tqdm(train_loader, desc="LR Finder", leave=False)
    for batch_num, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_num + 1))

        losses.append(smoothed_loss)
        log_lrs.append(torch.log10(torch.tensor(lr)).item())

        progress_bar.set_postfix({
            "LR": f"{lr:.2e}",
            "Loss": f"{smoothed_loss:.4f}"
        })

        # Stop if the loss explodes
        if batch_num > 0 and smoothed_loss > 4 * best_loss:
            break

        if smoothed_loss < best_loss or batch_num == 0:
            best_loss = smoothed_loss

        loss.backward()
        optimizer.step()

        # Update the learning rate for the next batch
        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return log_lrs, losses

if __name__ == '__main__':
    data_dir = r"C:\Users\mwang\ai_derm\dataset_categorized_final_split"
    train_dir = os.path.join(data_dir, "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------
    # 1) Model Setup
    # -------------------------
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    
    num_classes = len(os.listdir(train_dir))
    # Reduce dropout to 0.2
    model.classifier[2] = nn.Sequential(
        nn.Linear(model.classifier[2].in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    # -------------------------
    # 2) Dataset & Transforms
    # -------------------------
    # Minimal transforms (no augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_train_dataset = ImageFolder(train_dir, transform=train_transform)

    # Subset for LR finder
    subset_size = 2000
    indices = list(range(len(full_train_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    subset_dataset = Subset(full_train_dataset, subset_indices)

    # -------------------------
    # 3) Loss & DataLoader
    # -------------------------
    # Remove class weights and label smoothing
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        subset_dataset,
        batch_size=8,  # smaller batch for more steps
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------
    # 4) LR Finder
    # -------------------------
    # Expand the range to 1e-2
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    log_lrs, losses = lr_finder(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        init_value=1e-5,
        final_value=1e-2
    )

    plt.plot(log_lrs, losses)
    plt.xlabel("Log10 Learning Rate")
    plt.ylabel("Loss")
    plt.title("LR Finder (Minimal Augmentation, No Class Weights)")
    plt.show()

    # Reinitialize model before actual training.
