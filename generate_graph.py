import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(class_names, class_counts):
    """
    Plot the distribution of samples in each class.

    Args:
        class_names (list): List of class names.
        class_counts (list): List of sample counts for each class.
    """
    plt.figure(figsize=(12, 6))
    plt.barh(class_names, class_counts, color='skyblue')
    plt.xlabel("Number of Samples")
    plt.ylabel("Classes")
    plt.title("Class Distribution in the Dataset")
    plt.tight_layout()
    plt.show()

def plot_training_progress(epochs, train_acc, val_acc, train_loss, val_loss):
    """
    Plot training and validation accuracy and loss over epochs.

    Args:
        epochs (list): List of epoch numbers.
        train_acc (list): List of training accuracies.
        val_acc (list): List of validation accuracies.
        train_loss (list): List of training losses.
        val_loss (list): List of validation losses.
    """
    plt.figure(figsize=(12, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss', marker='o', color='orange')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example class distribution
    class_names = ["Acne", "Rosacea", "Lesions", "Eczema", "Psoriasis", "Others"]
    class_counts = [400, 300, 200, 150, 350, 100]
    plot_class_distribution(class_names, class_counts)

    # Example training progress
    epochs = np.arange(1, 21)
    train_acc = np.random.uniform(0.4, 0.8, len(epochs))  # Replace with actual data
    val_acc = np.random.uniform(0.2, 0.6, len(epochs))    # Replace with actual data
    train_loss = np.random.uniform(0.6, 1.2, len(epochs)) # Replace with actual data
    val_loss = np.random.uniform(1.0, 2.0, len(epochs))   # Replace with actual data
    plot_training_progress(epochs, train_acc, val_acc, train_loss, val_loss)
