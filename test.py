from collections import Counter
from preprocessing import get_dataloaders

# Path to your dataset
data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"

# Create DataLoaders
train_loader, test_loader, classes = get_dataloaders(data_dir)

# Analyze class distribution
class_counts = Counter(train_loader.dataset.targets)
print("Class Distribution:")
for class_idx, count in class_counts.items():
    print(f"Class {class_idx} ({train_loader.dataset.classes[class_idx]}): {count} samples")
