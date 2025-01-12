from torchvision.datasets import ImageFolder

# Path to your training dataset
train_dir = "C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split\\train"
train_dataset = ImageFolder(train_dir)

# Save class names
with open("class_names.txt", "w") as f:
    for class_name in train_dataset.classes:
        f.write(f"{class_name}\n")

print("Class names saved to class_names.txt")
