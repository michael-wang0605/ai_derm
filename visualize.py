import os
from PIL import Image

# Path to your dataset
data_dir = r"C:\\Users\\mwang\\ai_derm\\dataset_categorized_final_split"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

def rewrite_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Open the image and save it back to clean up any issues
                with Image.open(file_path) as img:
                    img = img.convert('RGB')  # Ensure the image is in RGB format
                    img.save(file_path, format='JPEG')  # Rewrite the image
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

# Rewrite images in training and testing directories
print("Rewriting training images...")
rewrite_images(train_dir)

print("Rewriting testing images...")
rewrite_images(test_dir)

print("All images have been rewritten successfully.")
