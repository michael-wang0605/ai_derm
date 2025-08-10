from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny
import os
from datetime import datetime

app = Flask(__name__)

# === Configuration ===
TRAIN_DIR = r"C:\Users\mwang\ai_derm\dataset_categorized_final_split\train"
UPLOAD_DIR = "uploads"
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 23  # Update if your dataset changes
MAX_FILE_SIZE_MB = 5

# === Ensure upload directory exists ===
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Class names from training folder ===
class_names = sorted(os.listdir(TRAIN_DIR))

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define model architecture ===
model = convnext_tiny(weights=None)
model.classifier = nn.Sequential(
    nn.Flatten(),                  # classifier.0
    nn.Dropout(0.3),               # classifier.1
    nn.Linear(768, NUM_CLASSES)   # classifier.2
)


# === Load model weights ===
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# === Image transform (match training) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if not file.mimetype.startswith('image/'):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

    if request.content_length and request.content_length > MAX_FILE_SIZE_MB * 1024 * 1024:
        return jsonify({'error': f'File size exceeds {MAX_FILE_SIZE_MB}MB.'}), 400

    try:
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        # Preprocess image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        condition = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

        return jsonify({
            'condition': condition,
            'class_index': class_idx,
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    try:
        return jsonify({'classes': class_names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
