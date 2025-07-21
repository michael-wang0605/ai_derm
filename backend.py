from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os

app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torch.hub.load('pytorch/vision', 'convnext_tiny', pretrained=False)
num_classes = 3  # Replace with len(os.listdir(train_dir)) from your dataset
model.classifier[2] = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[2].in_features, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.6),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.6),
    torch.nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing (same as test_transform in your training code)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Mock mappings (replace with your actual dataset classes and logic)
condition_map = {
    0: "Acne Vulgaris",
    1: "Eczema",
    2: "Psoriasis"
}
severity_map = {
    0: "Mild",
    1: "Moderate",
    2: "Severe"
}
treatment_map = {
    "Acne Vulgaris": "Topical retinoids, benzoyl peroxide",
    "Eczema": "Moisturizers, topical corticosteroids",
    "Psoriasis": "Topical corticosteroids, phototherapy"
}

# Mock drug interaction logic (replace with your actual database or logic)
def check_drug_interactions(treatment):
    if "benzoyl peroxide" in treatment.lower():
        return "Avoid combining with oral isotretinoin"
    elif "corticosteroids" in treatment.lower():
        return "Monitor for long-term use side effects"
    return "No warnings"

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if not file.mimetype.startswith('image/'):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

    # Validate file size (max 5MB)
    if file.content_length > 5 * 1024 * 1024:
        return jsonify({'error': 'File size exceeds 5MB. Please upload a smaller image.'}), 400

    try:
        # Process image
        image = Image.open(file.stream).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Run model inference
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        # Map outputs to response
        condition = condition_map.get(class_idx, "Unknown")
        severity = severity_map.get(class_idx % len(severity_map), "Unknown")
        treatment = treatment_map.get(condition, "Consult a dermatologist")
        warning = check_drug_interactions(treatment)

        return jsonify({
            'condition': condition,
            'severity': severity,
            'treatment': treatment,
            'warning': warning
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)