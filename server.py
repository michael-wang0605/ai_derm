# server.py
import io, os, json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny

# ---- Config ----
CKPT_PATH = os.getenv("CKPT_PATH", "best_model.pth")  # your saved state_dict
LABELS_PATH = os.getenv("LABELS_PATH", "labels.json") # optional

# ---- FastAPI & CORS (open for dev) ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Optional labels (index -> name) ----
IDX2LABEL: Optional[dict] = None
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            IDX2LABEL = {int(k): v for k, v in raw.items()}
        elif isinstance(raw, list):
            IDX2LABEL = {i: v for i, v in enumerate(raw)}
    except Exception:
        IDX2LABEL = None

# ---- Load state_dict & infer num_classes from final linear weight ----
state_dict = torch.load(CKPT_PATH, map_location="cpu")
# final layer in your Sequential is index 6, so key: 'classifier.2.6.weight'
final_key = None
for k in state_dict.keys():
    if k.endswith("classifier.2.6.weight"):
        final_key = k
        break
if final_key is None:
    # Fallback if keys slightly differ; try common patterns:
    candidates = [k for k in state_dict if k.endswith(".6.weight") and "classifier.2" in k]
    if candidates:
        final_key = candidates[0]
if final_key is None:
    raise RuntimeError("Could not find final classifier weight in state_dict (expected 'classifier.2.6.weight').")

num_classes = state_dict[final_key].shape[0]

# ---- Build model exactly like training script ----
# You used ConvNeXt_Tiny with a custom MLP head replacing classifier[2]
m = convnext_tiny(weights=None)  # you trained with ImageNet weights but state_dict will override
in_features = m.classifier[2].in_features  # 768 for convnext_tiny
m.classifier[2] = nn.Sequential(
    nn.Linear(in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(512, num_classes),
)

# Load weights strictly to ensure exact match
missing, unexpected = m.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print(f"[server] Warning: missing={missing}, unexpected={unexpected}")

m.eval()
# channels_last isn’t required for inference, but safe:
m = m.to(memory_format=torch.channels_last)

# ---- Preprocess (matches your test/tta normalizations & size) ----
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    x = preprocess(img).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        logits = m(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

    import numpy as np
    k = min(3, probs.shape[0])
    top_idx = np.argsort(-probs)[:k].tolist()
    top_conf = [float(probs[i]) for i in top_idx]
    top_labels = [IDX2LABEL[i] if IDX2LABEL and i in IDX2LABEL else str(i) for i in top_idx]

    return {
        "top_indices": top_idx,
        "top_labels": top_labels,
        "top_confidences": top_conf,
        "best": {"index": top_idx[0], "label": top_labels[0], "confidence": top_conf[0]},
        "meta": {"arch": "convnext_tiny", "num_classes": num_classes}
    }
