# diagnose_model.py
import torch

# Load the model state dict
pth_path = 'best_model.pth'
state_dict = torch.load(pth_path, map_location='cpu')

print("✅ Loaded state_dict successfully.")
print("\n📦 Top-level keys (usually 'model', or plain weight keys):")
print(list(state_dict.keys())[:5])

# Check if it's wrapped (e.g., saved like `torch.save(model.state_dict())`)
if isinstance(state_dict, dict) and all(isinstance(k, str) for k in state_dict.keys()):
    print("\n🔍 Detected a raw state_dict with layers:")
    for key in state_dict.keys():
        print(f"  - {key}")
    print("\n💡 Based on the layer keys, inspect which layers are used (e.g., classifier.2.1, classifier.0, etc.)")
else:
    print("\n⚠️ Unexpected format. This might be a full model object.")
    print(type(state_dict))
