import torch
import torch.nn.functional as F
import pandas as pd

def evaluate_and_save_per_class_metrics(model, dataloader, criterion, device, num_classes, filename="per_class_metrics.csv"):
    model.eval()
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_loss_sum = [0.0] * num_classes

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                loss = criterion(outputs[i].unsqueeze(0), labels[i].unsqueeze(0)).item()
                class_loss_sum[label] += loss
                if label == pred:
                    class_correct[label] += 1

    # Compile results into a dataframe
    data = []
    for cls in range(num_classes):
        accuracy = class_correct[cls] / class_total[cls] if class_total[cls] != 0 else 0
        loss = class_loss_sum[cls] / class_total[cls] if class_total[cls] != 0 else 0
        error_rate = 1 - accuracy
        data.append({
            "Class": cls,
            "Validation Accuracy": round(accuracy, 4),
            "Validation Loss": round(loss, 4),
            "Error Rate": round(error_rate, 4)
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Per-class metrics saved to {filename}")

