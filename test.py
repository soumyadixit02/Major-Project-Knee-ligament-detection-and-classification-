import torch
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import *
from dataset import test_loader
from model import build_model

# -----------------------
# Load Model
# -----------------------
model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

predictions = []
labels = []

# -----------------------
# Testing
# -----------------------
with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        predictions.extend(preds.cpu().numpy())
        labels.extend(targets.cpu().numpy())

# -----------------------
# Metrics
# -----------------------
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average="weighted", zero_division=0)
recall = recall_score(labels, predictions, average="weighted", zero_division=0)
f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

print("\n========== TEST RESULTS ==========")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(labels, predictions))

print("\nConfusion Matrix:\n")
print(confusion_matrix(labels, predictions))
cm = confusion_matrix(labels, predictions)

print(cm)




results = {
    "accuracy": accuracy * 100,
    "precision": precision * 100,
    "recall": recall * 100,
    "f1": f1 * 100,
    "confusion_matrix": cm.tolist()
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved successfully!")