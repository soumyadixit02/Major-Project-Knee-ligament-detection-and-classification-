import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tqdm import tqdm
from focal_loss import FocalLoss


from config import *
from dataset import (
    train_loader,
    valid_loader
)
from model import build_model
# ----------------------------
# Build Model
# ----------------------------
model = build_model()
model = model.to(DEVICE)
# ----------------------------


# ----------------------------
# Loss Function
# ----------------------------
# Class weights (ACL, LCL, MCL, Normal, PCL)
class_weights = torch.tensor(
    [1.0, 3.7, 2.9, 1.0, 21.7],
    dtype=torch.float32
).to(DEVICE)

criterion = FocalLoss(
    alpha=class_weights,
    gamma=2
)
# ----------------------------
# Optimizer
# ----------------------------
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ----------------------------
# Learning Rate Scheduler
# ----------------------------
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=3
)

# ----------------------------
# Variables to save best model
# ----------------------------
best_accuracy = 0.0
best_model = copy.deepcopy(model.state_dict())
patience = 5
counter = 0
# ----------------------------
# Training Function
# ----------------------------
def train_one_epoch():

    model.train()

    running_loss = 0

    predictions = []
    labels = []

    loop = tqdm(train_loader)

    for images, targets in loop:

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        predictions.extend(preds.cpu().numpy())
        labels.extend(targets.cpu().numpy())

        loop.set_postfix(loss=loss.item())

    accuracy = accuracy_score(labels, predictions)

    return running_loss / len(train_loader), accuracy
# ----------------------------
# Validation Function
# ----------------------------
def validate():

    model.eval()

    running_loss = 0

    predictions = []
    labels = []

    with torch.no_grad():

        for images, targets in valid_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, targets)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return (
        running_loss / len(valid_loader),
        accuracy,
        precision,
        recall,
        f1
    )
# ----------------------------
# Main Training Loop
# ----------------------------
counter = 0
patience = 5
print("\nTraining Started...\n")

for epoch in range(EPOCHS):

    start = time.time()

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc, precision, recall, f1 = validate()

    scheduler.step(val_loss)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    print(f"Train Loss      : {train_loss:.4f}")
    print(f"Train Accuracy  : {train_acc*100:.2f}%")
    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Validation Accuracy : {val_acc*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")
    print(f"F1 Score  : {f1*100:.2f}%")
    print(f"Time : {time.time()-start:.2f} sec")

    # Save Best Model
    if val_acc > best_accuracy:

        best_accuracy = val_acc

        best_model = copy.deepcopy(model.state_dict())

        os.makedirs("models", exist_ok=True)

        torch.save(best_model, MODEL_PATH)

        print("✅ Best Model Saved!")

        counter = 0

    else:

        counter += 1

        print(f"No improvement ({counter}/{patience})")

        if counter >= patience:

            print("\nEarly Stopping Triggered!")

            break

print("\n========================")
print("Training Completed!")
print(f"Best Validation Accuracy : {best_accuracy*100:.2f}%")
print("========================")