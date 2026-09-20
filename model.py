import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def build_model():

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze all layers
    for param in model.features.parameters():
        param.requires_grad = False

# Unfreeze the last 3 EfficientNet blocks
    for param in model.features[-3:].parameters():
     param.requires_grad = True
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, NUM_CLASSES)
    )

    return model