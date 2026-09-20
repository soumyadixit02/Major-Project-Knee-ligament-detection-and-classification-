import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import cv2
from PIL import Image

from config import TRAIN_DIR, VALID_DIR, TEST_DIR, BATCH_SIZE, IMAGE_SIZE

# --------------------------
# Data Augmentation (Training)
# --------------------------
class CLAHE(object):

    def __call__(self, img):

        img = np.array(img)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l,a,b))

        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img)
    
train_transform = transforms.Compose([
   transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    CLAHE(),

    transforms.RandomRotation(10),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.03,0.03),
        scale=(0.95,1.05)
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
# --------------------------
# Validation & Test
# --------------------------

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    CLAHE(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------
# Load Dataset
# --------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
valid_dataset = datasets.ImageFolder(VALID_DIR, transform=test_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# --------------------------
# Create Weighted Sampler
# --------------------------
targets = np.array(train_dataset.targets)

class_count = np.bincount(targets)

class_weights = 1.0 / class_count

sample_weights = class_weights[targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# --------------------------
# DataLoaders
# --------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Dataset Loaded Successfully!")

print("Classes:", train_dataset.classes)
print("Training Images:", len(train_dataset))
print("Validation Images:", len(valid_dataset))
print("Testing Images:", len(test_dataset))