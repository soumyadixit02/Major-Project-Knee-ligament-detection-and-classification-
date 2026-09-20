import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
TRAIN_DIR = "dataset_final/train"
VALID_DIR = "dataset_final/valid"
TEST_DIR = "dataset_final/test"

# Image
IMAGE_SIZE = 224

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001

# Classes
CLASSES = ["acl", "lcl", "mcl", "normal", "pcl"]

NUM_CLASSES = len(CLASSES)

# Save model
MODEL_PATH = "models/best_model.pth"