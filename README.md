# Knee Ligament Detection and Classification

A deep learning-based knee MRI image classification project using **EfficientNet-B0 Transfer Learning** to classify knee ligament conditions into five classes: ACL, LCL, MCL, Normal, and PCL.

## Project Overview

Knee ligament injuries are common and accurate classification of ligament conditions from MRI images can support medical image analysis.

This project uses deep learning and transfer learning to classify knee MRI images into five categories:

- ACL – Anterior Cruciate Ligament
- LCL – Lateral Collateral Ligament
- MCL – Medial Collateral Ligament
- Normal – No ligament abnormality
- PCL – Posterior Cruciate Ligament

The project includes image preprocessing, data augmentation, class imbalance handling, EfficientNet-B0 transfer learning, Focal Loss, model evaluation, and Grad-CAM visualization.

---

## Objectives

- Classify knee MRI images into five ligament-related classes.
- Apply transfer learning using EfficientNet-B0.
- Improve image quality using CLAHE preprocessing.
- Apply data augmentation to improve model generalization.
- Handle class imbalance using WeightedRandomSampler and Focal Loss.
- Evaluate the model using accuracy, precision, recall, and F1-score.
- Visualize important image regions using Grad-CAM.
- Develop a complete deep learning workflow for knee ligament classification.

---

## Classes

The model classifies images into:

```text
acl
lcl
mcl
normal
pcl

Methodology

Knee MRI Image
       ↓
Image Preprocessing
       ↓
CLAHE Enhancement
       ↓
Data Augmentation
       ↓
EfficientNet-B0
       ↓
Transfer Learning
       ↓
Focal Loss
       ↓
Model Training
       ↓
Validation
       ↓
Classification
       ↓
Grad-CAM Visualization


Image Preprocessing

The project uses Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.

The image is converted from RGB to LAB color space and CLAHE is applied to the luminance channel.

The following preprocessing techniques are used:

Image resizing
CLAHE enhancement
Tensor conversion
ImageNet normalization

Data Augmentation

Training images are augmented using:

Random rotation
Random affine transformation
Translation
Scaling
Brightness adjustment
Contrast adjustment

Validation and test images use preprocessing without random augmentation.

Model Architecture

The project uses EfficientNet-B0 with pretrained ImageNet weights.

Architecture
Input MRI Image
       ↓
EfficientNet-B0
       ↓
Pretrained Feature Extraction
       ↓
Fine-Tuning of Last 3 Feature Blocks
       ↓
Dropout (0.3)
       ↓
Fully Connected Layer
       ↓
5-Class Output

The original classifier is replaced with:

nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, NUM_CLASSES)
)

The pretrained feature layers are initially frozen, while the last three feature blocks are fine-tuned.

Handling Class Imbalance

The dataset contains different numbers of images for each class.

To address class imbalance, the project uses:

WeightedRandomSampler

A WeightedRandomSampler is used during training to provide greater representation to minority classes.

Focal Loss

A custom Focal Loss implementation is also used to focus learning on difficult examples and address class imbalance.

The implementation is available in:

focal_loss.py
Focal Loss

The project uses a custom Focal Loss implementation with:

Gamma = 2

Class weights are used together with the loss function.

Current class weights:

ACL    : 1.0
LCL    : 3.7
MCL    : 2.9
Normal : 1.0
PCL    : 21.7
Training Configuration

The current training configuration is:

Image Size     : 224 × 224
Batch Size     : 8
Epochs         : 30
Learning Rate  : 0.0001
Optimizer      : Adam

The project automatically selects the available device:

CUDA GPU → if available
CPU      → otherwise
Learning Rate Scheduling

The project uses ReduceLROnPlateau to reduce the learning rate when validation loss stops improving.

Configuration:

Mode       : min
Factor     : 0.1
Patience   : 3
Early Stopping

Early stopping is used to avoid unnecessary training when validation performance stops improving.

The current early stopping patience is:

5 epochs

The best model is saved based on validation accuracy.

Model path:

models/best_model.pth
Model Evaluation

The project calculates the following evaluation metrics:

Accuracy
Precision
Recall
F1-score
Validation Loss

The project uses Scikit-learn for metric calculation.

The actual metric values depend on the trained model and dataset. Therefore, this README does not use fabricated performance values.

For the final project report, the actual values obtained from model evaluation should be added here.

Example:

Accuracy  : Actual Result
Precision : Actual Result
Recall    : Actual Result
F1-Score  : Actual Result
Grad-CAM Visualization

The project includes Grad-CAM functionality for model interpretability.

Grad-CAM helps visualize the regions of an MRI image that contribute to the model's prediction.

The workflow is:

MRI Image
    ↓
EfficientNet-B0
    ↓
Predicted Class
    ↓
Gradient Information
    ↓
Activation Map
    ↓
Grad-CAM Heatmap
    ↓
Highlighted Image Region

Grad-CAM implementation:

gradcam_paper.py
Dataset Structure

The project uses an ImageFolder-compatible dataset structure:

dataset_final/
│
├── train/
│   ├── acl/
│   ├── lcl/
│   ├── mcl/
│   ├── normal/
│   └── pcl/
│
├── valid/
│   ├── acl/
│   ├── lcl/
│   ├── mcl/
│   ├── normal/
│   └── pcl/
│
└── test/
    ├── acl/
    ├── lcl/
    ├── mcl/
    ├── normal/
    └── pcl/

The dataset directories are configured in config.py.

Project Structure
knee_ligament_Project/
│
├── app.py
├── config.py
├── count_dataset.py
├── dataset.py
├── focal_loss.py
├── gradcam_paper.py
├── model.py
├── results.json
├── style.css
├── test.py
├── test_libraries.py
├── train.py
├── utils.py
│
├── models/
│   └── check_model.py
│
├── scripts/
│   ├── find_duplicates.py
│   ├── remove_duplicates.py
│   └── split_dataset.py
│
└── new.streamlit/
    └── config.toml
Technologies Used
Python
PyTorch
Torchvision
EfficientNet-B0
OpenCV
NumPy
Pillow
Scikit-learn
tqdm
Streamlit
Git
GitHub