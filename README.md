# Knee Ligament Detection and Classification

<p align="center">

### Deep Learning Based Knee MRI Classification using EfficientNet-B0

A five-class knee ligament classification system using Transfer Learning, CLAHE preprocessing, data augmentation, Focal Loss, and Grad-CAM visualization.

</p>

---

## 📌 Project Overview

Knee ligament injuries are common conditions that can affect mobility and physical activity. Medical imaging, particularly Magnetic Resonance Imaging (MRI), provides important visual information for analyzing ligament conditions.

This project presents a deep learning-based system for **knee ligament detection and classification from MRI images**.

The proposed system classifies knee MRI images into five categories:

- **ACL** — Anterior Cruciate Ligament
- **LCL** — Lateral Collateral Ligament
- **MCL** — Medial Collateral Ligament
- **Normal** — No ligament abnormality
- **PCL** — Posterior Cruciate Ligament

The project uses **EfficientNet-B0 with Transfer Learning** as the primary deep learning model. Image preprocessing is performed using **CLAHE**, while **data augmentation**, **WeightedRandomSampler**, and **Focal Loss** are used to improve learning from an imbalanced dataset.

The project also includes **Grad-CAM visualization** to provide an interpretable view of the regions of the MRI image that contribute to the model prediction.

---

## 🎯 Objectives

The main objectives of this project are:

- 🔹 To develop an automated knee MRI classification system.
- 🔹 To classify MRI images into five ligament-related categories.
- 🔹 To use Transfer Learning for effective feature extraction.
- 🔹 To improve MRI image contrast using CLAHE preprocessing.
- 🔹 To use data augmentation to improve model generalization.
- 🔹 To handle class imbalance using WeightedRandomSampler.
- 🔹 To use Focal Loss for difficult and minority-class samples.
- 🔹 To evaluate the model using Accuracy, Precision, Recall, and F1-score.
- 🔹 To visualize important regions using Grad-CAM.
- 🔹 To develop a user-friendly Streamlit-based application.

---

# 🔬 METHODOLOGY

The complete methodology of the proposed system consists of multiple stages, beginning with dataset preparation and ending with classification and model interpretability.

```text
                  Knee MRI Dataset
                         │
                         ▼
                 Dataset Preparation
                         │
                         ▼
                 Dataset Splitting
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           Train       Valid        Test
             │           │           │
             ▼           ▼           ▼
        Preprocessing  Preprocessing  Preprocessing
             │
             ▼
       CLAHE Enhancement
             │
             ▼
       Data Augmentation
             │
             ▼
       Weighted Sampling
             │
             ▼
      EfficientNet-B0
             │
             ▼
      Transfer Learning
             │
             ▼
         Focal Loss
             │
             ▼
        Model Training
             │
             ▼
          Validation
             │
             ▼
       Best Model Saving
             │
             ▼
       Model Evaluation
             │
             ▼
      Five-Class Prediction
             │
             ▼
       Grad-CAM Analysis
```

---

# 🗂️ 1. Dataset Preparation

The dataset contains knee MRI images belonging to five classes:

```text
ACL
LCL
MCL
Normal
PCL
```

The dataset is organized into training, validation, and testing directories.

```text
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
```

The dataset is loaded using the PyTorch `ImageFolder` structure.

---

# 🖼️ 2. Image Preprocessing

MRI images can have variations in contrast and brightness. Therefore, preprocessing is applied before feeding images into the deep learning model.

The preprocessing pipeline includes:

```text
Original MRI Image
        ↓
Image Resizing
        ↓
RGB → LAB
        ↓
CLAHE Enhancement
        ↓
LAB → RGB
        ↓
Tensor Conversion
        ↓
ImageNet Normalization
```

## CLAHE

**Contrast Limited Adaptive Histogram Equalization (CLAHE)** is used to improve local image contrast.

The image is converted into LAB color space, and CLAHE is applied to the luminance channel.

This preprocessing helps enhance important visual structures in MRI images while limiting excessive contrast amplification.

---

# 🔄 3. Data Augmentation

Data augmentation is applied to training images to increase variation in the training data and improve model generalization.

The augmentation pipeline includes:

- 🔄 Random rotation
- 🔄 Random affine transformation
- ↔️ Translation
- 🔍 Scaling
- ☀️ Brightness adjustment
- ◐ Contrast adjustment

The training pipeline can be represented as:

```text
MRI Image
    ↓
Resize
    ↓
CLAHE
    ↓
Random Rotation
    ↓
Random Affine Transformation
    ↓
Color Jitter
    ↓
Tensor Conversion
    ↓
ImageNet Normalization
```

Validation and testing images are processed without random augmentation so that model performance can be evaluated consistently.

---

# ⚖️ 4. Handling Class Imbalance

The dataset contains different numbers of images across the five classes.

Class imbalance can cause a deep learning model to learn majority classes more strongly than minority classes.

Therefore, this project uses two techniques:

### WeightedRandomSampler

`WeightedRandomSampler` provides increased sampling probability to classes with fewer training examples.

```text
Minority Class
      ↓
Higher Sampling Weight
      ↓
More Balanced Training
```

### Focal Loss

Focal Loss is used to focus training on difficult examples and reduce the influence of easily classified samples.

The project contains a custom implementation in:

```text
focal_loss.py
```

---

# 🔥 5. Focal Loss

The project uses a custom Focal Loss implementation.

The focal loss parameter is:

```text
Gamma = 2
```

The current class weights used by the project are:

| Class | Weight |
|---|---:|
| ACL | 1.0 |
| LCL | 3.7 |
| MCL | 2.9 |
| Normal | 1.0 |
| PCL | 21.7 |

These weights are used to give additional importance to classes with lower representation.

---

# 🧠 6. Model Architecture

The project uses **EfficientNet-B0** with pretrained ImageNet weights.

EfficientNet-B0 is used as the backbone for extracting useful visual features from knee MRI images.

The architecture is:

```text
                 Input MRI Image
                        │
                        ▼
                 EfficientNet-B0
                        │
                        ▼
             Pretrained Feature Layers
                        │
                        ▼
              Fine-Tuned Feature Blocks
                        │
                        ▼
                   Dropout
                   (0.3)
                        │
                        ▼
              Fully Connected Layer
                        │
                        ▼
                 5-Class Output
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
         ACL           LCL           MCL
          │             │             │
          └─────────────┼─────────────┘
                        │
                  Normal / PCL
```

The original EfficientNet classifier is replaced with a custom classifier:

```python
nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, NUM_CLASSES)
)
```

The project fine-tunes the last three feature blocks while using pretrained weights for the remaining feature extraction layers.

---

# 🔄 7. Transfer Learning

Transfer Learning is used instead of training the complete neural network from scratch.

The EfficientNet-B0 model is initialized with pretrained ImageNet weights.

The approach is:

```text
Pretrained EfficientNet-B0
          ↓
Freeze Earlier Features
          ↓
Fine-Tune Last Feature Blocks
          ↓
Replace Original Classifier
          ↓
Train for 5 Classes
```

This allows the model to reuse previously learned visual features while adapting the network to knee MRI classification.

---

# ⚙️ 8. Training Configuration

The current training configuration is:

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Image Size | 224 × 224 |
| Batch Size | 8 |
| Epochs | 30 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Dropout | 0.3 |
| Focal Loss Gamma | 2 |

The project automatically selects the available device:

```text
CUDA GPU → If available
CPU      → Otherwise
```

---

# 📉 9. Learning Rate Scheduling

The project uses `ReduceLROnPlateau`.

The learning rate is reduced when the validation loss stops improving.

Configuration:

| Parameter | Value |
|---|---:|
| Mode | `min` |
| Factor | `0.1` |
| Patience | `3` |

This helps the model continue learning when improvement becomes slower during training.

---

# ⏹️ 10. Early Stopping

Early stopping is used to prevent unnecessary training when validation performance stops improving.

```text
Early Stopping Patience = 5 epochs
```

The best-performing model is saved during training.

The saved model is:

```text
models/best_model.pth
```

---

# 🏋️ 11. Model Training Workflow

The complete training workflow is:

```text
Load Dataset
      ↓
Read Training Images
      ↓
Apply CLAHE
      ↓
Apply Data Augmentation
      ↓
Create Weighted Sampler
      ↓
Load EfficientNet-B0
      ↓
Load Pretrained Weights
      ↓
Fine-Tune Selected Layers
      ↓
Apply Focal Loss
      ↓
Train Model
      ↓
Validate Model
      ↓
Calculate Validation Loss
      ↓
Calculate Validation Accuracy
      ↓
Learning Rate Scheduling
      ↓
Early Stopping
      ↓
Save Best Model
```

Training can be started using:

```bash
python train.py
```

---

# 📊 12. Model Evaluation

The model is evaluated using standard classification metrics.

The project evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- Validation Loss

The evaluation process is:

```text
Test MRI Images
       ↓
Trained EfficientNet-B0
       ↓
Predicted Classes
       ↓
Compare with Actual Classes
       ↓
Calculate Metrics
       ↓
Generate Evaluation Results
```

### Evaluation Metrics

#### Accuracy

Accuracy represents the proportion of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

#### Precision

Precision measures how many predicted positive samples are actually correct.

```text
Precision =
True Positive / (True Positive + False Positive)
```

#### Recall

Recall measures how many actual positive samples are correctly identified.

```text
Recall =
True Positive / (True Positive + False Negative)
```

#### F1-Score

F1-score combines precision and recall.

```text
F1-Score =
2 × (Precision × Recall) /
(Precision + Recall)
```

---

# 📈 13. Confusion Matrix

A confusion matrix can be used to analyze classification performance across the five classes.

```text
             Predicted
          ACL LCL MCL Normal PCL
Actual
ACL
LCL
MCL
Normal
PCL
```

The confusion matrix helps identify which classes are being confused with one another.

The five classes are:

```text
ACL
LCL
MCL
Normal
PCL
```

---

# 🔥 14. Grad-CAM Visualization

The project includes Grad-CAM for model interpretability.

**Gradient-weighted Class Activation Mapping (Grad-CAM)** provides a visual explanation of which regions of an image contributed to the model's prediction.

The workflow is:

```text
Input MRI Image
       ↓
EfficientNet-B0
       ↓
Predicted Class
       ↓
Calculate Gradients
       ↓
Generate Activation Map
       ↓
Generate Heatmap
       ↓
Overlay Heatmap on MRI
       ↓
Visual Explanation
```

Grad-CAM implementation is included in:

```text
gradcam_paper.py
```

The visualization helps demonstrate which image regions the model is focusing on during classification.

---

# 🌐 15. Streamlit Application

The project includes a Streamlit-based application for interacting with the trained model.

The application can be launched using:

```bash
streamlit run app.py
```

The application is designed to provide a user-friendly interface for the knee MRI classification system.

The workflow is:

```text
Upload MRI Image
       ↓
Preprocess Image
       ↓
Load Trained Model
       ↓
Generate Prediction
       ↓
Display Predicted Class
       ↓
Display Confidence
       ↓
Grad-CAM Visualization
```

---

# 📁 16. Project Structure

```text
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
```

---

# 🛠️ 17. Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming Language |
| 🔥 PyTorch | Deep Learning Framework |
| 👁️ Torchvision | Computer Vision and Pretrained Models |
| 🧠 EfficientNet-B0 | Transfer Learning Architecture |
| 🖼️ OpenCV | Image Processing and CLAHE |
| 🔢 NumPy | Numerical Computing |
| 🖼️ Pillow | Image Processing |
| 📊 Scikit-learn | Model Evaluation |
| ⏳ tqdm | Progress Display |
| 🌐 Streamlit | Web Application |
| 🔥 Grad-CAM | Model Interpretability |
| 📦 Git | Version Control |
| 🐙 GitHub | Repository Hosting |

---

# 🚀 18. Installation

## Clone the Repository

```bash
git clone https://github.com/soumyadixit02/Major-Project-Knee-ligament-detection-and-classification-.git
```

## Navigate to the Project

```bash
cd Major-Project-Knee-ligament-detection-and-classification-
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment on Windows

```powershell
venv\Scripts\activate
```

## Install PyTorch and Torchvision

```bash
pip install torch torchvision
```

## Install Required Libraries

```bash
pip install numpy opencv-python Pillow scikit-learn tqdm streamlit
```

---

# ⚙️ 19. Configuration

The main project configuration is available in:

```text
config.py
```

Important parameters include:

```python
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001
```

The project also checks whether CUDA is available and uses the GPU when available.

---

# 🔢 20. Dataset Counting

The project includes:

```text
count_dataset.py
```

Run:

```bash
python count_dataset.py
```

This can be used to inspect the number of images available in the training, validation, and testing datasets.

---

# 🏃 21. Running the Project

## Train the Model

```bash
python train.py
```

## Run the Streamlit Application

```bash
streamlit run app.py
```

## Count Dataset Images

```bash
python count_dataset.py
```

## Run Testing

```bash
python test.py
```

---

# 📈 22. Results

The final performance of the model should be reported using the actual evaluation output obtained after training.

The main metrics are:

| Metric | Result |
|---|---|
| Accuracy | Actual Evaluation Result |
| Precision | Actual Evaluation Result |
| Recall | Actual Evaluation Result |
| F1-Score | Actual Evaluation Result |

No fabricated performance values are included in this README.

The final results should be updated after evaluating the trained model.

---

# 🔍 23. Model Interpretability

The combination of classification and Grad-CAM provides two outputs:

```text
                    MRI Image
                       │
                       ▼
                EfficientNet-B0
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Classification       Grad-CAM
              │                 │
              ▼                 ▼
       Predicted Class      Heatmap
              │                 │
              └────────┬────────┘
                       ▼
               Final Explanation
```

This allows the project to provide both a classification result and a visual explanation of the model's attention.

---

# 🔮 24. Future Enhancements

Possible future improvements include:

- 📌 Increasing the size and diversity of the dataset.
- 📌 Further improving class balancing.
- 📌 Hyperparameter optimization.
- 📌 Comparing additional pretrained architectures.
- 📌 Improving Grad-CAM visualization.
- 📌 External validation using additional datasets.
- 📌 More detailed confusion matrix analysis.
- 📌 Model deployment on a production server.
- 📌 Integration with a larger medical imaging workflow.
- 📌 Further optimization of inference speed.

---

# ⚠️ 25. Disclaimer

This project is developed for **educational, academic, and research purposes**.

The predictions generated by this system should not be considered a medical diagnosis.

The system is intended to demonstrate the application of deep learning and computer vision techniques to knee MRI image classification. Medical decisions should be made by qualified healthcare professionals using appropriate clinical evaluation and diagnostic procedures.

---

# 👩‍💻 26. Author

## Soumya Dixit

GitHub:

**soumyadixit02**

---

# ⭐ 27. Project Highlights

```text
🦿 Knee MRI Classification
🎯 Five-Class Ligament Classification
🧠 EfficientNet-B0
🔄 Transfer Learning
🖼️ CLAHE Preprocessing
🔁 Data Augmentation
⚖️ WeightedRandomSampler
🔥 Focal Loss
📉 Learning Rate Scheduling
⏹️ Early Stopping
📊 Accuracy
📊 Precision
📊 Recall
📊 F1-Score
📋 Confusion Matrix
🔥 Grad-CAM Visualization
🌐 Streamlit Application
🐙 GitHub Version Control
```

---

<p align="center">

##  Knee Ligament Detection and Classification

**Deep Learning • Computer Vision • Medical Image Analysis**

⭐ If you find this project useful, consider giving the repository a star!

</p>
