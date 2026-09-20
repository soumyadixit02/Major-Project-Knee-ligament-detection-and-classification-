import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import DEVICE, IMAGE_SIZE, CLASSES, MODEL_PATH
from model import build_model


# ============================================================
# SETTINGS
# ============================================================

# CHANGE ONLY THIS:
# Put the MRI image you want to analyze in the project folder.
IMAGE_PATH = "sample_mri.jpg"

# Output folder for IEEE paper figures
OUTPUT_DIR = "gradcam_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("Loading EfficientNet-B0...")

    model = build_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# PREPROCESSING
# Same preprocessing used in your project
# ============================================================

def preprocess_image(image_path):

    print("\nLoading image:")
    print(image_path)

    pil_image = Image.open(image_path).convert("RGB")

    original_rgb = np.array(pil_image)

    # Resize
    resized_rgb = cv2.resize(
        original_rgb,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        resized_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_gray = clahe.apply(gray)

    # Convert back to RGB
    clahe_rgb = cv2.cvtColor(
        clahe_gray,
        cv2.COLOR_GRAY2RGB
    )

    # Normalize to 0-1
    normalized = clahe_rgb.astype(
        np.float32
    ) / 255.0

    # ImageNet normalization
    imagenet_mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    imagenet_std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    normalized = (
        normalized - imagenet_mean
    ) / imagenet_std

    # HWC -> CHW
    chw = normalized.transpose(
        2, 0, 1
    )

    # Add batch dimension
    tensor = torch.tensor(
        chw,
        dtype=torch.float32
    ).unsqueeze(0)

    return (
        original_rgb,
        resized_rgb,
        clahe_rgb,
        tensor
    )


# ============================================================
# FIND GRAD-CAM TARGET LAYER
# Your project uses model.features[-1]
# ============================================================

def find_target_layer(model):

    if hasattr(model, "features"):

        return model.features[-1]

    # Fallback
    last_conv = None

    for module in model.modules():

        if isinstance(
            module,
            torch.nn.Conv2d
        ):

            last_conv = module

    return last_conv


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.activations = None

        self.gradients = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )


    def save_activation(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output.detach()


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0].detach()


    def generate(
        self,
        input_tensor,
        class_idx,
        size
    ):

        self.model.zero_grad()

        input_tensor = (
            input_tensor
            .to(DEVICE)
            .clone()
            .requires_grad_(True)
        )

        # Forward pass
        output = self.model(
            input_tensor
        )

        # Select class score
        score = output[
            0,
            class_idx
        ]

        # Backpropagation
        score.backward()

        # Get gradients
        gradients = self.gradients[0]

        # Get activations
        activations = self.activations[0]

        # Global average pooling
        weights = gradients.mean(
            dim=(1, 2)
        )

        # Create CAM
        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device
        )

        for i, weight in enumerate(weights):

            cam += (
                weight *
                activations[i]
            )

        # ReLU
        cam = torch.relu(cam)

        # Convert to NumPy
        cam = cam.cpu().numpy()

        # Resize CAM
        cam = cv2.resize(
            cam,
            (size, size)
        )

        # Normalize 0-1
        cam_min = cam.min()
        cam_max = cam.max()

        if (
            cam_max - cam_min
            > 1e-8
        ):

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(cam)

        return cam


# ============================================================
# CREATE HEATMAP AND OVERLAY
# Same method used in your project
# ============================================================

def create_visualizations(
    original_rgb,
    cam
):

    base = cv2.resize(
        original_rgb,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert CAM to 0-255
    heatmap_uint8 = np.uint8(
        255 * cam
    )

    # Apply JET heatmap
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # BGR -> RGB
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = cv2.addWeighted(
        base,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    return (
        base,
        heatmap_color,
        overlay
    )


# ============================================================
# SAVE INDIVIDUAL IMAGES
# ============================================================

def save_image(
    image,
    filename
):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    Image.fromarray(image).save(
        path
    )

    print(
        "Saved:",
        path
    )


# ============================================================
# CREATE IEEE FIGURE
# Original | Heatmap | Overlay
# ============================================================

def create_ieee_figure(
    base,
    heatmap,
    overlay,
    predicted_class,
    confidence
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    # ------------------------------------
    # Original
    # ------------------------------------

    axes[0].imshow(base)

    axes[0].set_title(
        "Original MRI",
        fontsize=12,
        fontweight="bold"
    )

    axes[0].axis("off")


    # ------------------------------------
    # Grad-CAM Heatmap
    # ------------------------------------

    axes[1].imshow(heatmap)

    axes[1].set_title(
        "Grad-CAM Heatmap",
        fontsize=12,
        fontweight="bold"
    )

    axes[1].axis("off")


    # ------------------------------------
    # Overlay
    # ------------------------------------

    axes[2].imshow(overlay)

    axes[2].set_title(
        f"Grad-CAM Overlay\n"
        f"{predicted_class} ({confidence:.2f}%)",
        fontsize=12,
        fontweight="bold"
    )

    axes[2].axis("off")


    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "Fig_6_GradCAM.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n======================================")
    print("IEEE FIGURE CREATED")
    print("======================================")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("GRAD-CAM FIGURE GENERATOR")
    print("======================================")

    # ------------------------------------
    # Check MRI image
    # ------------------------------------

    if not os.path.exists(IMAGE_PATH):

        print("\nERROR:")
        print(
            f"Image not found: {IMAGE_PATH}"
        )

        print(
            "\nPut your MRI image in the "
            "project folder and name it:"
        )

        print("sample_mri.jpg")

        return


    # ------------------------------------
    # Load model
    # ------------------------------------

    model = load_model()


    # ------------------------------------
    # Preprocess
    # ------------------------------------

    (
        original_rgb,
        resized_rgb,
        clahe_rgb,
        input_tensor
    ) = preprocess_image(
        IMAGE_PATH
    )


    # ------------------------------------
    # Prediction
    # ------------------------------------

    input_device = input_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            input_device
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    probabilities = (
        probabilities
        .cpu()
        .numpy()[0]
    )

    predicted_idx = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASSES[
        predicted_idx
    ]

    confidence = (
        probabilities[predicted_idx]
        * 100
    )


    print("\n======================================")
    print("MODEL PREDICTION")
    print("======================================")

    print(
        "Predicted Class :",
        predicted_class
    )

    print(
        "Confidence      :",
        f"{confidence:.2f}%"
    )


    # ------------------------------------
    # Print all probabilities
    # ------------------------------------

    print("\nClass probabilities:")

    for i, class_name in enumerate(
        CLASSES
    ):

        print(
            f"{class_name:10s} : "
            f"{probabilities[i] * 100:.2f}%"
        )


    # ------------------------------------
    # Target layer
    # ------------------------------------

    target_layer = find_target_layer(
        model
    )

    print(
        "\nGrad-CAM target layer:"
    )

    print(target_layer)


    # ------------------------------------
    # Generate Grad-CAM
    # ------------------------------------

    gradcam = GradCAM(
        model,
        target_layer
    )

    cam = gradcam.generate(
        input_tensor,
        predicted_idx,
        IMAGE_SIZE
    )


    # ------------------------------------
    # Create visualizations
    # ------------------------------------

    (
        base,
        heatmap,
        overlay
    ) = create_visualizations(
        original_rgb,
        cam
    )


    # ------------------------------------
    # Save individual images
    # ------------------------------------

    save_image(
        base,
        "original_mri.png"
    )

    save_image(
        heatmap,
        "gradcam_heatmap.png"
    )

    save_image(
        overlay,
        "gradcam_overlay.png"
    )


    # ------------------------------------
    # Create final IEEE figure
    # ------------------------------------

    create_ieee_figure(
        base,
        heatmap,
        overlay,
        predicted_class,
        confidence
    )


    print("\n======================================")
    print("DONE")
    print("======================================")

    print(
        "\nOpen the folder:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print(
        "\nUse Fig_6_GradCAM.png "
        "for your IEEE paper."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()