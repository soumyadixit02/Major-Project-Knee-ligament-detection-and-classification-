"""
================================================================================
AI Assisted Knee Ligament Tear Detection & Classification System
--------------------------------------------------------------------------------
UI LAYER ONLY.
================================================================================
"""

import os
import json
import cv2
import numpy as np
import torch
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

# ------------------------------------------------------------------------------
# PROJECT-OWNED MODULES
# ------------------------------------------------------------------------------
from config import DEVICE, IMAGE_SIZE, CLASSES, NUM_CLASSES, MODEL_PATH
from model import build_model

# ==============================================================================
# CONSTANTS & INLINE CSS
# ==============================================================================

RESULTS_PATH = "results.json"
LOGO_PATH = "iamges/logo.png"

# Integrated Bulletproof Card Layout CSS (Updated for proper box separation)
INLINE_CSS = """
<style>
/* ============================================================
   Knee AI - Bulletproof Card Layout CSS
   ============================================================ */

/* 1. Force Main Background to Light Gray */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #F4F7FB !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1500px; }

/* 2. Turn Native Containers into Elevated Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.06) !important;
    border: 1px solid #E2E8F0 !important;
    height: 100%;
}

/* 3. Protect the Sidebar from Card Styling */
section[data-testid="stSidebar"] {
    background-color: #0D1B2A !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] * {
    color: #E0E6ED;
}

/* 4. Force File Uploader Styling */
[data-testid="stFileUploadDropzone"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"] * {
    color: #4A5568 !important;
}

/* 5. Typography & Internal Component Styling */
.card-title { font-size: 16px; font-weight: 700; color: #1E293B; margin-bottom: 16px; margin-top: 0; }
.sidebar-brand { display: flex; align-items: center; gap: 10px; padding: 6px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px; }
.sidebar-brand-title { font-size: 19px; font-weight: 700; color: #FFFFFF !important; line-height: 1.1; }
.sidebar-brand-subtitle { font-size: 11.5px; color: #7C93AC !important; }
.sidebar-section-title { margin-top: 22px; margin-bottom: 8px; font-size: 11px; letter-spacing: 0.07em; color: #6C87A3 !important; font-weight: 700; }
.sidebar-info-row { display: flex; justify-content: space-between; font-size: 12.5px; color: #C9D6E3 !important; padding: 6px 2px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sidebar-info-row span:last-child { font-weight: 600; color: #FFFFFF !important; }

.header-block { display: flex; align-items: center; gap: 16px; padding: 4px 0; }
.header-title { font-size: 23px; font-weight: 800; color: #1D3557; line-height: 1.28; letter-spacing: -0.01em; }
.header-subtitle { font-size: 13px; color: #6C87A3; margin-top: 6px; }

.project-by-card { display: flex; align-items: center; justify-content: flex-end; gap: 12px; height: 100%; }
.project-by-icon { font-size: 22px; background: #EAF2FB; color: #1D3557; border-radius: 50%; padding: 10px 13px; }
.project-by-label { font-size: 10.5px; color: #8FA3B8; letter-spacing: 0.03em; }
.project-by-name { font-size: 14.5px; font-weight: 700; color: #1D3557; }
.project-by-sub { font-size: 11px; color: #8FA3B8; }

.section-divider { height: 18px; }

.placeholder-box, .upload-placeholder, .pipeline-placeholder-box { background: #F4F8FC; border: 1.5px dashed #C7D3DE; border-radius: 12px; padding: 20px; text-align: center; color: #8FA3B8; font-size: 13px; }

.upload-instructions { font-size: 13px; color: #6C87A3; margin-bottom: 12px; text-align: center; }
.upload-success { margin-top: 10px; background: #E7F7EE; color: #1B7F4C; border-radius: 9px; padding: 9px 12px; font-size: 12.5px; font-weight: 600; text-align: center; white-space: nowrap; }
.pipeline-step-title { font-size: 12px; font-weight: 700; color: #457B9D; margin-bottom: 8px; text-align: center; white-space: nowrap; }

/* PREDICTION CARD LAYOUT FIXES */
.prediction-result-box { background: linear-gradient(135deg, #F0FAF4, #E9F8EF); border-radius: 14px; padding: 18px; box-shadow: inset 0 0 0 1px rgba(42,157,143,0.08); }
.prediction-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.prediction-group { display: flex; align-items: center; gap: 12px; flex-shrink: 1; min-width: 0; }
.prediction-label { font-size: 9.5px; color: #6C87A3; font-weight: 700; letter-spacing: 0.05em; white-space: nowrap; }
.prediction-value { font-size: 17px; font-weight: 800; color: #1D3557; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prediction-confidence { text-align: right; flex-shrink: 0; }
.prediction-confidence-value { font-size: 17px; font-weight: 800; color: #1B7F4C; margin-top: 2px; white-space: nowrap; }

.severity-row { display: flex; justify-content: space-between; align-items: center; margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(29,53,87,0.06); }
.severity-badge { color: #FFFFFF; font-weight: 700; font-size: 12px; padding: 5px 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.12); white-space: nowrap; }
.confidence-progress-label { font-size: 12px; color: #6C87A3; margin: 12px 0 4px 0; font-weight: 600; }

.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.prob-label { width: 72px; font-size: 12.5px; color: #1D3557; font-weight: 600; white-space: nowrap; }
.prob-bar-track { flex: 1; background: #EEF2F6; border-radius: 6px; height: 11px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }
.prob-value { width: 58px; font-size: 12px; color: #6C87A3; text-align: right; font-weight: 600; }

.gradcam-legend { display: flex; align-items: center; gap: 10px; font-size: 11px; color: #6C87A3; margin-top: 10px; font-weight: 600; }
.gradcam-gradient-bar { flex: 1; height: 8px; border-radius: 4px; background: linear-gradient(to right, #2E86DE, #2A9D8F, #F4A261, #E63946); }

.metric-card { background: #FBFDFF; border-radius: 12px; padding: 14px 10px; text-align: center; box-shadow: 0 2px 8px rgba(29,53,87,0.06); border: 1px solid rgba(29,53,87,0.05); margin-bottom: 12px; }
.metric-icon { font-size: 19px; }
.metric-label { font-size: 10.5px; color: #6C87A3; font-weight: 700; margin: 6px 0; letter-spacing: 0.03em; }
.metric-value { font-size: 19px; font-weight: 800; }

.summary-point { font-size: 12.5px; color: #1D3557; padding: 7px 0; border-bottom: 1px solid rgba(29,53,87,0.05); }
.summary-point:last-child { border-bottom: none; }
.summary-check { color: #FFFFFF; background: #2A9D8F; font-weight: 800; margin-right: 10px; font-size: 10px; border-radius: 50%; padding: 2px 6px; }

.footer-block { background: #0D1B2A; color: #E0E6ED; border-radius: 14px; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; font-size: 13px; box-shadow: 0 4px 16px rgba(13,27,42,0.25); }
.footer-sub { font-size: 11px; color: #8FA3B8; }
div[data-testid="stProgress"] > div > div { border-radius: 10px; background-color: #2A9D8F !important; }
img { border-radius: 10px; }
</style>
"""

CLASS_DISPLAY_NAMES = {"acl": "ACL Tear", "lcl": "LCL Tear", "mcl": "MCL Tear", "pcl": "PCL Tear", "normal": "Normal"}
CLASS_BAR_COLORS = {"acl": "#E63946", "lcl": "#F4A261", "mcl": "#E9C46A", "pcl": "#8E44AD", "normal": "#2A9D8F"}
SEVERITY_COLORS = {"No Tear": "#2A9D8F", "Low": "#2E86DE", "Moderate": "#F4A261", "High": "#E63946"}

NAV_ITEMS = ["Home", "Upload MRI", "Preprocessing", "Prediction", "Probability", "Grad-CAM", "Performance", "Confusion Matrix", "Project Summary"]
NAV_ICONS = ["house", "cloud-upload", "image", "activity", "bar-chart", "fire", "graph-up", "grid-3x3", "info-circle"]
NAV_ANCHORS = {k: f"section-{k.lower().replace(' ', '-')}" for k in NAV_ITEMS}

DATASET_SPLIT_COUNTS = {"Training Images": 1674, "Validation Images": 358, "Testing Images": 361}

# ==============================================================================
# HELPER: LOAD IMAGE AS BASE64 FOR HTML INJECTION
# ==============================================================================
def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ==============================================================================
# PAGE CONFIG 
# ==============================================================================

def configure_page():
    st.set_page_config(page_title="Knee AI - Smart Diagnosis", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
    st.markdown(INLINE_CSS, unsafe_allow_html=True) 

# ==============================================================================
# CACHED RESOURCE LOADERS
# ==============================================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_data(show_spinner=False)
def load_results(results_path: str):
    if not os.path.exists(results_path):
        return None
    with open(results_path, "r") as f:
        return json.load(f)

def find_gradcam_target_layer(model):
    if hasattr(model, "features"):
        return model.features[-1]
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv

# ==============================================================================
# PREPROCESSING PIPELINE
# ==============================================================================

def preprocess_pipeline(pil_image: Image.Image):
    original_rgb = np.array(pil_image.convert("RGB"))
    resized_rgb = cv2.resize(original_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe_op.apply(gray)
    clahe_rgb = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2RGB)

    normalized = clahe_rgb.astype(np.float32) / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - imagenet_mean) / imagenet_std

    chw = normalized.transpose(2, 0, 1)
    tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0)

    return {
        "original": original_rgb,
        "resized": resized_rgb,
        "clahe": clahe_rgb,
        "tensor": tensor,
    }

# ==============================================================================
# PREDICTION & GRAD-CAM
# ==============================================================================

def run_prediction(model, input_tensor: torch.Tensor):
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    probs_np = probabilities.detach().cpu().numpy()[0]
    predicted_idx = int(np.argmax(probs_np))
    return predicted_idx, probs_np

def get_severity(predicted_class: str, confidence: float):
    if predicted_class == "normal":
        return "No Tear"
    if confidence > 0.90:
        return "High"
    elif confidence >= 0.75:
        return "Moderate"
    return "Low"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int, size: int):
        self.model.zero_grad()
        input_tensor = input_tensor.to(DEVICE).clone().requires_grad_(True)
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam).cpu().numpy()
        cam = cv2.resize(cam, (size, size))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

def overlay_gradcam(original_rgb: np.ndarray, cam: np.ndarray, size: int):
    base = cv2.resize(original_rgb, (size, size))
    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base, 0.55, heatmap_color, 0.45, 0)
    return base, heatmap_color, overlay

# ==============================================================================
# UI HELPER COMPONENTS
# ==============================================================================

def render_metric_card(col, label: str, value: str, icon: str, color: str):
    col.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {color};">
            <div class="metric-icon" style="color:{color};">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def render_probability_bar(class_key: str, probability: float):
    display_name = CLASS_DISPLAY_NAMES.get(class_key, class_key.upper())
    color = CLASS_BAR_COLORS.get(class_key, "#457B9D")
    pct = probability * 100
    st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">{display_name}</div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width:{pct:.2f}%; background-color:{color};"></div>
            </div>
            <div class="prob-value">{pct:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

def render_pipeline_step(col, title: str, image: np.ndarray, step_no: int):
    col.markdown(f"<div class='pipeline-step-title'>{step_no}. {title}</div>", unsafe_allow_html=True)
    col.image(image, use_container_width=True)

def render_arrow(col):
    col.markdown(
        """
        <div style='display: flex; justify-content: center; font-size: 32px; color: #2563eb; font-weight: 900; margin-top: 40px;'>
            →
        </div>
        """,
        unsafe_allow_html=True
    )

def render_confusion_matrix(matrix: list, classes: list):
    matrix_np = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(matrix_np, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([c.upper() for c in classes], fontsize=9)
    ax.set_yticklabels([c.upper() for c in classes], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=10, fontweight="bold")

    max_val = matrix_np.max() if matrix_np.size else 1
    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            val = matrix_np[i, j]
            text_color = "white" if val > max_val * 0.5 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.patch.set_facecolor('white')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def anchor(name: str):
    st.markdown(f"<div id='{name}'></div>", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================

def render_sidebar(results: dict, logo_src: str):
    with st.sidebar:
        st.markdown(f"""
            <div class="sidebar-brand">
                <img src="{logo_src}" alt="Logo" style="height: 38px; width: auto; object-fit: contain; border-radius: 6px;">
                <div>
                    <div class="sidebar-brand-title">Knee AI</div>
                    <div class="sidebar-brand-subtitle">Smart Diagnosis</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None, options=NAV_ITEMS, icons=NAV_ICONS, default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"font-size": "16px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "2px 0"},
                "nav-link-selected": {"background-color": "#1D3557"},
            },
        )

        st.markdown("<div class='sidebar-section-title'>MODEL INFORMATION</div>", unsafe_allow_html=True)
        model_info = {"Model": "EfficientNet-B0", "Framework": "PyTorch", "Device": str(DEVICE), "Image Size": f"{IMAGE_SIZE} x {IMAGE_SIZE}", "Classes": f"{NUM_CLASSES} Classes"}
        for k, v in model_info.items():
            st.markdown(f"<div class='sidebar-info-row'><span>{k}</span><span>{v}</span></div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section-title'>DATASET DETAILS</div>", unsafe_allow_html=True)
        for label, count in DATASET_SPLIT_COUNTS.items():
            st.markdown(f"<div class='sidebar-info-row'><span>{label}</span><span>{count}</span></div>", unsafe_allow_html=True)

        return selected

def scroll_to_anchor(anchor_id: str):
    st.markdown(f"""
        <script>
            const el = window.parent.document.getElementById("{anchor_id}");
            if (el) {{ el.scrollIntoView({{behavior: "smooth", block: "start"}}); }}
        </script>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    configure_page()
    results = load_results(RESULTS_PATH)
    
    # Generate Logo base64
    logo_base64 = get_image_base64(LOGO_PATH)
    logo_src = f"data:image/png;base64,{logo_base64}" if logo_base64 else ""

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "prediction_data" not in st.session_state:
        st.session_state.prediction_data = None

    selected_nav = render_sidebar(results, logo_src)

    # ---------------------------------------------------------------- HEADER
    anchor(NAV_ANCHORS["Home"])
    
    st.markdown(f"""
        <div class="header-block">
            <img src="{logo_src}" alt="Knee AI Logo" style="height: 75px; width: auto; object-fit: contain; margin-right: 8px;">
            <div style="flex: 1;">
                <div class="header-title">AI Assisted Knee Ligament</div>
                <div class="header-title">Tear Detection & Classification System</div>
                <div class="header-subtitle">Powered by EfficientNet-B0 &nbsp;•&nbsp; PyTorch &nbsp;•&nbsp; CLAHE &nbsp;•&nbsp; Focal Loss</div>
            </div>
            <div class="project-by-card">
                <div>
                    <div class="project-by-label" style="text-align: right;">Project By</div>
                    <div class="project-by-name" style="text-align: right;">Soumya Dixit</div>
                    <div class="project-by-sub" style="text-align: right;">Final Year Project</div>
                </div>
                <div class="project-by-icon">👤</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # -------------------------------------------------- UPLOAD + PREPROCESS ROW
    anchor(NAV_ANCHORS["Upload MRI"])
    
    upload_col, pipeline_col = st.columns([1.2, 2.6])

    with upload_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Upload Knee MRI Image</div>", unsafe_allow_html=True)
            st.markdown("<div class='upload-instructions'>Upload a Knee MRI image to start the analysis</div>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

            if uploaded_file is not None:
                st.session_state.uploaded_image = Image.open(uploaded_file)
                st.markdown("<div class='upload-success'>✓ Image Uploaded Successfully</div>", unsafe_allow_html=True)
            elif st.session_state.uploaded_image is None:
                st.markdown("<div class='upload-placeholder'>No image uploaded yet</div>", unsafe_allow_html=True)

    anchor(NAV_ANCHORS["Preprocessing"])
    with pipeline_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Image Preprocessing Pipeline</div>", unsafe_allow_html=True)

            if st.session_state.uploaded_image is not None:
                pipeline_data = preprocess_pipeline(st.session_state.uploaded_image)

                step_cols = st.columns([1, 0.15, 1, 0.15, 1, 0.15, 1])
                
                render_pipeline_step(step_cols[0], "Original MRI", pipeline_data["original"], 1)
                render_arrow(step_cols[1])
                
                render_pipeline_step(step_cols[2], "Resized", pipeline_data["resized"], 2)
                render_arrow(step_cols[3])
                
                render_pipeline_step(step_cols[4], "CLAHE Enhanced", pipeline_data["clahe"], 3)
                render_arrow(step_cols[5])

                model = load_model()
                target_layer = find_gradcam_target_layer(model)
                predicted_idx, probabilities = run_prediction(model, pipeline_data["tensor"])

                tensor_vis = pipeline_data["tensor"][0].permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                tensor_vis = np.clip((tensor_vis * std + mean) * 255.0, 0, 255).astype(np.uint8)
                
                render_pipeline_step(step_cols[6], "Normalized", tensor_vis, 4)

                predicted_class = CLASSES[predicted_idx]
                confidence = float(probabilities[predicted_idx])
                severity = get_severity(predicted_class, confidence)

                st.session_state.prediction_data = {
                    "pipeline": pipeline_data, "predicted_class": predicted_class,
                    "confidence": confidence, "severity": severity, "probabilities": probabilities,
                    "model": model, "target_layer": target_layer, "predicted_idx": predicted_idx,
                }
            else:
                step_cols = st.columns([1, 0.15, 1, 0.15, 1, 0.15, 1])
                step_titles = ["Original MRI", "Resized", "CLAHE Enhanced", "Normalized"]
                
                for i, idx in enumerate([0, 2, 4, 6]):
                    step_cols[idx].markdown(f"<div class='pipeline-step-title'>{i + 1}. {step_titles[i]}</div>", unsafe_allow_html=True)
                    step_cols[idx].markdown("<div class='pipeline-placeholder-box'>Awaiting upload</div>", unsafe_allow_html=True)
                
                render_arrow(step_cols[1])
                render_arrow(step_cols[3])
                render_arrow(step_cols[5])
                
                st.session_state.prediction_data = None

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # -------------------------------------------- PREDICTION | PROBABILITY | CAM ROW
    anchor(NAV_ANCHORS["Prediction"])
    
    # 5. Given the Prediction Column slightly more room relative to the others to prevent cramming
    pred_col, prob_col, cam_col = st.columns([1.6, 1.1, 1.3])

    pdata = st.session_state.prediction_data

    with pred_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Prediction Result</div>", unsafe_allow_html=True)
            if pdata is not None:
                severity_color = SEVERITY_COLORS.get(pdata["severity"], "#457B9D")
                display_name = CLASS_DISPLAY_NAMES.get(pdata["predicted_class"], pdata["predicted_class"].upper())
                confidence_pct = pdata["confidence"] * 100

                # Re-structured Flexbox: Logo+Text in one group to push the percentage entirely to the right
                st.markdown(f"""
                    <div class="prediction-result-box">
                        <div class="prediction-row">
                            <div class="prediction-group">
                                <img src="{logo_src}" alt="Logo" style="height: 40px; width: auto; object-fit: contain;">
                                <div>
                                    <div class="prediction-label">PREDICTION</div>
                                    <div class="prediction-value">{display_name}</div>
                                </div>
                            </div>
                            <div class="prediction-confidence">
                                <div class="prediction-label">CONFIDENCE</div>
                                <div class="prediction-confidence-value">{confidence_pct:.2f}%</div>
                            </div>
                        </div>
                        <div class="severity-row">
                            <span class="prediction-label">SEVERITY LEVEL</span>
                            <span class="severity-badge" style="background-color:{severity_color};">{pdata['severity']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<div class='confidence-progress-label'>Confidence Progress</div>", unsafe_allow_html=True)
                st.progress(min(max(pdata["confidence"], 0.0), 1.0))
            else:
                st.markdown("<div class='placeholder-box'>Upload an MRI image to generate a prediction.</div>", unsafe_allow_html=True)

    anchor(NAV_ANCHORS["Probability"])
    with prob_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Class Probabilities</div>", unsafe_allow_html=True)
            if pdata is not None:
                for class_key, prob in zip(CLASSES, pdata["probabilities"]):
                    render_probability_bar(class_key, float(prob))
            else:
                st.markdown("<div class='placeholder-box'>Class probabilities will appear here after upload.</div>", unsafe_allow_html=True)

    anchor(NAV_ANCHORS["Grad-CAM"])
    with cam_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Grad-CAM Visualization</div>", unsafe_allow_html=True)
            if pdata is not None and pdata["target_layer"] is not None:
                gradcam = GradCAM(pdata["model"], pdata["target_layer"])
                cam = gradcam.generate(pdata["pipeline"]["tensor"], pdata["predicted_idx"], IMAGE_SIZE)
                base_img, heatmap_img, overlay_img = overlay_gradcam(pdata["pipeline"]["original"], cam, IMAGE_SIZE)

                cam_left, cam_right = st.columns(2)
                cam_left.markdown("<div class='pipeline-step-title'>Original MRI</div>", unsafe_allow_html=True)
                cam_left.image(base_img, use_container_width=True)
                cam_right.markdown("<div class='pipeline-step-title'>Grad-CAM Heatmap</div>", unsafe_allow_html=True)
                cam_right.image(overlay_img, use_container_width=True)

                st.markdown("""
                    <div class="gradcam-legend">
                        <span>Low Attention</span>
                        <div class="gradcam-gradient-bar"></div>
                        <span>High Attention</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='placeholder-box'>Grad-CAM will be generated from the uploaded MRI.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ------------------------------------------- PERFORMANCE | CONFUSION | SUMMARY ROW
    anchor(NAV_ANCHORS["Performance"])
    perf_col, conf_col, summary_col = st.columns([1.2, 1.6, 1.2])

    with perf_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Model Performance</div>", unsafe_allow_html=True)
            if results is not None:
                metric_row1 = st.columns(2)
                metric_row2 = st.columns(2)
                render_metric_card(metric_row1[0], "ACCURACY", f"{results.get('accuracy', 0):.2f}%", "🎯", "#2E86DE")
                render_metric_card(metric_row1[1], "PRECISION", f"{results.get('precision', 0):.2f}%", "📐", "#2A9D8F")
                render_metric_card(metric_row2[0], "RECALL", f"{results.get('recall', 0):.2f}%", "🔁", "#F4A261")
                render_metric_card(metric_row2[1], "F1 SCORE", f"{results.get('f1', 0):.2f}%", "⭐", "#8E44AD")
            else:
                st.markdown(f"<div class='placeholder-box'>'{RESULTS_PATH}' not found.</div>", unsafe_allow_html=True)

    anchor(NAV_ANCHORS["Confusion Matrix"])
    with conf_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Confusion Matrix (Test Set)</div>", unsafe_allow_html=True)
            if results is not None and "confusion_matrix" in results:
                render_confusion_matrix(results["confusion_matrix"], CLASSES)
            else:
                st.markdown("<div class='placeholder-box'>Confusion matrix will be loaded from results.json.</div>", unsafe_allow_html=True)

    anchor(NAV_ANCHORS["Project Summary"])
    with summary_col:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Project Summary</div>", unsafe_allow_html=True)
            summary_points = [
                "Deep Learning based Knee Ligament Tear Detection",
                f"{NUM_CLASSES} Classes: " + ", ".join(c.upper() for c in CLASSES),
                "EfficientNet-B0 Architecture",
                "CLAHE Preprocessing",
                "Grad-CAM Explainability",
                "Fine-tuned for High Accuracy",
            ]
            for point in summary_points:
                st.markdown(f"<div class='summary-point'><span class='summary-check'>✓</span>{point}</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------- FOOTER
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class="footer-block">
            <div>🛡️ AI Assisted Knee Ligament Tear Detection & Classification System<br>
            <span class="footer-sub">Powered by Deep Learning • PyTorch • EfficientNet-B0
                     © 2026 Soumya Dixit</span></div>
            <div class="footer-sub">Empowering Healthcare with AI</div>
        </div>
    """, unsafe_allow_html=True)

    if selected_nav in NAV_ANCHORS:
        scroll_to_anchor(NAV_ANCHORS[selected_nav])

if __name__ == "__main__":
    main()