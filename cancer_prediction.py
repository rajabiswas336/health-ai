# cancer_prediction.py — v2.0
# Model: Raja336/skin-cancer-convnext (HuggingFace)
# Added: pure-PyTorch Grad-CAM + Monte Carlo Dropout uncertainty + safety flag

import numpy as np
from PIL import Image
import io
import streamlit as st

CLASS_LABELS = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)",
]

RISK_LEVELS = {
    "Actinic Keratoses (akiec)": "⚠️ Pre-cancerous",
    "Basal Cell Carcinoma (bcc)": "🔴 Malignant",
    "Benign Keratosis (bkl)": "🟢 Benign",
    "Dermatofibroma (df)": "🟢 Benign",
    "Melanoma (mel)": "🔴 Malignant — Urgent",
    "Melanocytic Nevi (nv)": "🟢 Benign (Mole)",
    "Vascular Lesions (vasc)": "🟡 Usually Benign",
}

# Classes that trigger the urgent safety warning in the UI
URGENT_CLASSES = {
    "Melanoma (mel)",
    "Basal Cell Carcinoma (bcc)",
    "Actinic Keratoses (akiec)",
}


@st.cache_resource(show_spinner=False)
def load_model():
    """Load ConvNextForImageClassification from HuggingFace Hub. Cached."""
    from transformers import ConvNextForImageClassification, ConvNextImageProcessor

    MODEL_ID = "Raja336/skin-cancer-convnext"
    model     = ConvNextForImageClassification.from_pretrained(MODEL_ID)
    processor = ConvNextImageProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


# ── Grad-CAM (pure PyTorch, no extra library) ─────────────────────────────────
def generate_gradcam(model, pixel_values_tensor, predicted_idx: int, img_pil: Image.Image) -> bytes:
    """
    Pure-PyTorch Grad-CAM on the last ConvNext stage.

    How it works:
      1. Register a forward hook on the last encoder stage to capture its output
         feature map (the activation).
      2. Run a FULL forward pass WITH gradients enabled (can't use torch.no_grad here).
      3. Back-propagate the score for the predicted class only.
      4. Pool the gradients over spatial dims → per-channel weights.
      5. Weighted-sum the activation channels → raw CAM.
      6. ReLU + normalise + resize to original image size.
      7. Apply a JET colormap and alpha-blend onto the original image.

    Returns PNG bytes ready for st.image().
    """
    import torch
    import cv2

    # --- 1. Hook to capture the last stage's output feature map ---
    activation = {}
    def forward_hook(module, input, output):
        activation["feat"] = output   # shape: (1, C, H', W')

    # ConvNextForImageClassification structure on HuggingFace:
    #   model.convnext.encoder.stages[-1]  ← last ConvNext stage
    hook_handle = model.convnext.encoder.stages[-1].register_forward_hook(forward_hook)

    # --- 2. Forward pass WITH gradients ---
    model.zero_grad()
    # pixel_values_tensor must NOT be inside torch.no_grad()
    pv = pixel_values_tensor.detach().requires_grad_(False)
    outputs = model(pixel_values=pv)

    # --- 3. Back-prop the predicted class score ---
    score = outputs.logits[0, predicted_idx]
    score.backward()

    # --- 4. Gradient of the score w.r.t. the feature map ---
    # We need gradients at the hook layer, so we use the logits gradient
    # channelled through the activation. Use register_full_backward_hook
    # on the same stage for cleaner access — but the simplest working
    # approach for ConvNext is to use the gradient of logits w.r.t.
    # the captured feature map via autograd.
    feat = activation["feat"]           # (1, C, H', W')

    # Re-run with requires_grad on the feature map itself
    hook_handle.remove()

    # Clean approach: re-register, this time capturing grads too
    saved = {}
    def full_hook(module, grad_input, grad_output):
        saved["grad"] = grad_output[0]  # (1, C, H', W')

    bwd_handle = model.convnext.encoder.stages[-1].register_full_backward_hook(full_hook)
    fwd_handle = model.convnext.encoder.stages[-1].register_forward_hook(
        lambda m, i, o: saved.update({"feat": o})
    )

    model.zero_grad()
    outputs2 = model(pixel_values=pv)
    outputs2.logits[0, predicted_idx].backward()

    bwd_handle.remove()
    fwd_handle.remove()

    feat_map = saved["feat"][0].detach()    # (C, H', W')
    grad_map = saved["grad"][0].detach()    # (C, H', W')

    # --- 5. Global-average-pool gradients → weights, then weighted sum ---
    weights = grad_map.mean(dim=(1, 2))     # (C,)
    cam = (weights[:, None, None] * feat_map).sum(dim=0)  # (H', W')

    # --- 6. ReLU, normalise, resize ---
    cam = torch.relu(cam).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    orig_w, orig_h = img_pil.size
    cam_resized = cv2.resize(cam, (orig_w, orig_h))

    # --- 7. Colormap + blend ---
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig_arr = np.array(img_pil.convert("RGB").resize((orig_w, orig_h)))
    overlay  = (0.55 * orig_arr + 0.45 * heatmap).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return buf.getvalue()


# ── Monte Carlo Dropout uncertainty ──────────────────────────────────────────
def mc_dropout_uncertainty(model, inputs: dict, n_passes: int = 20) -> dict:
    """
    Run N stochastic forward passes with dropout active.
    Returns per-class mean probability and std (uncertainty).

    Why this works: ConvNext uses dropout in its classifier head.
    By calling model.train() we re-enable those dropout layers,
    then run multiple passes. The std across passes = model uncertainty.
    A std > 0.10 on the top class means the model is unsure.
    """
    import torch

    # Enable dropout layers only (not BatchNorm)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            out   = model(**inputs)
            probs = torch.nn.functional.softmax(out.logits, dim=-1)[0]
            all_probs.append(probs.numpy())

    # Restore eval mode
    model.eval()

    stacked    = np.array(all_probs)          # (n_passes, 7)
    mean_probs = stacked.mean(axis=0)         # (7,)
    std_probs  = stacked.std(axis=0)          # (7,)  ← uncertainty

    return {
        CLASS_LABELS[i]: {
            "mean": float(mean_probs[i]),
            "std":  float(std_probs[i]),
        }
        for i in range(len(CLASS_LABELS))
    }


# ── Main prediction function ──────────────────────────────────────────────────
def predict_skin_cancer(image_bytes: bytes) -> dict:
    """
    Full pipeline: preprocess → predict → Grad-CAM → uncertainty.

    Returns dict with keys:
        "class"         — predicted class label
        "confidence"    — softmax probability of top class (0–1)
        "risk"          — risk string e.g. '🔴 Malignant — Urgent'
        "all_probs"     — {class: probability} for all 7 classes
        "gradcam_bytes" — PNG bytes of heatmap overlay (or None on error)
        "uncertainty"   — {class: {mean, std}} from MC Dropout (or None)
        "is_urgent"     — True if class is malignant/pre-cancerous
    """
    import torch

    model, processor = load_model()

    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    # ── Standard inference (no_grad for speed) ──
    with torch.no_grad():
        outputs       = model(**inputs)
        probs         = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        predicted_idx = int(torch.argmax(probs).item())
        confidence    = float(probs[predicted_idx].item())

    predicted_class = CLASS_LABELS[predicted_idx]

    all_probs = {
        CLASS_LABELS[i]: float(probs[i].item())
        for i in range(len(CLASS_LABELS))
    }

    # ── Grad-CAM ──
    try:
        gradcam_bytes = generate_gradcam(
            model,
            inputs["pixel_values"],
            predicted_idx,
            img,
        )
    except Exception as e:
        print(f"[Grad-CAM] failed (safe to ignore): {e}")
        gradcam_bytes = None

    # ── MC Dropout uncertainty ──
    try:
        uncertainty = mc_dropout_uncertainty(model, inputs, n_passes=20)
    except Exception as e:
        print(f"[MC Dropout] failed (safe to ignore): {e}")
        uncertainty = None

    return {
        "class":         predicted_class,
        "confidence":    confidence,
        "risk":          RISK_LEVELS.get(predicted_class, "Unknown"),
        "all_probs":     all_probs,
        "gradcam_bytes": gradcam_bytes,
        "uncertainty":   uncertainty,
        "is_urgent":     predicted_class in URGENT_CLASSES,
    }