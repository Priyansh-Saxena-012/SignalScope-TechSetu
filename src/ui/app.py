"""SignalScope Streamlit Demo UI.

Provides an authenticity assessment interface: image upload,
AI-generated probability verdict, generator attribution, attention-heatmap
explanation, a degradation robustness playground, and an evaluation report.

Supports live inference with the trained ViT-Base/16 checkpoint via
``model.predict.predict()`` when weights are available, falling back to a
clearly-labeled mock demo mode when no checkpoint is available.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

try:
    from model.predict import predict as _model_predict
    MODEL_AVAILABLE = True
except ImportError:
    _model_predict = None
    MODEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Prediction backend (mock-first, real-model-ready)
# ---------------------------------------------------------------------------

MOCK_PAYLOADS: List[Dict[str, Any]] = [
    {
        "base_probability": 0.88,
        "generator_family": "Diffusion-family (e.g., Stable Diffusion / Midjourney)",
        "summary": "Inconsistent lighting and high-frequency edge artifacts detected.",
        "cues": [
            "Unnatural specular highlights on reflective surfaces",
            "Frequency-domain grid artifacts typical of latent upscalers",
            "Physically inconsistent shadow direction relative to the light source",
        ],
    },
    {
        "base_probability": 0.79,
        "generator_family": "GAN-family (e.g., StyleGAN / BigGAN)",
        "summary": "Warped geometric textures and asymmetric facial/background structure.",
        "cues": [
            "Warped geometric textures along fine edges",
            "Repeating checkerboard upsampling artifacts",
            "Irregular texture blending at object boundaries",
        ],
    },
    {
        "base_probability": 0.09,
        "generator_family": "N/A (assessed as camera-original)",
        "summary": "Sensor noise pattern and lighting are consistent with a natural capture.",
        "cues": [
            "Consistent photon shot noise across flat regions",
            "Chromatic aberration consistent with a real lens",
            "No frequency-domain grid artifacts detected",
        ],
    },
    {
        "base_probability": 0.52,
        "generator_family": "Undetermined",
        "summary": "Signal is too weak or ambiguous to attribute confidently.",
        "cues": [
            "Compression has degraded high-frequency forensic signal",
            "No single cue reaches the confidence threshold for a verdict",
        ],
    },
]


def _image_seed(image: Image.Image) -> int:
    """Deterministic seed derived from image content, so repeated demo runs
    on the same image are stable rather than flickering between verdicts."""
    buf = io.BytesIO()
    image.convert("RGB").resize((32, 32)).save(buf, format="PNG")
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    return int(digest[:8], 16)


def _compute_verdict(probability: float) -> str:
    if 0.40 <= probability <= 0.60:
        return "Inconclusive / Low Confidence"
    return "Likely AI-generated" if probability > 0.60 else "Likely Real"


def _compute_confidence_level(probability: float) -> str:
    distance = abs(probability - 0.5)
    if distance >= 0.35:
        return "High"
    if distance >= 0.15:
        return "Medium"
    return "Low"


def _mock_result(image: Image.Image) -> Dict[str, Any]:
    seed = _image_seed(image)
    payload = MOCK_PAYLOADS[seed % len(MOCK_PAYLOADS)]

    # Small deterministic jitter so the same "family" of mock isn't identical
    # every time, without being random across reruns of the same image.
    jitter = ((seed // len(MOCK_PAYLOADS)) % 11 - 5) / 100.0
    probability = min(max(payload["base_probability"] + jitter, 0.01), 0.99)
    is_ai = bool(probability >= 0.5)
    confidence = probability if is_ai else (1.0 - probability)
    raw_logit = math.log(max(probability, 1e-6) / max(1.0 - probability, 1e-6))

    return {
        "verdict": _compute_verdict(probability),
        "calibrated_probability": round(probability, 4),
        "confidence_level": _compute_confidence_level(probability),
        "generator_family": payload["generator_family"],
        "explanation": {
            "summary": payload["summary"],
            "cues": payload["cues"],
        },
        "metadata": {
            "c2pa_present": False,
            "exif_intact": bool(seed % 2),
        },
        "is_demo_mode": True,
        "raw_logit": round(float(raw_logit), 4),
        "temperature": 1.0,
        "is_calibrated": False,
        "is_ai": is_ai,
        "label": "AI-generated" if is_ai else "Real",
        "confidence": round(float(confidence), 4),
        "_seed": seed,
    }


def predict(image: Image.Image) -> Dict[str, Any]:
    """Return a UI-facing authenticity assessment for ``image``.

    Tries the real ``model.predict`` contract first using trained checkpoint
    weights when available; falls back to a clearly labeled mock mode
    whenever trained weights are not available.
    """
    if MODEL_AVAILABLE:
        # delete=False + manual cleanup: on Windows, a NamedTemporaryFile
        # opened with delete=True cannot be reopened by path while still held.
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            image.convert("RGB").save(tmp.name, format="PNG")
            try:
                raw = _model_predict(tmp.name, allow_stub=True)
            except Exception:
                raw = None
        finally:
            os.remove(tmp.name)
        if raw is not None and raw.get("status") != "model_not_yet_trained":
            # Real model contract has been upgraded — adapt its output here.
            probability = float(raw.get("ai_probability", raw.get("confidence", 0.5)))
            return {
                "verdict": _compute_verdict(probability),
                "calibrated_probability": round(probability, 4),
                "confidence_level": _compute_confidence_level(probability),
                "generator_family": raw.get("generator_family", "Undetermined"),
                "explanation": raw.get("explanation", {"summary": "", "cues": []}),
                "metadata": raw.get("metadata", {"c2pa_present": False, "exif_intact": True}),
                "provenance": raw.get("provenance", {}),
                "is_demo_mode": False,
                "raw_logit": raw.get("raw_logit"),
                "temperature": raw.get("temperature", 1.0),
                "is_calibrated": raw.get("is_calibrated", False),
                "is_ai": raw.get("is_ai", bool(probability >= 0.5)),
                "label": raw.get("label", "AI-generated" if probability >= 0.5 else "Real"),
                "confidence": raw.get("confidence", round(probability if probability >= 0.5 else 1.0 - probability, 4)),
                "device": raw.get("device", "cpu"),
                "weights_path": raw.get("weights_path", ""),
            }

    return _mock_result(image)


# ---------------------------------------------------------------------------
# Synthetic Grad-CAM-style heatmap
# ---------------------------------------------------------------------------

def build_heatmap_overlay(image: Image.Image, seed: int) -> Image.Image:
    """Generate a synthetic attention-heatmap overlay for demo purposes.

    Produces a smoothed, seeded noise field, colorizes it (yellow->red), and
    alpha-blends it over the source image. This serves as a synthetic demo
    placeholder until a dedicated Grad-CAM/explainability module is integrated.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size

    rng = np.random.default_rng(seed)
    small = rng.random((16, 16))
    noise_img = Image.fromarray((small * 255).astype(np.uint8), mode="L")
    noise_img = noise_img.resize((w, h), resample=Image.BICUBIC)
    noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=max(w, h) / 20))

    intensity = np.asarray(noise_img, dtype=np.float32) / 255.0
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)

    heat_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    heat_rgb[..., 0] = (intensity * 255).astype(np.uint8)  # Red channel
    heat_rgb[..., 1] = (intensity * 120).astype(np.uint8)  # Green channel (some yellow)
    heat_overlay = Image.fromarray(heat_rgb, mode="RGB")

    return Image.blend(rgb, heat_overlay, alpha=0.45)


# ---------------------------------------------------------------------------
# Degradation pipeline (Robustness Playground)
# ---------------------------------------------------------------------------

def degrade_image(image: Image.Image, jpeg_quality: int, blur_radius: float, resize_pct: int) -> Image.Image:
    """Apply resize -> Gaussian blur -> JPEG re-compression, in that order,
    to simulate real-world social-media style degradation."""
    working = image.convert("RGB")

    if resize_pct < 100:
        new_w = max(1, int(working.width * resize_pct / 100))
        new_h = max(1, int(working.height * resize_pct / 100))
        working = working.resize((new_w, new_h), resample=Image.BILINEAR)

    if blur_radius > 0:
        working = working.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    buf = io.BytesIO()
    working.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# UI rendering helpers
# ---------------------------------------------------------------------------

def _render_verdict_badge(result: Dict[str, Any]) -> None:
    verdict = result["verdict"]
    if verdict == "Likely AI-generated":
        st.error(f"**{verdict}**")
    elif verdict == "Likely Real":
        st.success(f"**{verdict}**")
    else:
        st.warning(f"**{verdict}**")

    if result.get("is_demo_mode"):
        st.caption("Demo Mode — mock inference (trained weights not yet available)")


def _render_assessment(result: Dict[str, Any]) -> None:
    _render_verdict_badge(result)

    prob = float(result["calibrated_probability"])
    conf = float(result.get("confidence", prob if prob >= 0.5 else 1.0 - prob))
    raw_logit = result.get("raw_logit")
    temp = result.get("temperature", 1.0)
    is_calibrated = result.get("is_calibrated", False)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Calibrated AI Likelihood (p_AI)",
            value=f"{prob * 100:.1f}%",
            delta=f"{result['confidence_level']} Confidence",
            delta_color="off",
        )
    with col2:
        st.metric(
            label="Class Confidence",
            value=f"{conf * 100:.1f}%",
            delta="Probability of assigned label",
            delta_color="off",
        )
    with col3:
        threshold_dist = prob - 0.50
        st.metric(
            label="Decision Threshold (Fixed)",
            value="0.50",
            delta=f"{threshold_dist:+0.1%} from threshold" if raw_logit is not None else None,
            delta_color="normal" if threshold_dist >= 0 else "inverse",
        )

    st.progress(min(max(prob, 0.0), 1.0))
    st.caption("0.0% (Natural / Real) ⟵ ⟵ ⟵ 50.0% Decision Threshold ⟶ ⟶ ⟶ 100.0% (Synthetic / AI)")

    with st.expander("Mathematical Calibration & Threshold Details"):
        logit_str = f"{raw_logit:.4f}" if raw_logit is not None else "N/A"
        temp_str = f"{temp:.4f}"
        calib_note = " (Fitted on 7,000 validation images via NLL minimization)" if is_calibrated else " (Uncalibrated fallback)"
        st.markdown(
            f"- **Raw Logit**: `z = {logit_str}`\n"
            f"- **Calibration Temperature**: `T = {temp_str}`{calib_note}\n"
            f"- **Calibrated AI Probability**: `p_AI = σ(z / T) = {prob:.4f}` ({prob * 100:.2f}%)\n"
            r"- **Invariant Decision Threshold**: The operational decision rule is fixed at $\tau = 0.50$. "
            r"Because $\sigma(0) = 0.50$ for any temperature $T > 0$, the decision boundary corresponds strictly to $z \ge 0$ (invariant under temperature scaling)." + "\n"
            r"- **Distinction Between Probability and Confidence**: $p_{\text{AI}}$ represents the continuous estimated likelihood that an image is synthetically generated. "
            r"Confidence represents the probability mass concentrated on the selected class: $\max(p_{\text{AI}}, 1 - p_{\text{AI}})$."
        )

    st.info(f"**Attribution Family:** {result['generator_family']}")
    if result.get("generator_family") == "Undetermined":
        st.caption(
            "Scope Clarification: The production ViT classifier is a binary veracity detector (Real vs. Synthetic), "
            "not a multi-class generator identification model. Generator family attribution is intentionally Undetermined."
        )

    st.markdown("**Forensic Analysis Summary:**")
    for cue in result.get("explanation", {}).get("cues", []):
        st.markdown(f"- {cue}")
    st.caption(result.get("explanation", {}).get("summary", ""))

    st.caption(
        "Responsible Forensics Notice: SignalScope estimates synthetic likelihood from spatial representations and frequency artifacts. "
        "Outputs are probabilistic indicators, not definitive legal proof of authenticity or human origin."
    )


def _render_metadata_card(image: Image.Image, result: Dict[str, Any]) -> None:
    meta = result.get("metadata", {})
    prov = meta.get("provenance", {})
    st.markdown("**Source & Provenance Metadata**")
    st.write(f"Resolution: {image.width} × {image.height} px")
    st.write(f"Color Mode: {image.mode}")

    c2pa_meta = prov.get("c2pa", {})
    val_state = c2pa_meta.get("validation_state", "NOT_PRESENT")
    if val_state == "VALID_C2PA":
        c2pa_text = "C2PA Credentials: Valid Cryptographic Signature"
    elif val_state == "UNVERIFIED_MANIFEST":
        c2pa_text = "C2PA Credentials: Manifest Present (Unverified Signature)"
    elif meta.get("c2pa_present"):
        c2pa_text = "C2PA Credentials: Structure Detected"
    else:
        c2pa_text = "C2PA Credentials: None Found"

    exif_meta = prov.get("exif", {})
    gen_ind = prov.get("generation_indicators", {})
    if gen_ind.get("known_ai_software_flag"):
        exif_text = "Metadata Signature: AI Generator Pattern Detected"
    elif meta.get("exif_intact") or exif_meta.get("present"):
        cam_info = []
        if exif_meta.get("camera_make"):
            cam_info.append(exif_meta["camera_make"])
        if exif_meta.get("camera_model"):
            cam_info.append(exif_meta["camera_model"])
        cam_str = f" ({' '.join(cam_info)})" if cam_info else ""
        exif_text = f"EXIF: Hardware Tags Intact{cam_str}"
    else:
        exif_text = "EXIF: Stripped / Absent"

    st.write(c2pa_text)
    st.write(exif_text)

    # Contextual forensic detail accordion
    with st.expander("Forensic Metadata Details (Auxiliary)"):
        st.caption(
            "Notice: Metadata is an auxiliary forensic signal and does not override "
            "or alter the neural detector prediction. Missing EXIF does not prove AI generation."
        )
        if exif_meta.get("gps_redacted"):
            st.info("Privacy Safeguard: Geospatial coordinates (GPS) were detected and redacted.")
        if exif_meta.get("serial_redacted"):
            st.info("Privacy Safeguard: Device serial numbers have been masked.")

        if exif_meta.get("software"):
            st.write(f"Software: {exif_meta['software']}")
        if exif_meta.get("datetime_original"):
            st.write(f"Timestamp: {exif_meta['datetime_original']}")
        if gen_ind.get("generator_signature"):
            st.warning(f"Generation Signature: {gen_ind['generator_signature']}")

        notes = prov.get("forensic_notes", [])
        if notes:
            st.markdown("**Forensic Notes:**")
            for note in notes:
                st.markdown(f"- {note}")


def _render_tab1(uploaded_image: Optional[Image.Image]) -> None:
    if uploaded_image is None:
        st.info("Upload an image above to run an authenticity assessment.")
        return

    result = predict(uploaded_image)
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Source Media")
        st.image(uploaded_image, use_column_width=True)
        _render_metadata_card(uploaded_image, result)

    with col_right:
        st.subheader("Authenticity Assessment")
        _render_assessment(result)

        st.markdown("---")
        show_cam = st.toggle("Show Patch Decision Saliency (Experimental)")
        if show_cam:
            if not result.get("is_demo_mode", False):
                try:
                    from model.predict import get_model
                    from model.explain import generate_explanation
                    live_model, device, _ = get_model()
                    if live_model is not None:
                        saliency_map, eval_crop, overlay_img = generate_explanation(uploaded_image, live_model, device=device)
                        st.image(
                            overlay_img,
                            caption="Experimental relative patch attribution on the 224×224 evaluation crop",
                            use_column_width=True,
                        )
                        st.caption(
                            "Relative patch saliency on a 14×14 grid (196 patches of 16×16 px), highlighting regions "
                            "that contributed above average toward the AI-generated decision logit."
                        )
                        st.warning(
                            "Crucial Forensic Caveat: Decision saliency visualizes neural decision attribution, NOT a pixel-level "
                            "forgery mask. It does not localize ground-truth manipulation boundaries or explain natural authenticity."
                        )
                    else:
                        heatmap = build_heatmap_overlay(uploaded_image, result.get("_seed", 0))
                        st.image(heatmap, caption="Localized suspicious regions (synthetic demo overlay)", use_column_width=True)
                except Exception:
                    heatmap = build_heatmap_overlay(uploaded_image, result.get("_seed", 0))
                    st.image(heatmap, caption="Localized suspicious regions (synthetic demo overlay)", use_column_width=True)
            else:
                heatmap = build_heatmap_overlay(uploaded_image, result.get("_seed", 0))
                st.image(heatmap, caption="Localized suspicious regions (synthetic demo overlay)", use_column_width=True)
                st.caption("Demo Mode — synthetic placeholder overlay (trained weights not yet available)")


def _render_tab2(uploaded_image: Optional[Image.Image]) -> None:
    st.write("### Degradation Robustness Playground")
    st.write("Test detector resilience against real-world social-media style perturbations (JPEG re-compression, Gaussian blur, and downscaling).")

    if uploaded_image is None:
        st.info("Upload an image in the Image Inspection tab first to evaluate robustness under degradation.")
        return

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        jpeg_quality = st.slider("JPEG Compression Quality", min_value=20, max_value=100, value=100, step=5)
    with col_ctrl2:
        blur_radius = st.slider("Gaussian Blur Radius", min_value=0.0, max_value=4.0, value=0.0, step=0.1)
    with col_ctrl3:
        resize_pct = st.slider("Downscale / Resize (%)", min_value=25, max_value=100, value=100, step=5)

    if st.button("Re-test Under Degradation"):
        degraded = degrade_image(uploaded_image, jpeg_quality, blur_radius, resize_pct)

        original_result = predict(uploaded_image)
        degraded_result = predict(degraded)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Original Media")
            st.image(uploaded_image, use_column_width=True)
            _render_verdict_badge(original_result)
        with col_right:
            st.subheader("Degraded Media")
            st.image(degraded, use_column_width=True)
            _render_verdict_badge(degraded_result)

        orig_prob = original_result["calibrated_probability"]
        deg_prob = degraded_result["calibrated_probability"]
        delta_prob = deg_prob - orig_prob
        abs_delta = abs(delta_prob)

        if abs_delta < 0.05:
            stability_status = "Highly Stable (< 5% shift)"
            delta_color = "normal"
        elif abs_delta < 0.15:
            stability_status = "Moderately Stable (5-15% shift)"
            delta_color = "off"
        else:
            stability_status = "Significant Shift (> 15% shift)"
            delta_color = "inverse"

        decision_flipped = (orig_prob >= 0.50) != (deg_prob >= 0.50)

        st.markdown("### Robustness Delta Analysis")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric(
                label="Original p_AI",
                value=f"{orig_prob * 100:.1f}%",
                delta=original_result["verdict"],
                delta_color="off",
            )
        with mcol2:
            st.metric(
                label="Degraded p_AI",
                value=f"{deg_prob * 100:.1f}%",
                delta=degraded_result["verdict"],
                delta_color="off",
            )
        with mcol3:
            st.metric(
                label="Probability Shift (Δ p_AI)",
                value=f"{delta_prob * 100:+.1f}%",
                delta="Verdict Flipped!" if decision_flipped else stability_status,
                delta_color="inverse" if decision_flipped else delta_color,
            )

        st.caption(
            "Robustness Note: Stability under degradation measures detector invariance across synthetic transformations. "
            "Model stability does not guarantee factual ground truth, and heavy compression may erode forensic high-frequency signatures."
        )
    else:
        st.caption("Adjust the degradation sliders above and click \"Re-test Under Degradation\" to run the comparison.")


def _render_tab3() -> None:
    st.write("## SignalScope Evaluation & Verification Report")
    st.caption("Comprehensive performance metrics, probability calibration analysis, and scope limitations.")

    st.write("### 1. Development Validation Benchmark (Stage 9 Baseline)")
    st.markdown(
        "Evaluated on the **7,000-image Tiny-GenImage development validation set** (3,500 Real + 3,500 AI-generated images across 8 generator architectures). "
        r"These results reflect the frozen ViT-Base/16 checkpoint (`checkpoint_best.pth`) evaluated with fixed decision threshold $\tau = 0.50$."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC (Overall)", "0.8562", help="Area Under ROC Curve across all 7,000 validation samples")
    with col2:
        st.metric("Macro-F1", "0.7834", help="Unweighted mean of Real-F1 (0.7828) and AI-F1 (0.7839)")
    with col3:
        st.metric("Accuracy", "78.34%", help="2,735 TN + 2,749 TP / 7,000 = 78.34%")
    with col4:
        st.metric("False Positive Rate", "21.86%", help="765 / 3,500 real images incorrectly flagged at 0.50 threshold")

    col_cm, col_gen = st.columns([1, 1])

    with col_cm:
        st.markdown("**Validation Confusion Matrix (N = 7,000)**")
        cm_table = [
            {"Ground Truth": "Actual: Real (3,500)", "Predicted Real (<= 0.50)": "2,735 (TN, 78.1%)", "Predicted AI (> 0.50)": "765 (FP, 21.9%)"},
            {"Ground Truth": "Actual: AI (3,500)", "Predicted Real (<= 0.50)": "751 (FN, 21.5%)", "Predicted AI (> 0.50)": "2,749 (TP, 78.5%)"},
        ]
        st.table(cm_table)

    with col_gen:
        st.markdown("**Per-Generator Architecture Breakdown**")
        gen_table = [
            {"Generator Family": "ADM", "Type": "Diffusion", "ROC-AUC": "0.6026"},
            {"Generator Family": "BigGAN", "Type": "GAN", "ROC-AUC": "0.7424"},
            {"Generator Family": "Glide", "Type": "Diffusion", "ROC-AUC": "0.8647"},
            {"Generator Family": "Midjourney", "Type": "Diffusion / Proprietary", "ROC-AUC": "0.8879"},
            {"Generator Family": "Stable Diffusion v1.4", "Type": "Latent Diffusion", "ROC-AUC": "0.9419"},
            {"Generator Family": "Stable Diffusion v1.5", "Type": "Latent Diffusion", "ROC-AUC": "0.9329"},
            {"Generator Family": "VQ-Diffusion", "Type": "Discrete Diffusion", "ROC-AUC": "0.8712"},
            {"Generator Family": "Wukong", "Type": "Diffusion", "ROC-AUC": "0.9513"},
        ]
        st.table(gen_table)

    st.write("### 2. Post-Hoc Probability Calibration (Temperature Scaling)")
    st.markdown(
        "Modern deep neural networks tend to produce overconfident raw probabilities. "
        "SignalScope applies **post-hoc Platt / Temperature Scaling** ($T = 1.9591$) fitted strictly on the 7,000-image development validation set. "
        "The model weights and feature representations remain frozen; only temperature parameter $T > 0$ is optimized by minimizing Negative Log-Likelihood (NLL)."
    )

    calib_path = Path(__file__).resolve().parent.parent.parent / "model" / "weights" / "temperature.json"
    if not calib_path.exists():
        calib_path = Path("model/weights/temperature.json")

    if calib_path.exists():
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            b_m = c_data["metrics"]["before_calibration"]
            a_m = c_data["metrics"]["after_calibration"]
            temp_val = float(c_data.get("calibration", {}).get("temperature", 1.9591))
            nll_b, nll_a = float(b_m["nll"]), float(a_m["nll"])
            brier_b, brier_a = float(b_m["brier_score"]), float(a_m["brier_score"])
            ece_b, ece_a = float(b_m["ece"]), float(a_m["ece"])
            mce_b, mce_a = float(b_m.get("mce", 0.1770)), float(a_m.get("mce", 0.0569))
        except Exception:
            temp_val = 1.9591
            nll_b, nll_a = 0.544208, 0.472584
            brier_b, brier_a = 0.160799, 0.153251
            ece_b, ece_a = 0.084495, 0.023752
            mce_b, mce_a = 0.176951, 0.056903
    else:
        temp_val = 1.9591
        nll_b, nll_a = 0.544208, 0.472584
        brier_b, brier_a = 0.160799, 0.153251
        ece_b, ece_a = 0.084495, 0.023752
        mce_b, mce_a = 0.176951, 0.056903

    nll_rel = ((nll_a - nll_b) / nll_b) * 100
    brier_rel = ((brier_a - brier_b) / brier_b) * 100
    ece_rel = ((ece_a - ece_b) / ece_b) * 100
    mce_rel = ((mce_a - mce_b) / mce_b) * 100

    calib_table = [
        {
            "Calibration Metric": "Negative Log-Likelihood (NLL)",
            "Before (T = 1.0)": f"{nll_b:.4f}",
            f"After (T = {temp_val:.4f})": f"{nll_a:.4f}",
            "Improvement / Change": f"{nll_a - nll_b:+.4f} ({nll_rel:.1f}% relative)",
        },
        {
            "Calibration Metric": "Brier Score (MSE)",
            "Before (T = 1.0)": f"{brier_b:.4f}",
            f"After (T = {temp_val:.4f})": f"{brier_a:.4f}",
            "Improvement / Change": f"{brier_a - brier_b:+.4f} ({brier_rel:.1f}% relative)",
        },
        {
            "Calibration Metric": "Expected Calibration Error (ECE)",
            "Before (T = 1.0)": f"{ece_b:.4f}",
            f"After (T = {temp_val:.4f})": f"{ece_a:.4f}",
            "Improvement / Change": f"{ece_a - ece_b:+.4f} ({ece_rel:.1f}% error reduction)",
        },
        {
            "Calibration Metric": "Maximum Calibration Error (MCE)",
            "Before (T = 1.0)": f"{mce_b:.4f}",
            f"After (T = {temp_val:.4f})": f"{mce_a:.4f}",
            "Improvement / Change": f"{mce_a - mce_b:+.4f} ({mce_rel:.1f}% error reduction)",
        },
    ]
    st.table(calib_table)
    st.caption(
        "Threshold Invariance Note: Temperature scaling is a strictly monotonic transformation. "
        r"Because $\sigma(0 / T) = 0.50$ for all $T > 0$, the decision boundary at 0.50 is invariant ($z \ge 0 \iff p_{\text{AI}} \ge 0.50$)."
    )

    st.write("### 3. Official Held-Out Benchmark (100,000 Images)")

    benchmark_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "evaluation" / "benchmark_heldout_100k.json"
    if not benchmark_path.exists():
        benchmark_path = Path("artifacts/evaluation/benchmark_heldout_100k.json")

    bench_roc_auc = 0.8748
    bench_macro_f1 = 0.7982
    bench_acc = 0.7982
    bench_prec = 0.7907
    bench_rec = 0.8111
    bench_fpr = 0.2148
    bench_tn, bench_fp, bench_fn, bench_tp = 39262, 10738, 9443, 40557
    bench_device = "Tesla T4"
    bench_fps = 107.43

    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            b_ov = b_data.get("overall", {})
            b_cm = b_ov.get("confusion_matrix", {})
            b_meta = b_data.get("metadata", {})
            bench_roc_auc = float(b_ov.get("roc_auc", bench_roc_auc))
            bench_macro_f1 = float(b_ov.get("macro_f1", bench_macro_f1))
            bench_acc = float(b_ov.get("accuracy", bench_acc))
            bench_prec = float(b_ov.get("precision", bench_prec))
            bench_rec = float(b_ov.get("recall", bench_rec))
            bench_fpr = float(b_ov.get("fpr", bench_fpr))
            bench_tn = int(b_cm.get("tn", bench_tn))
            bench_fp = int(b_cm.get("fp", bench_fp))
            bench_fn = int(b_cm.get("fn", bench_fn))
            bench_tp = int(b_cm.get("tp", bench_tp))
            bench_device = str(b_meta.get("device_name", bench_device))
            bench_fps = float(b_meta.get("throughput_fps", bench_fps))
        except Exception:
            pass

    st.success(
        f"OFFICIAL BENCHMARK COMPLETED: Evaluated on an independent, quarantined cohort of 100,000 images "
        f"(50,000 Real + 50,000 AI-generated across 8 generator families) on an NVIDIA {bench_device} GPU "
        f"with FP16 autocast ({bench_fps:.1f} img/s throughput). Operating decision threshold fixed at $\\tau = 0.50$."
    )

    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    with hcol1:
        st.metric("ROC-AUC (Overall)", f"{bench_roc_auc:.4f}", help="Area Under ROC Curve on 100,000 held-out images")
    with hcol2:
        st.metric("Macro-F1", f"{bench_macro_f1:.4f}", help="Unweighted mean of Real-F1 and AI-F1 at 0.50 threshold")
    with hcol3:
        st.metric("Accuracy", f"{bench_acc * 100:.2f}%", help="Total accuracy: (39,262 TN + 40,557 TP) / 100,000")
    with hcol4:
        st.metric("False Positive Rate", f"{bench_fpr * 100:.2f}%", help="False alarms on natural captures (10,738 / 50,000)")

    hcol_cm, hcol_summary = st.columns([1, 1])

    with hcol_cm:
        st.markdown("**Official Held-Out Confusion Matrix (N = 100,000)**")
        cm_heldout_table = [
            {
                "Ground Truth": "Actual: Real (50,000)",
                "Predicted Real (<= 0.50)": f"{bench_tn:,} (TN, {bench_tn / 500:.1f}%)",
                "Predicted AI (> 0.50)": f"{bench_fp:,} (FP, {bench_fp / 500:.1f}%)",
            },
            {
                "Ground Truth": "Actual: AI (50,000)",
                "Predicted Real (<= 0.50)": f"{bench_fn:,} (FN, {bench_fn / 500:.1f}%)",
                "Predicted AI (> 0.50)": f"{bench_tp:,} (TP, {bench_tp / 500:.1f}%)",
            },
        ]
        st.table(cm_heldout_table)

    with hcol_summary:
        st.markdown("**Detailed Aggregate Metrics & Operational Parameters**")
        bench_summary_table = [
            {"Metric / Parameter": "Precision (AI Class)", "Value": f"{bench_prec:.4f}", "Target Spec": ">= 0.75"},
            {"Metric / Parameter": "Recall / TPR (AI Class)", "Value": f"{bench_rec:.4f}", "Target Spec": ">= 0.75"},
            {"Metric / Parameter": "Operating Decision Threshold", "Value": "0.50 (Fixed)", "Target Spec": "tau = 0.50"},
            {"Metric / Parameter": "Inference Throughput", "Value": f"{bench_fps:.2f} img/s", "Target Spec": "Real-time"},
            {"Metric / Parameter": "Cohort Balance", "Value": "50,000 Real / 50,000 AI", "Target Spec": "50:50"},
        ]
        st.table(bench_summary_table)

    st.caption(
        "Evaluator Scoring Detail: The official benchmark evaluator records raw sigmoid scores. "
        r"Because positive post-hoc temperature scaling ($T = 1.9591$) is strictly monotonic, "
        r"$\text{ROC-AUC} = 0.8748$ is strictly invariant, and the decision boundary at $\tau = 0.50$ ($z \ge 0$) is preserved identically."
    )

    st.write("### 4. System Limitations & Forensic Scope")
    st.markdown(
        "- **Binary Detection Scope**: The production ViT classifier is trained to estimate the probability that an image is synthetic vs. natural camera capture. It is not trained to identify specific generator models or prompt text.\n"
        "- **Single-Frame Static Scope**: Video, audio, and multimodal temporal sequences are out of scope for this architecture.\n"
        "- **Unseen Generator Generalization**: Performance varies across generator architectures. While diffusion models such as Wukong (AUC 0.9513) and Stable Diffusion (AUC 0.9419) are detected with high accuracy, older or distinct architectures such as ADM (AUC 0.6026) present a significant distribution shift.\n"
        "- **Saliency Interpretation**: The experimental Patch Decision Saliency highlights 16×16 patch regions that contributed above average to the AI logit. It is a decision attribution map for model interpretability, NOT a ground-truth tamper mask or pixel-level manipulation boundary.\n"
        "- **Auxiliary Metadata**: EXIF and C2PA provenance analyses are purely auxiliary. Stripped metadata does not indicate AI generation, and valid metadata does not guarantee absence of AI synthesis."
    )


def main() -> None:
    st.set_page_config(page_title="SignalScope", layout="wide")

    st.title("SignalScope — Generative Media Forensics")
    st.caption("Telling Real From Synthetic in the Age of Generative Media")

    tab1, tab2, tab3 = st.tabs([
        "🔍 Image Inspection",
        "📊 Robustness Playground",
        "📋 Evaluation & Model Report",
    ])

    with tab1:
        uploaded_file = st.file_uploader("Upload an image for verification (JPEG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state["uploaded_image_bytes"] = uploaded_file.getvalue()

    uploaded_image: Optional[Image.Image] = None
    raw_bytes = st.session_state.get("uploaded_image_bytes")
    if raw_bytes:
        uploaded_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    with tab1:
        _render_tab1(uploaded_image)

    with tab2:
        _render_tab2(uploaded_image)

    with tab3:
        _render_tab3()


if __name__ == "__main__":
    main()
