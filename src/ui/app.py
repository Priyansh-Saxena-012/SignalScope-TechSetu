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
import os
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

    prob = result["calibrated_probability"]
    st.metric(
        label="AI-generated probability",
        value=f"{prob * 100:.1f}%",
        delta=result["confidence_level"] + " confidence",
        delta_color="off",
    )
    st.progress(min(max(prob, 0.0), 1.0))

    st.info(f"**Attribution Family:** {result['generator_family']}")

    st.markdown("**Grounded Findings:**")
    for cue in result["explanation"]["cues"]:
        st.markdown(f"- {cue}")
    st.caption(result["explanation"]["summary"])


def _render_metadata_card(image: Image.Image, result: Dict[str, Any]) -> None:
    meta = result.get("metadata", {})
    prov = meta.get("provenance", {})
    st.markdown("**Source & Provenance Metadata**")
    st.write(f"Resolution: {image.width} x {image.height} px")
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
    if prov:
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
                            "Relative patch saliency on a 14×14 grid, highlighting regions that contributed above average "
                            "toward the AI-generated decision. This illustrates model decision focus and is not a pixel-level forgery mask."
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
    st.write("Test detector resilience against real-world social-media style degradation.")

    if uploaded_image is None:
        st.info("Upload an image in the Image Inspection tab first.")
        return

    jpeg_quality = st.slider("JPEG Compression Quality", min_value=20, max_value=100, value=100, step=5)
    blur_radius = st.slider("Gaussian Blur Radius", min_value=0.0, max_value=4.0, value=0.0, step=0.1)
    resize_pct = st.slider("Downscale / Resize (%)", min_value=25, max_value=100, value=100, step=5)

    if st.button("Re-test Under Degradation"):
        degraded = degrade_image(uploaded_image, jpeg_quality, blur_radius, resize_pct)

        original_result = predict(uploaded_image)
        degraded_result = predict(degraded)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Original")
            st.image(uploaded_image, use_column_width=True)
        with col_right:
            st.subheader("Degraded Preview")
            st.image(degraded, use_column_width=True)

        orig_pct = original_result["calibrated_probability"] * 100
        deg_pct = degraded_result["calibrated_probability"] * 100
        delta = deg_pct - orig_pct
        status = "Stable" if abs(delta) < 10 else "Shifted"

        st.markdown("**Delta Analysis**")
        st.write(f"Original: {orig_pct:.1f}% → Degraded: {deg_pct:.1f}% | Status: {status}")
    else:
        st.caption("Adjust the sliders and click \"Re-test Under Degradation\" to run the comparison.")


def _render_tab3() -> None:
    st.write("### Development Validation Set (Stage 9 Baseline)")
    st.caption("Evaluated on 7,000 Tiny-GenImage development validation images. These are development validation results, NOT official held-out benchmark results.")

    dev_metrics_table = [
        {"Metric": "Validation Samples", "Value": "7,000"},
        {"Metric": "ROC-AUC (Overall)", "Value": "0.8562"},
        {"Metric": "Macro-F1", "Value": "0.7834"},
        {"Metric": "Accuracy", "Value": "78.34%"},
    ]
    st.table(dev_metrics_table)

    st.write("### Official Held-Out Benchmark (100k)")
    st.caption("Preliminary — pending final official held-out benchmark (Stage 10). Numbers below are placeholders, not final results.")

    metrics_table = [
        {"Metric": "ROC-AUC (Overall)", "Value": "TBD"},
        {"Metric": "ROC-AUC (Unseen-Generator Split)", "Value": "TBD"},
        {"Metric": "Macro-F1", "Value": "TBD"},
        {"Metric": "Accuracy", "Value": "TBD"},
        {"Metric": "Expected Calibration Error (ECE)", "Value": "TBD"},
    ]
    st.table(metrics_table)

    st.markdown("**Confusion Matrix (placeholder)**")
    confusion_table = [
        {"": "Actual: Real", "Predicted: Real": "TBD", "Predicted: AI": "TBD"},
        {"": "Actual: AI", "Predicted: Real": "TBD", "Predicted: AI": "TBD"},
    ]
    st.table(confusion_table)

    st.markdown("**Known Limitations**")
    st.markdown(
        "- Metrics above are placeholders until final official held-out benchmark evaluation completes on the official held-out test set.\n"
        "- Robustness to heavy compression and unseen generator families has not yet been empirically validated.\n"
        "- Video and multi-frame content are out of scope for this detector.\n"
        "- Generator attribution is Undetermined because the ViT baseline operates as a binary veracity detector.\n"
        "- Decision saliency is an experimental relative attribution method operating at 16×16 patch resolution. "
        "It highlights regions driving the AI logit, but does not explain authenticity (realness) and is not a pixel-level forgery or tamper mask."
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
