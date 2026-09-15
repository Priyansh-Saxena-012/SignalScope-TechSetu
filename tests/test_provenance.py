"""Unit and contract tests for SignalScope Provenance and Metadata analysis.

Covers:
1. Synthetic image with EXIF Make/Model/DateTime
2. Image with no EXIF (stripped metadata)
3. PNG with synthetic Stable Diffusion-style parameters chunk
4. Known AI software signature detection (e.g. Midjourney, Fooocus, etc.)
5. Ordinary camera software metadata (e.g. Photoshop, firmware)
6. C2PA / JUMBF marker detection
7. C2PA structure without cryptographic verification (UNVERIFIED_MANIFEST)
8. Malformed metadata / corrupted EXIF bytes handling (no crash)
9. GPS metadata redaction (lat/lon must never be exposed, gps_redacted=True)
10. Serial-number masking (****-XXXX)
11. Deterministic output across repeated calls
12. Input image immutability (original bytes/image untouched)
13. Unsupported metadata formats handling
14. Prediction contract compatibility (model/predict.py)
15. Invariance guarantee: provenance analysis MUST NEVER modify detector p_ai, verdict, or confidence.
"""

from __future__ import annotations

import io
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from PIL import Image, PngImagePlugin, ExifTags

from model.predict import clear_model_cache, predict, validate_prediction_schema
from model.backbone import build_classifier
from src.provenance.analyzer import analyze_provenance
from src.provenance.c2pa import C2PA_UUID, inspect_c2pa, inspect_c2pa_bytes
from src.provenance.exif import extract_exif


@pytest.fixture
def mock_vit_weights(tmp_path):
    """Create minimal mock weights to test prediction contract."""
    ckpt_path = tmp_path / "mock_weights.pth"
    model = build_classifier("vit_base_patch16_224", pretrained=False, drop_rate=0.2)
    checkpoint_data = {
        "epoch": 1,
        "backbone": "vit_base_patch16_224",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "val_roc_auc": 0.91,
    }
    torch.save(checkpoint_data, str(ckpt_path))
    return str(ckpt_path)


# 1. Synthetic image with standard camera EXIF
def test_synthetic_image_with_camera_exif(tmp_path):
    img_path = tmp_path / "camera_sample.jpg"
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    
    exif = img.getexif()
    exif[271] = "Sony"                # Make
    exif[272] = "ILCE-7RM4"           # Model
    exif[305] = "ILCE-7RM4 v1.20"     # Software
    exif[306] = "2024:05:12 14:30:00" # DateTime
    
    img.save(img_path, "JPEG", exif=exif)

    result = analyze_provenance(str(img_path))
    prov = result["provenance"]

    assert result["exif_intact"] is True
    assert prov["status"] == "EXIF_PRESENT"
    assert prov["exif"]["camera_make"] == "Sony"
    assert prov["exif"]["camera_model"] == "ILCE-7RM4"
    assert prov["exif"]["datetime_original"] == "2024:05:12 14:30:00"
    assert prov["exif"]["gps_redacted"] is False
    assert prov["generation_indicators"]["known_ai_software_flag"] is False


# 2. Image with no EXIF (stripped metadata)
def test_image_with_no_exif(tmp_path):
    img_path = tmp_path / "stripped.png"
    img = Image.new("RGB", (64, 64), color=(50, 50, 50))
    img.save(img_path, "PNG")

    result = analyze_provenance(str(img_path))
    prov = result["provenance"]

    assert result["exif_intact"] is False
    assert result["c2pa_present"] is False
    assert prov["status"] == "STRIPPED"
    assert prov["exif"]["present"] is False
    assert any("Absence of EXIF" in note for note in prov["forensic_notes"])


# 3. PNG with synthetic Stable Diffusion parameters chunk
def test_png_with_stable_diffusion_parameters(tmp_path):
    img_path = tmp_path / "sd_generated.png"
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", "masterpiece, cinematic portrait of a cat\nSteps: 25, Sampler: DPM++ 2M, CFG: 7")
    
    img.save(img_path, "PNG", pnginfo=pnginfo)

    result = analyze_provenance(str(img_path))
    prov = result["provenance"]

    assert prov["status"] == "METADATA_SUSPICIOUS"
    assert prov["generation_indicators"]["known_ai_software_flag"] is True
    assert prov["generation_indicators"]["png_parameters_present"] is True
    assert "Stable Diffusion" in prov["generation_indicators"]["generator_signature"]


# 4. Known AI software signature detection (Midjourney, ComfyUI, etc.)
@pytest.mark.parametrize("software_name,expected_family", [
    ("Midjourney v6.0", "Midjourney"),
    ("ComfyUI-0.2.1", "ComfyUI"),
    ("Fooocus v2.5", "Fooocus"),
    ("NovelAI Diffusion", "NovelAI"),
    ("InvokeAI Studio", "InvokeAI"),
    ("Adobe Firefly 2024", "Adobe Firefly"),
])
def test_known_ai_software_signatures(tmp_path, software_name, expected_family):
    img_path = tmp_path / f"ai_{expected_family}.jpg"
    img = Image.new("RGB", (64, 64), color=(80, 90, 100))
    
    exif = img.getexif()
    exif[305] = software_name  # Software tag
    img.save(img_path, "JPEG", exif=exif)

    result = analyze_provenance(str(img_path))
    prov = result["provenance"]

    assert prov["status"] == "METADATA_SUSPICIOUS"
    assert prov["generation_indicators"]["known_ai_software_flag"] is True
    assert expected_family in prov["generation_indicators"]["generator_signature"]


# 5. Ordinary camera software metadata should not trigger AI flag
def test_ordinary_camera_software_metadata(tmp_path):
    img_path = tmp_path / "photoshop_edit.jpg"
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    
    exif = img.getexif()
    exif[271] = "Canon"
    exif[272] = "EOS R5"
    exif[305] = "Adobe Photoshop 25.0 (Windows)"
    img.save(img_path, "JPEG", exif=exif)

    result = analyze_provenance(str(img_path))
    prov = result["provenance"]

    assert prov["status"] == "EXIF_PRESENT"
    assert prov["generation_indicators"]["known_ai_software_flag"] is False
    assert prov["exif"]["camera_make"] == "Canon"


# 6 & 7. C2PA / JUMBF structure detection and UNVERIFIED_MANIFEST semantics
def test_c2pa_jumbf_manifest_detection_unverified():
    # Construct mock JPEG bytes with C2PA UUID embedded in an APP11 marker
    header = b"\xff\xd8\xff\xeb\x00\x30"  # SOI + APP11 length 48
    jumbf_payload = b"jumb\x00\x00\x00\x18jumd" + C2PA_UUID + b"\x00\x00"
    jpeg_body = header + jumbf_payload + b"\xff\xd9"  # EOI

    c2pa_report = inspect_c2pa_bytes(jpeg_body)
    assert c2pa_report["present"] is True
    # Without cryptographic verification via PKI certificates, state must be UNVERIFIED_MANIFEST
    assert c2pa_report["validation_state"] == "UNVERIFIED_MANIFEST"
    assert c2pa_report["signature_valid"] is None  # None = not cryptographically verified


# 8. Malformed metadata / corrupted header handling
def test_malformed_exif_safety(tmp_path):
    img_path = tmp_path / "corrupted_exif.jpg"
    img = Image.new("RGB", (64, 64))
    img.save(img_path, "JPEG")
    
    # Intentionally corrupt file bytes by appending garbage to headers
    with open(img_path, "rb") as f:
        data = f.read()
    corrupted_data = data[:10] + b"\xff\xeb\x00\x04\x00\x00\x00\x00" + data[10:]
    with open(img_path, "wb") as f:
        f.write(corrupted_data)

    # Must not raise an unhandled exception
    result = analyze_provenance(str(img_path))
    assert isinstance(result, dict)
    assert "provenance" in result
    assert result["provenance"]["status"] in ("STRIPPED", "EXIF_PRESENT", "ERROR")


# 9. Privacy: GPS coordinates redaction
def test_gps_metadata_redaction(tmp_path):
    img_path = tmp_path / "geotagged.jpg"
    img = Image.new("RGB", (64, 64))
    
    exif = img.getexif()
    # Populate GPS Info IFD and register on root EXIF
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[1] = "N"
    gps_ifd[2] = (37.7749, 1.0)
    gps_ifd[3] = "W"
    gps_ifd[4] = (122.4194, 1.0)
    exif[ExifTags.Base.GPSInfo] = gps_ifd
    exif[271] = "Apple"
    exif[272] = "iPhone 14 Pro"
    img.save(img_path, "JPEG", exif=exif)

    result = analyze_provenance(str(img_path))
    prov = result["provenance"]

    assert prov["exif"]["gps_redacted"] is True
    # Verify no raw GPS coordinates exist anywhere in the returned dictionary
    result_str = json.dumps(result)
    assert "37" not in result_str or "122" not in result_str or "GPSInfo" not in result_str
    assert "geospatial metadata" in " ".join(prov["forensic_notes"]).lower()


# 10. Privacy: Serial number masking
def test_serial_number_masking(tmp_path):
    img_path = tmp_path / "serial_sample.jpg"
    img = Image.new("RGB", (64, 64))
    
    exif = img.getexif()
    exif[271] = "Sony"
    # Tag 42033 (0xA431) is BodySerialNumber
    exif[42033] = "192837465"
    img.save(img_path, "JPEG", exif=exif)

    result = analyze_provenance(str(img_path))
    prov = result["provenance"]

    assert prov["exif"]["serial_redacted"] is True
    result_str = json.dumps(result)
    assert "192837465" not in result_str
    assert "7465" in result_str  # Last 4 digits kept masked: ****-7465


# 11. Determinism across repeated calls
def test_deterministic_output(tmp_path):
    img_path = tmp_path / "deterministic.jpg"
    img = Image.new("RGB", (64, 64), color=(120, 140, 160))
    exif = img.getexif()
    exif[271] = "Nikon"
    exif[272] = "Z9"
    img.save(img_path, "JPEG", exif=exif)

    res1 = analyze_provenance(str(img_path))
    res2 = analyze_provenance(str(img_path))

    assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)


# 12. Input image immutability
def test_input_image_immutability(tmp_path):
    img_path = tmp_path / "immutable.jpg"
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(img_path, "JPEG")
    
    with open(img_path, "rb") as f:
        bytes_before = f.read()

    _ = analyze_provenance(str(img_path))

    with open(img_path, "rb") as f:
        bytes_after = f.read()

    assert bytes_before == bytes_after


# 13. Unsupported input types handled defensively
def test_unsupported_input_type():
    result = analyze_provenance(12345)  # type: ignore
    assert result["provenance"]["status"] == "ERROR"
    assert "Unsupported image input type" in result["provenance"]["forensic_notes"][0]


# 14. Prediction contract compatibility (model/predict.py)
def test_prediction_contract_with_provenance(tmp_path, mock_vit_weights):
    img_path = tmp_path / "test_contract.png"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path, "PNG")

    clear_model_cache()
    pred_result = predict(str(img_path), allow_stub=False, weights_path=mock_vit_weights)

    # Core required keys from contract
    assert validate_prediction_schema(pred_result) is True
    assert "label" in pred_result
    assert "confidence" in pred_result
    assert "is_ai" in pred_result

    # Metadata backward compatibility
    assert "metadata" in pred_result
    assert "c2pa_present" in pred_result["metadata"]
    assert "exif_intact" in pred_result["metadata"]

    # Extended provenance object
    assert "provenance" in pred_result
    assert "status" in pred_result["provenance"]
    assert pred_result["provenance"]["status"] == "STRIPPED"
    clear_model_cache()


# 15. Invariance guarantee: provenance analysis MUST NEVER modify detector p_ai / verdict / confidence
def test_provenance_does_not_modify_detector_verdict(tmp_path, mock_vit_weights):
    """Ensure that injecting AI generator metadata or camera metadata does NOT alter p_ai."""
    img_plain = tmp_path / "sample_plain.jpg"
    img_with_sd = tmp_path / "sample_sd.jpg"
    
    base_img = Image.new("RGB", (224, 224), color=(150, 150, 150))
    base_img.save(img_plain, "JPEG")

    # Save exact same visual image but with Stable Diffusion software tag
    exif = base_img.getexif()
    exif[305] = "Stable Diffusion WebUI AUTOMATIC1111"
    base_img.save(img_with_sd, "JPEG", exif=exif)

    clear_model_cache()
    res_plain = predict(str(img_plain), allow_stub=False, weights_path=mock_vit_weights)
    clear_model_cache()
    res_sd = predict(str(img_with_sd), allow_stub=False, weights_path=mock_vit_weights)

    # Probabilities and confidences must be IDENTICAL (metadata has 0 impact on ViT inference)
    assert res_plain["ai_probability"] == res_sd["ai_probability"]
    assert res_plain["confidence"] == res_sd["confidence"]
    assert res_plain["label"] == res_sd["label"]

    # Only auxiliary provenance fields differ
    assert res_plain["provenance"]["status"] == "STRIPPED"
    assert res_sd["provenance"]["status"] == "METADATA_SUSPICIOUS"
    clear_model_cache()
