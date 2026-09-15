"""Main provenance and metadata analyzer orchestrator for SignalScope.

Integrates EXIF inspection, PNG generation chunk parsing, and conservative C2PA/JUMBF
detection into the standardized, deterministic provenance schema.

Guarantees:
- AUXILIARY ONLY: Never modifies detector predictions, p_ai, confidence, or threshold.
- Non-destructive: Does not alter or mutate the input image or file.
- Defensive: Catches and formats errors without crashing.
- Safe serialization: All output fields are native, JSON-serializable Python types.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from PIL import Image

from src.provenance.exif import extract_exif
from src.provenance.c2pa import inspect_c2pa


class ProvenanceResult(TypedDict, total=False):
    status: str
    format: str
    file_size_bytes: int
    image_dimensions: List[int]
    c2pa: Dict[str, Any]
    exif: Dict[str, Any]
    generation_indicators: Dict[str, Any]
    forensic_notes: List[str]


def analyze_provenance(
    image_or_path: Union[str, Path, Image.Image, bytes],
) -> Dict[str, Any]:
    """Analyze image provenance, EXIF metadata, and C2PA Content Credentials.

    Parameters
    ----------
    image_or_path : Union[str, Path, Image.Image, bytes]
        Input image file path, Path object, PIL Image instance, or raw bytes.

    Returns
    -------
    dict
        Structured output conforming to the approved provenance schema:
        {
            "c2pa_present": bool,
            "exif_intact": bool,
            "provenance": {
                "status": "VALID_C2PA" | "EXIF_PRESENT" | "STRIPPED" | "METADATA_SUSPICIOUS" | "ERROR",
                "format": str,
                "file_size_bytes": int,
                "image_dimensions": [int, int],
                "c2pa": { ... },
                "exif": { ... },
                "generation_indicators": { ... },
                "forensic_notes": [ ... ]
            }
        }
    """
    forensic_notes: List[str] = []
    file_size_bytes = 0
    img_format = "UNKNOWN"
    dimensions = [0, 0]
    pil_img: Optional[Image.Image] = None

    # Defensive input handling
    try:
        if isinstance(image_or_path, (str, Path)):
            path_obj = Path(image_or_path)
            if not path_obj.exists():
                return _error_result(f"File not found: {str(path_obj)}")
            file_size_bytes = path_obj.stat().st_size
            pil_img = Image.open(str(path_obj))
            img_format = (pil_img.format or path_obj.suffix.lstrip(".").upper()) or "UNKNOWN"
            dimensions = [pil_img.width, pil_img.height]
            c2pa_result = inspect_c2pa(str(path_obj))

        elif isinstance(image_or_path, bytes):
            file_size_bytes = len(image_or_path)
            import io
            buf = io.BytesIO(image_or_path)
            pil_img = Image.open(buf)
            img_format = pil_img.format or "UNKNOWN"
            dimensions = [pil_img.width, pil_img.height]
            c2pa_result = inspect_c2pa(image_or_path)

        elif isinstance(image_or_path, Image.Image):
            pil_img = image_or_path
            img_format = pil_img.format or "UNKNOWN"
            dimensions = [pil_img.width, pil_img.height]
            c2pa_result = inspect_c2pa(image_or_path)

        else:
            return _error_result(f"Unsupported image input type: {type(image_or_path).__name__}")

    except Exception as exc:
        return _error_result(f"Failed to open or inspect image container: {str(exc)}")

    # Extract EXIF and generation indicators
    try:
        exif_result = extract_exif(pil_img)
    except Exception as exc:
        exif_result = {
            "present": False,
            "camera_make": None,
            "camera_model": None,
            "software": None,
            "datetime_original": None,
            "lens_model": None,
            "exposure_info": {},
            "gps_redacted": False,
            "serial_redacted": False,
            "generation_indicators": {
                "known_ai_software_flag": False,
                "generator_signature": None,
                "png_parameters_present": False,
            },
            "raw_tags_count": 0,
            "notes": [f"EXIF parsing error: {str(exc)}"],
        }

    # Aggregate forensic notes
    forensic_notes.extend(exif_result.get("notes", []))
    if c2pa_result.get("details"):
        forensic_notes.append(c2pa_result["details"])

    # Determine overall provenance status
    # Hierarchy:
    # 1. VALID_C2PA (genuine cryptographic verification)
    # 2. METADATA_SUSPICIOUS (known AI generator signature flagged in headers/chunks)
    # 3. EXIF_PRESENT (camera hardware EXIF intact)
    # 4. STRIPPED (no EXIF or C2PA)
    # 5. ERROR (fatal parsing failure)
    generation_indicators = exif_result.get("generation_indicators", {})
    known_ai_flag = generation_indicators.get("known_ai_software_flag", False)
    c2pa_state = c2pa_result.get("validation_state", "NOT_PRESENT")

    if c2pa_state == "VALID_C2PA":
        status = "VALID_C2PA"
    elif known_ai_flag:
        status = "METADATA_SUSPICIOUS"
        forensic_notes.append(
            "Contextual advisory: AI generator indicators detected in metadata headers. "
            "Note that metadata is auxiliary and does not substitute for visual artifact analysis."
        )
    elif exif_result.get("present", False):
        status = "EXIF_PRESENT"
    else:
        status = "STRIPPED"
        forensic_notes.append(
            "Advisory: Absence of EXIF metadata is standard across social media and messaging platforms, "
            "and cannot alone be interpreted as proof of AI synthesis."
        )

    clean_exif = {
        "present": exif_result.get("present", False),
        "camera_make": exif_result.get("camera_make"),
        "camera_model": exif_result.get("camera_model"),
        "software": exif_result.get("software"),
        "datetime_original": exif_result.get("datetime_original"),
        "lens_model": exif_result.get("lens_model"),
        "exposure_info": exif_result.get("exposure_info", {}),
        "gps_redacted": exif_result.get("gps_redacted", False),
        "serial_redacted": exif_result.get("serial_redacted", False),
        "masked_serial": exif_result.get("masked_serial"),
    }

    c2pa_present = bool(c2pa_result.get("present", False))
    exif_intact = bool(clean_exif["present"])

    return {
        "c2pa_present": c2pa_present,
        "exif_intact": exif_intact,
        "provenance": {
            "status": status,
            "format": str(img_format).upper(),
            "file_size_bytes": file_size_bytes,
            "image_dimensions": dimensions,
            "c2pa": c2pa_result,
            "exif": clean_exif,
            "generation_indicators": generation_indicators,
            "forensic_notes": forensic_notes,
        },
    }


def _error_result(error_msg: str) -> Dict[str, Any]:
    """Construct an honest ERROR provenance result."""
    return {
        "c2pa_present": False,
        "exif_intact": False,
        "provenance": {
            "status": "ERROR",
            "format": "UNKNOWN",
            "file_size_bytes": 0,
            "image_dimensions": [0, 0],
            "c2pa": {
                "present": False,
                "signature_valid": None,
                "claim_generator": None,
                "assertions": [],
                "validation_state": "NOT_PRESENT",
                "details": error_msg,
            },
            "exif": {
                "present": False,
                "camera_make": None,
                "camera_model": None,
                "software": None,
                "datetime_original": None,
                "lens_model": None,
                "exposure_info": {},
                "gps_redacted": False,
                "serial_redacted": False,
            },
            "generation_indicators": {
                "known_ai_software_flag": False,
                "generator_signature": None,
                "png_parameters_present": False,
            },
            "forensic_notes": [f"Provenance analysis error: {error_msg}"],
        },
    }
