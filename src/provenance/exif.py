"""EXIF metadata extraction, sanitization, and forensic analysis.

Extracts standard EXIF tags (Make, Model, Software, DateTime, Lens, Exposure),
redacts sensitive geospatial (GPS) and device serial numbers, and inspects
PNG text chunks for known AI generative model signatures (Stable Diffusion,
ComfyUI, Midjourney, DALL-E, etc.).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ExifTags

# Known generative AI software patterns (regexes)
KNOWN_AI_PATTERNS = [
    (re.compile(r"stable[ _-]?diffusion", re.IGNORECASE), "Stable Diffusion"),
    (re.compile(r"midjourney", re.IGNORECASE), "Midjourney"),
    (re.compile(r"dall[ _-]?e", re.IGNORECASE), "DALL-E"),
    (re.compile(r"comfyui", re.IGNORECASE), "ComfyUI"),
    (re.compile(r"fooocus", re.IGNORECASE), "Fooocus"),
    (re.compile(r"novelai", re.IGNORECASE), "NovelAI"),
    (re.compile(r"invokeai", re.IGNORECASE), "InvokeAI"),
    (re.compile(r"adobe\s*firefly", re.IGNORECASE), "Adobe Firefly"),
    (re.compile(r"automatic1111", re.IGNORECASE), "AUTOMATIC1111 WebUI"),
]

# Sensitive tag IDs and names to redact or mask
GPS_TAG_ID = 0x8825  # Tag 34853 (GPSInfo)
SERIAL_TAG_NAMES = {
    "BodySerialNumber",
    "LensSerialNumber",
    "CameraSerialNumber",
    "SerialNumber",
    "InternalSerialNumber",
}


def _mask_serial(serial_val: Any) -> str:
    """Mask serial number to preserve privacy while confirming tag presence."""
    s = str(serial_val).strip()
    if len(s) <= 4:
        return "****"
    return f"****-{s[-4:]}"


def _sanitize_val(val: Any, max_len: int = 256) -> Any:
    """Sanitize raw EXIF values to ensure JSON serializability and bounded size."""
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, bytes):
        try:
            val = val.decode("utf-8", errors="replace")
        except Exception:
            val = "<binary data>"
    s = str(val).strip().replace("\x00", "")
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def extract_exif(img: Image.Image) -> Dict[str, Any]:
    """Extract and analyze EXIF and image metadata safely.

    Parameters
    ----------
    img : Image.Image
        Input PIL image.

    Returns
    -------
    dict
        Structured dictionary containing:
        - present (bool)
        - camera_make (Optional[str])
        - camera_model (Optional[str])
        - software (Optional[str])
        - datetime_original (Optional[str])
        - lens_model (Optional[str])
        - exposure_info (dict)
        - gps_redacted (bool)
        - serial_redacted (bool)
        - generation_indicators (dict)
        - raw_tags_count (int)
        - notes (List[str])
    """
    notes: List[str] = []
    gps_redacted = False
    serial_redacted = False
    masked_serial: Optional[str] = None

    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    datetime_original: Optional[str] = None
    lens_model: Optional[str] = None

    exposure_info: Dict[str, Any] = {
        "f_number": None,
        "exposure_time": None,
        "iso": None,
        "focal_length": None,
    }

    generation_indicators: Dict[str, Any] = {
        "known_ai_software_flag": False,
        "generator_signature": None,
        "png_parameters_present": False,
    }

    raw_exif_dict: Dict[int, Any] = {}
    try:
        exif_obj = img.getexif()
        if exif_obj is not None:
            raw_exif_dict = dict(exif_obj)
    except Exception as exc:
        notes.append(f"EXIF parsing error: {str(exc)}")

    # Check for GPSInfo tag (34853 / 0x8825) or IFD GPS block
    if GPS_TAG_ID in raw_exif_dict:
        gps_redacted = True
        notes.append("Geospatial metadata (GPS coordinates) detected and redacted for privacy.")
        raw_exif_dict.pop(GPS_TAG_ID, None)

    try:
        if hasattr(exif_obj, "get_ifd"):
            gps_ifd = exif_obj.get_ifd(ExifTags.IFD.GPSInfo)
            if gps_ifd:
                gps_redacted = True
                if "Geospatial metadata (GPS coordinates) detected and redacted for privacy." not in notes:
                    notes.append("Geospatial metadata (GPS coordinates) detected and redacted for privacy.")
    except Exception:
        pass

    # Humanize tags
    human_tags: Dict[str, Any] = {}
    for tag_id, val in raw_exif_dict.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        
        # Privacy check for serial numbers
        if tag_name in SERIAL_TAG_NAMES:
            serial_redacted = True
            masked_val = _mask_serial(val)
            masked_serial = masked_val
            human_tags[tag_name] = masked_val
            notes.append(f"Device identifier ({tag_name}) detected and masked for privacy: {masked_val}")
            continue

        human_tags[tag_name] = _sanitize_val(val)

    if "Make" in human_tags:
        camera_make = str(human_tags["Make"])
    if "Model" in human_tags:
        camera_model = str(human_tags["Model"])
    if "Software" in human_tags:
        software = str(human_tags["Software"])
    if "DateTimeOriginal" in human_tags:
        datetime_original = str(human_tags["DateTimeOriginal"])
    elif "DateTime" in human_tags:
        datetime_original = str(human_tags["DateTime"])
    if "LensModel" in human_tags:
        lens_model = str(human_tags["LensModel"])

    # Exposure tags
    if "FNumber" in human_tags:
        try:
            exposure_info["f_number"] = float(human_tags["FNumber"])
        except (ValueError, TypeError):
            exposure_info["f_number"] = str(human_tags["FNumber"])
    if "ExposureTime" in human_tags:
        exposure_info["exposure_time"] = str(human_tags["ExposureTime"])
    if "ISOSpeedRatings" in human_tags:
        try:
            exposure_info["iso"] = int(human_tags["ISOSpeedRatings"])
        except (ValueError, TypeError):
            exposure_info["iso"] = str(human_tags["ISOSpeedRatings"])
    if "FocalLength" in human_tags:
        try:
            exposure_info["focal_length"] = float(human_tags["FocalLength"])
        except (ValueError, TypeError):
            exposure_info["focal_length"] = str(human_tags["FocalLength"])

    # Inspect PNG chunks and img.info dictionary (e.g. parameters, prompt, workflow)
    png_info = getattr(img, "info", {})
    if isinstance(png_info, dict):
        # Stable Diffusion parameters chunk
        if "parameters" in png_info:
            generation_indicators["png_parameters_present"] = True
            generation_indicators["known_ai_software_flag"] = True
            param_str = _sanitize_val(png_info["parameters"], max_len=120)
            generation_indicators["generator_signature"] = f"Stable Diffusion parameters: {param_str}"
            notes.append("PNG text chunk 'parameters' detected (common in Stable Diffusion WebUI / ComfyUI generation).")

        for key in ["prompt", "workflow", "Comment"]:
            if key in png_info and not generation_indicators["generator_signature"]:
                val_str = _sanitize_val(png_info[key], max_len=120)
                for pattern, name in KNOWN_AI_PATTERNS:
                    if pattern.search(val_str):
                        generation_indicators["known_ai_software_flag"] = True
                        generation_indicators["generator_signature"] = f"{name} signature in PNG chunk '{key}'"
                        notes.append(f"Known AI generator pattern ({name}) detected in PNG chunk '{key}'.")
                        break

        # Check Software field in info if not present in EXIF
        if not software and "Software" in png_info:
            software = _sanitize_val(png_info["Software"])

    # Inspect software string for AI signatures
    if software:
        for pattern, name in KNOWN_AI_PATTERNS:
            if pattern.search(software):
                generation_indicators["known_ai_software_flag"] = True
                if not generation_indicators["generator_signature"]:
                    generation_indicators["generator_signature"] = f"{name} identified in Software header ({software})"
                notes.append(f"Known generative AI software tag ({name}) detected in Software field.")
                break

    exif_present = bool(camera_make or camera_model or datetime_original or lens_model or len(human_tags) > 2)

    if exif_present:
        if camera_make or camera_model:
            notes.append(f"Hardware camera profile found: {camera_make or 'Unknown Make'} {camera_model or ''}".strip())
    else:
        notes.append("No standard camera EXIF metadata found (typical for web-compressed or direct synthetic images).")

    return {
        "present": exif_present,
        "camera_make": camera_make,
        "camera_model": camera_model,
        "software": software,
        "datetime_original": datetime_original,
        "lens_model": lens_model,
        "exposure_info": exposure_info,
        "gps_redacted": gps_redacted,
        "serial_redacted": serial_redacted,
        "masked_serial": masked_serial,
        "generation_indicators": generation_indicators,
        "raw_tags_count": len(human_tags),
        "notes": notes,
    }
