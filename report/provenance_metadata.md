# SignalScope Stage 13: Provenance & Metadata Forensic Analysis

## Executive Summary
This report documents the architectural implementation and forensic boundaries of the **Provenance and Metadata Analysis Engine** in SignalScope (Stage 13). 

Designed to fulfill the SIH bonus requirement for media provenance, attribution, and metadata inspection, this module inspects EXIF data, container headers, PNG text chunks, and C2PA Content Credentials. Crucially, this analysis is strictly treated as an **auxiliary forensic signal**: it provides essential investigative context, but is never conflated with mathematical proof of authenticity or generation, and never modifies the decision boundaries of the frozen ViT-Base/16 neural detector.

---

## 1. System Architecture

```
                                  Input Image (File / Bytes / PIL)
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
         Visual Deep Learning Pipeline                                Provenance & Metadata Engine
       (FROZEN ViT-Base/16 Checkpoint)                            (src/provenance/analyzer.py)
                   │                                                           │
                   ▼                                      ┌────────────────────┼────────────────────┐
      p_ai, prediction, Grad-CAM                          ▼                    ▼                    ▼
                   │                                EXIF Extractor       PNG/Software Info    C2PA Box Parser
                   │                               (Pillow getexif)    (Prompt/Tool chunks) (JUMBF / c2pa UUID)
                   │                                      │                    │                    │
                   │                                      └────────────────────┼────────────────────┘
                   │                                                           ▼
                   │                                             Provenance Aggregator & Validator
                   │                                             - Privacy Redaction (GPS/Serial)
                   │                                             - Forensic Heuristics & Consistency
                   │                                             - Verification Status Determination
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 ▼
                                        SignalScope Result
                                  {
                                    "prediction": "AI" / "Real",
                                    "p_ai": 0.87,
                                    "provenance": { ... structured schema ... },
                                    ...
                                  }
```

### Core Design Guarantees:
1. **Auxiliary & Independent**: The provenance analyzer operates as a decoupled signal. Under no circumstances does it modify `ai_probability`, alter the `label`, change `confidence`, or shift the fixed `0.50` decision threshold.
2. **Zero-Crash Defensive Execution**: Malformed EXIF structures, corrupted bytes, and exotic image containers are intercepted safely without disrupting inference.
3. **No Added Mandatory Dependencies**: Implemented strictly using Pillow (`pillow==10.2.0` / `pillow>=10.2.0`) and the Python standard library (`struct`, `io`, `re`, `json`). No native Rust binaries or unsigned `.dll` files are loaded.

---

## 2. Supported Metadata & Extraction Capabilities

### 2.1 EXIF & Camera Profile Inspection
- **Hardware Profile**: Camera Make (tag 271), Camera Model (tag 272), Lens Model (tag 42036).
- **Exposure Configuration**: F-Number (tag 33437), Exposure Time (tag 33434), ISO Speed (tag 34855), Focal Length (tag 37386).
- **Temporal Profile**: DateTimeOriginal (tag 36867) or DateTime (tag 306).
- **Format & Geometry**: Native image dimensions `[width, height]`, file size in bytes, and container format (`JPEG`, `PNG`, `WEBP`, `TIFF`).

### 2.2 PNG Generation Metadata & Text Chunks
Many generative AI tools (such as Stable Diffusion WebUI, ComfyUI, Fooocus, and NovelAI) embed generation parameters directly inside PNG text chunks or EXIF Software headers:
- `parameters`: Full prompt, negative prompt, sampler, seed, CFG scale, and model hash.
- `prompt` & `workflow`: Node graph definitions embedded by node-based tools (ComfyUI).
- `Software`: Generator software versions and application strings.

The engine parses these chunks and conservative regex heuristics match against a curated registry of generative platforms:
- **Stable Diffusion**
- **Midjourney**
- **DALL-E**
- **ComfyUI**
- **Fooocus**
- **NovelAI**
- **InvokeAI**
- **Adobe Firefly**

---

## 3. Privacy Safeguards

In digital forensics, raw image files often leak sensitive personally identifiable information (PII). SignalScope enforces strict privacy safeguards:

1. **Geospatial (GPS) Redaction**:
   - Tag `0x8825` (`GPSInfo`) and IFD `GPSInfo` are intercepted immediately.
   - Raw latitude, longitude, and altitude coordinates are stripped before the metadata payload is serialized or returned to UI/API.
   - The engine flags `"gps_redacted": true` and attaches a forensic advisory note (`"Geospatial metadata (GPS coordinates) detected and redacted for privacy."`), confirming the tag was present without exposing the user's location.
2. **Hardware Serial Number Masking**:
   - Device serial tags (`BodySerialNumber`, `LensSerialNumber`, `CameraSerialNumber`, `SerialNumber`, `InternalSerialNumber`) are masked to the last 4 characters (e.g., `Sony-****-7465`).
   - The engine flags `"serial_redacted": true` to protect against device fingerprinting.

---

## 4. C2PA / JUMBF Manifest Handling & Verification Semantics

### 4.1 Pure-Python JUMBF Container Scanner
The engine implements an in-memory byte scanner for ISO/IEC 19566-5 JUMBF (JPEG Universal Metadata Box Format) containers:
- **JPEG**: Scans `APP11` markers (`0xFF 0xEB`) containing `jumb` / `jumd` box headers and the standard C2PA UUID (`63327061-0011-0010-8000-00aa00389b71`).
- **PNG / WEBP**: Scans chunk markers for `c2pa` and `caPI` boxes.
- **Assertion Inspection**: Identifies top-level uncompressed assertions (`c2pa.actions`, `c2pa.hash.data`, `c2pa.thumbnail`) and claim generator strings (e.g., `Adobe Firefly`, `Truepic`, `Leica`).

### 4.2 Exact Verification Semantics
Crucially, SignalScope enforces strict cryptographic distinction:
- **`VALID_C2PA`**: Used **ONLY** when a genuine cryptographic X.509 PKI trust chain verification has succeeded via an active C2PA validator.
- **`UNVERIFIED_MANIFEST`**: Applied when a C2PA/JUMBF structure is detected in the image stream, but cryptographic signature verification has not been performed (pure-Python fallback mode without PKI root anchors).
- **`NOT_PRESENT`**: No C2PA / JUMBF markers detected.
- **`TAMPERED_OR_CORRUPT`**: C2PA envelope present but hash or signature parsing fails.

---

## 5. Structured Data Schema

The provenance output conforms to a deterministic, JSON-safe schema with top-level backward compatibility:

```json
{
  "c2pa_present": false,
  "exif_intact": true,
  "provenance": {
    "status": "EXIF_PRESENT",
    "format": "JPEG",
    "file_size_bytes": 145890,
    "image_dimensions": [1024, 768],
    "c2pa": {
      "present": false,
      "signature_valid": null,
      "claim_generator": null,
      "assertions": [],
      "validation_state": "NOT_PRESENT",
      "details": "No C2PA Content Credentials or JUMBF manifest detected in media stream."
    },
    "exif": {
      "present": true,
      "camera_make": "Sony",
      "camera_model": "ILCE-7RM4",
      "software": null,
      "datetime_original": "2024:05:12 14:30:00",
      "lens_model": null,
      "exposure_info": {
        "f_number": 2.8,
        "exposure_time": "1/250",
        "iso": 100,
        "focal_length": 35.0
      },
      "gps_redacted": false,
      "serial_redacted": false,
      "masked_serial": null
    },
    "generation_indicators": {
      "known_ai_software_flag": false,
      "generator_signature": null,
      "png_parameters_present": false
    },
    "forensic_notes": [
      "Hardware camera profile found: Sony ILCE-7RM4",
      "No C2PA Content Credentials or JUMBF manifest detected in media stream."
    ]
  }
}
```

### Status Hierarchy:
1. **`VALID_C2PA`**: Active cryptographic signature verified against trust anchors.
2. **`METADATA_SUSPICIOUS`**: Generative software signature or parameters detected.
3. **`EXIF_PRESENT`**: Camera hardware metadata intact.
4. **`STRIPPED`**: No EXIF or C2PA metadata found.
5. **`ERROR`**: Malformed file or unreadable container.

---

## 6. Forensic Interpretation & Hard Boundaries

Digital forensics requires strict adherence to evidentiary limits:

| Observation | Forensic Reality & Interpretation | SignalScope System Rule |
| :--- | :--- | :--- |
| **Missing EXIF** | Over 90% of web images (WhatsApp, X, Instagram, Reddit) strip EXIF during recompression. | **Never flags an image as AI simply because EXIF is stripped.** |
| **Intact Camera EXIF** | EXIF is an unauthenticated header and can be forged trivially with one line of Python. | **Never treats camera EXIF as proof of real-world authenticity.** |
| **Missing C2PA** | Most modern consumer cameras and mobile phones do not yet embed C2PA manifests. | **Never flags an image as untrusted or synthetic due to missing C2PA.** |
| **AI Software Tag Detected** | Provides strong contextual corroboration of AI generation tools. | **Reported as an auxiliary indicator; model visual score $p_{\text{AI}}$ remains objective.** |

---

## 7. Verification & Automated Test Results

The suite `tests/test_provenance.py` executes 19 comprehensive tests:
- **`test_synthetic_image_with_camera_exif`**: PASS
- **`test_image_with_no_exif`**: PASS
- **`test_png_with_stable_diffusion_parameters`**: PASS
- **`test_known_ai_software_signatures` (6 parameter variations)**: PASS
- **`test_ordinary_camera_software_metadata`**: PASS
- **`test_c2pa_jumbf_manifest_detection_unverified`**: PASS
- **`test_malformed_exif_safety`**: PASS
- **`test_gps_metadata_redaction`**: PASS
- **`test_serial_number_masking`**: PASS
- **`test_deterministic_output`**: PASS
- **`test_input_image_immutability`**: PASS
- **`test_unsupported_input_type`**: PASS
- **`test_prediction_contract_with_provenance`**: PASS
- **`test_provenance_does_not_modify_detector_verdict`**: PASS

**Regression Test Suites**:
- `tests/test_predict_contract.py`: 8/8 PASSED.
- `tests/test_robustness.py` & `tests/test_model_and_eval.py`: 26/26 PASSED.

---

## 8. Alignment with SIH Problem Statement
- **Bonus Capability Fulfilled**: Provides automated metadata and provenance analysis without compromising the frozen model.
- **Explainability & Attribution**: Detects generator metadata and tool signatures (Midjourney, Stable Diffusion, ComfyUI) to support attribution reporting.
- **Scientific Rigor**: Honestly distinguishes cryptographically verified signatures from unverified manifests and unauthenticated EXIF tags.
