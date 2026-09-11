"""SignalScope Dataset Inspection Tool.

Performs exhaustive inspection of the provided dataset without assuming a fixed
directory structure, specific generator names, or prompt metadata:
- Total image count and file formats
- Real vs. AI/Synthetic class counts and balance
- Directory structure and tree depth
- Generator discovery and distribution (counts per generator)
- Metadata files (.csv, .json, .parquet) and prompt text scanning
- File integrity checks and exact duplicate risk analysis via hashing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Known generator keywords across major diffusion and GAN benchmarks (including GenImage)
KNOWN_GENERATOR_KEYWORDS = [
    "stable_diffusion", "sdxl", "sd_v1_4", "sd_v1_5", "sd",
    "midjourney", "dalle", "dalle3", "flux", "gan", "stylegan",
    "biggan", "wukong", "adm", "glide", "vqdm"
]


def compute_file_hash(path: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for exact-duplicate detection."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_class_from_path(p: Path) -> Optional[str]:
    """Determine class label ('real', 'synthetic', or None) from directory path parts.

    Supports diverse real-vs-fake benchmarks including CIFAKE, GenImage, and ImageNet:
    - Real cues: 'real', 'nature', 'natural', 'authentic', 'photo', 'original', '0_real'
    - Fake cues: 'fake', 'synthetic', 'ai', 'generated', 'synth', '1_fake'
    """
    parts = [part.lower() for part in p.parts[:-1]]
    # Check innermost folder first
    for part in reversed(parts):
        if part in {"real", "nature", "natural", "authentic", "photo", "original", "0_real", "0-real"}:
            return "real"
        if part in {"fake", "synthetic", "ai", "generated", "synth", "1_fake", "1-fake"}:
            return "synthetic"
        if part.startswith("0_") and "real" in part:
            return "real"
        if part.startswith("1_") and "fake" in part:
            return "synthetic"
        if "real" in part and "unreal" not in part and "surreal" not in part:
            return "real"
        if any(fk in part for fk in ["fake", "synth", "synthetic"]):
            return "synthetic"
    return None


def detect_generator_from_path(p: Path, root_path: Path) -> Optional[str]:
    """Infer generator identifier from directory hierarchy or filename."""
    try:
        rel_parts = [part.lower() for part in p.relative_to(root_path).parts[:-1]]
    except Exception:
        rel_parts = [part.lower() for part in p.parts[:-1]]

    # 1. Match against known generator keyword list
    for part in rel_parts:
        for kw in KNOWN_GENERATOR_KEYWORDS:
            if kw == part or kw in part:
                return kw

    # 2. Dynamic inference: In GenImage-style hierarchies (<generator>/<split>/<class>/image.jpg),
    # identify the non-standard partition folder
    standard_structural_names = {
        "train", "val", "test", "real", "fake", "ai", "nature", "natural",
        "0_real", "1_fake", "images", "data", "dataset", "subsets"
    }
    for part in rel_parts:
        if part not in standard_structural_names and not part.isdigit():
            return part

    return None


def inspect_dataset(data_dir: str, sample_hash_limit: int = 500) -> Dict[str, Any]:
    """Inspect dataset directory structure, formats, labels, and metadata.

    Parameters
    ----------
    data_dir : str
        Path to the root of the dataset directory.
    sample_hash_limit : int, default=500
        Maximum number of images to sample per class for duplicate and integrity checks.

    Returns
    -------
    dict
        Comprehensive inspection report dictionary.
    """
    root_path = Path(data_dir)
    report: Dict[str, Any] = {
        "target_directory": str(root_path.resolve()),
        "status": "ok",
        "exists": root_path.exists(),
        "directory_structure": {},
        "image_formats": {},
        "total_images": 0,
        "class_labels": {},
        "class_balance_ratio": None,
        "metadata_files_found": [],
        "generator_metadata": {
            "has_generator_labels": False,
            "detected_generators": [],
            "source": None,
        },
        "generator_distribution": {},
        "prompt_metadata": {
            "has_prompt_text": False,
            "sample_prompts": [],
            "source": None,
        },
        "duplicate_risk": {
            "sample_size": 0,
            "exact_duplicates_found": 0,
            "duplicate_pairs": [],
        },
        "corrupt_images": 0,
        "unsupported_files": 0,
    }

    if not root_path.exists():
        report["status"] = "dataset_not_found"
        report["error"] = f"Directory not found: {data_dir}"
        return report

    # 1. Scan directory structure
    subdirs = [p for p in root_path.rglob("*") if p.is_dir()]
    report["directory_structure"] = {
        "total_subdirectories": len(subdirs),
        "subdirectory_names": sorted(list({p.name for p in subdirs})),
        "relative_tree": [str(p.relative_to(root_path)) for p in subdirs[:30]],
    }

    # 2. Check for metadata files (.csv, .json, .parquet, .txt)
    meta_extensions = {".csv", ".json", ".parquet", ".tsv", ".yaml", ".yml"}
    metadata_files = [p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in meta_extensions]
    report["metadata_files_found"] = [str(p.relative_to(root_path)) for p in metadata_files]

    # 3. Analyze images, formats, classes, and generator distribution
    class_counter: Counter = Counter()
    format_counter: Counter = Counter()
    generator_counter: Counter = Counter()
    detected_generator_tags = set()
    image_paths: List[Path] = []

    for p in root_path.rglob("*"):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()
        if suffix in VALID_IMAGE_EXTENSIONS:
            format_counter[suffix] += 1
            image_paths.append(p)

            # Determine class label
            cls = detect_class_from_path(p)
            if cls == "real":
                class_counter["real"] += 1
            elif cls == "synthetic":
                class_counter["synthetic"] += 1
                # Infer generator for synthetic images
                gen = detect_generator_from_path(p, root_path)
                if gen:
                    generator_counter[gen] += 1
                    detected_generator_tags.add(gen)
                else:
                    generator_counter["unspecified"] += 1
            else:
                class_counter["unassigned_or_ambiguous"] += 1
        else:
            if suffix not in meta_extensions:
                report["unsupported_files"] += 1

    report["total_images"] = len(image_paths)
    report["image_formats"] = dict(format_counter)
    report["class_labels"] = dict(class_counter)
    report["generator_distribution"] = dict(generator_counter)

    # Compute class balance ratio
    if class_counter["real"] > 0 and class_counter["synthetic"] > 0:
        ratio = class_counter["synthetic"] / class_counter["real"]
        report["class_balance_ratio"] = round(ratio, 4)

    # 4. Check for generator & prompt metadata in sidecars or tables
    if metadata_files:
        for mf in metadata_files:
            try:
                if mf.suffix.lower() == ".csv":
                    import pandas as pd
                    df = pd.read_csv(mf, nrows=10)
                    cols = [c.lower() for c in df.columns]
                    if any(k in c for c in cols for k in ["generator", "model", "engine"]):
                        report["generator_metadata"]["has_generator_labels"] = True
                        report["generator_metadata"]["source"] = str(mf.name)
                    if any(k in c for c in cols for k in ["prompt", "caption", "text"]):
                        report["prompt_metadata"]["has_prompt_text"] = True
                        report["prompt_metadata"]["source"] = str(mf.name)
                elif mf.suffix.lower() == ".json":
                    with open(mf, "r", encoding="utf-8") as jf:
                        sample_data = json.load(jf)
                    sample_str = str(sample_data)[:500].lower()
                    if "generator" in sample_str:
                        report["generator_metadata"]["has_generator_labels"] = True
                        report["generator_metadata"]["source"] = str(mf.name)
                    if "prompt" in sample_str or "caption" in sample_str:
                        report["prompt_metadata"]["has_prompt_text"] = True
                        report["prompt_metadata"]["source"] = str(mf.name)
            except Exception:
                pass

    if detected_generator_tags:
        report["generator_metadata"]["has_generator_labels"] = True
        report["generator_metadata"]["detected_generators"] = sorted(list(detected_generator_tags))
        if not report["generator_metadata"]["source"]:
            report["generator_metadata"]["source"] = "filepath_inference"

    # 5. Integrity & Duplicate Risk Analysis (Sample-based hashing & PIL verification)
    hashes: Dict[str, str] = {}
    sample_to_check = image_paths[:sample_hash_limit]
    report["duplicate_risk"]["sample_size"] = len(sample_to_check)

    for img_path in sample_to_check:
        try:
            # Check basic image readability
            with Image.open(img_path) as im:
                im.verify()

            h = compute_file_hash(str(img_path))
            if h in hashes:
                report["duplicate_risk"]["exact_duplicates_found"] += 1
                report["duplicate_risk"]["duplicate_pairs"].append(
                    (str(img_path.name), str(hashes[h]))
                )
            else:
                hashes[h] = img_path.name
        except Exception:
            report["corrupt_images"] += 1

    return report


def print_inspection_report(report: Dict[str, Any]) -> None:
    """Print human-readable formatted inspection report."""
    print("=" * 70)
    print("SIGNALSCOPE DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"Target Directory       : {report.get('target_directory')}")
    print(f"Directory Status       : {report.get('status')}")

    if report.get("status") == "dataset_not_found":
        print("\n[WARNING] Dataset directory not found on disk.")
        print(f"Error: {report.get('error')}")
        print("=" * 70)
        return

    print(f"Total Images Found     : {report.get('total_images')}")
    print(f"Image Formats          : {report.get('image_formats')}")
    print(f"Class Labels Counts    : {report.get('class_labels')}")
    print(f"Class Balance Ratio    : {report.get('class_balance_ratio')} (Synthetic / Real)")
    print(f"Metadata Files Found   : {report.get('metadata_files_found')}")

    gen_meta = report.get("generator_metadata", {})
    print(f"Generator Labels Found : {gen_meta.get('has_generator_labels')} (Source: {gen_meta.get('source')})")
    print(f"Generator Distribution : {report.get('generator_distribution')}")

    prompt_meta = report.get("prompt_metadata", {})
    print(f"Prompt Metadata Found  : {prompt_meta.get('has_prompt_text')} (Source: {prompt_meta.get('source')})")

    dup_meta = report.get("duplicate_risk", {})
    print(f"Exact Duplicate Check  : {dup_meta.get('exact_duplicates_found')} duplicates found in sample of {dup_meta.get('sample_size')}")
    print(f"Corrupt / Unreadable   : {report.get('corrupt_images')}")
    print(f"Unsupported Files      : {report.get('unsupported_files')}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SignalScope dataset structure and metadata.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to dataset directory (default: data)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save inspection report as JSON."
    )

    args = parser.parse_args()
    report = inspect_dataset(args.data_dir)
    print_inspection_report(report)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output_json}")


if __name__ == "__main__":
    main()
