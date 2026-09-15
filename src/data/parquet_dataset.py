r"""SignalScope Tiny-GenImage Parquet Dataset Loader.

Provides a memory-efficient, PyTorch-compatible Dataset for loading images,
binary labels, and generator metadata from Parquet files (Tiny-GenImage benchmark).

Authoritative Dataset Specification:
As documented in the official dataset card (C:\Datasets\SignalScope_Train\README.md):
- Task A: Binary Veracity Classification (`label`)
    0: "real" (Image is a real photograph / non-AI generated)
    1: "fake" (Image was created by an AI generation model)
- Task B: AI Model Source Identification (`generator`)
    0: "Real"
    1: "ADM" (Ablated Diffusion Model)
    2: "BigGAN"
    3: "GLIDE"
    4: "Midjourney"
    5: "SD14" (Stable Diffusion 1.4; absent from Tiny-GenImage train/val splits;
               not to be confused with the separate official 100k held-out benchmark)
    6: "SD15" (Stable Diffusion 1.5)
    7: "VQDM" (Vector Quantized Diffusion Model)
    8: "Wukong"

Features:
- Minimal, robust lazy loading: reads records on demand via DuckDB.
- Windows multiprocessing compatibility: seamlessly supports PyTorch DataLoader
  multi-worker pipelines (spawn context) via lazy connection instantiation.
- Strict test-set isolation: rejects any path pointing to or within the held-out benchmark.
- Configurable sources: file path, directory, glob pattern, or sequence of files.
- Preserves raw labels, semantic label names, and generator metadata.
"""

from __future__ import annotations

import glob
import io
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import duckdb
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.path_safety import TestSetContaminationError, validate_path_safety


# Authoritative Label Definitions (from C:\Datasets\SignalScope_Train\README.md)
TINY_GENIMAGE_LABEL_MAP: Dict[int, str] = {
    0: "real",
    1: "fake",
}

# Authoritative Generator Model Definitions (from C:\Datasets\SignalScope_Train\README.md)
# Note: Generator 5 ("sd14") is defined in the dataset schema but absent from Tiny-GenImage
# train/val splits. The official held-out benchmark is the separate 100k test set.
TINY_GENIMAGE_GENERATOR_MAP: Dict[int, str] = {
    0: "real",
    1: "adm",
    2: "biggan",
    3: "glide",
    4: "midjourney",
    5: "sd14",
    6: "sd15",
    7: "vqdm",
    8: "wukong",
}

DEFAULT_TINY_GENIMAGE_TRAIN_SOURCE = r"C:\Datasets\SignalScope_Train\data\train-*.parquet"
DEFAULT_TINY_GENIMAGE_VAL_SOURCE = r"C:\Datasets\SignalScope_Train\data\validation-*.parquet"


def resolve_parquet_files(
    source: Union[str, Path, Sequence[Union[str, Path]]],
    held_out_test_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Resolve and validate a list of Parquet file paths from a path, glob, or sequence.

    Parameters
    ----------
    source : str, Path, or sequence of such
        File path, directory path, glob pattern, or sequence of files.
    held_out_test_dir : str or Path, optional
        Protected test set path to validate against.

    Returns
    -------
    list of str
        Sorted list of normalized absolute Parquet file paths.
    """
    files: List[str] = []

    if isinstance(source, (str, Path)):
        src_str = str(source)
        # Check test set isolation before globbing or listing
        validate_path_safety(src_str, protected_paths=held_out_test_dir, context_desc="parquet dataset source")

        p = Path(src_str)
        if "*" in src_str or "?" in src_str:
            matches = glob.glob(src_str)
            files = sorted(matches)
        elif p.is_dir():
            files = sorted(str(f) for f in p.glob("*.parquet"))
        elif p.is_file():
            files = [str(p)]
        else:
            matches = glob.glob(src_str)
            files = sorted(matches)
    elif isinstance(source, (list, tuple, set)):
        for item in source:
            item_str = str(item)
            validate_path_safety(item_str, protected_paths=held_out_test_dir, context_desc="parquet dataset source")
            if Path(item_str).is_file():
                files.append(str(Path(item_str).resolve()))
        files = sorted(list(set(files)))
    else:
        raise TypeError(f"Unsupported source type: {type(source)}. Expected str, Path, or Sequence.")

    if not files:
        raise FileNotFoundError(f"No Parquet files found matching source: '{source}'")

    # Validate each resolved file for test-set isolation
    for f in files:
        validate_path_safety(f, protected_paths=held_out_test_dir, context_desc="resolved parquet file")

    return [os.path.normpath(f) for f in files]


class TinyGenImageParquetDataset(Dataset):
    """PyTorch Dataset for reading Tiny-GenImage Parquet datasets lazily via DuckDB.

    Parameters
    ----------
    source : str, Path, or sequence of such
        Parquet file path, directory path, glob pattern, or sequence of files.
    transform : callable, optional
        Torchvision image transformation pipeline.
    held_out_test_dir : str or Path, optional
        Protected test set path to validate isolation against.
    """

    def __init__(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        transform: Optional[Callable] = None,
        held_out_test_dir: Optional[Union[str, Path]] = None,
    ):
        self.source = source
        self.transform = transform
        self.held_out_test_dir = held_out_test_dir

        # 1. Resolve and validate files against test-set isolation
        self.files = resolve_parquet_files(source, held_out_test_dir=held_out_test_dir)

        # 2. Query file-level row counts to build cumulative index map
        self._file_rows: List[int] = []
        self._cumulative_rows: List[int] = []
        self._total_rows = 0
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

        self._build_index_map()

    def _build_index_map(self) -> None:
        """Query metadata across files to build cumulative file row offsets."""
        conn = duckdb.connect()
        try:
            total = 0
            for f in self.files:
                f_sql = f.replace("\\", "/")
                # Fast metadata query summing row counts across row groups (column_id=0 avoids duplicate counts)
                query = f"""
                    SELECT COALESCE(SUM(row_group_num_rows), 0)
                    FROM parquet_metadata('{f_sql}')
                    WHERE column_id = 0
                """
                count = conn.execute(query).fetchone()[0]
                count_int = int(count)
                self._file_rows.append(count_int)
                total += count_int
                self._cumulative_rows.append(total)

            self._total_rows = total
        finally:
            conn.close()

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        """Lazily initialize process-safe DuckDB connection."""
        if self._conn is None:
            self._conn = duckdb.connect()
        return self._conn

    def __len__(self) -> int:
        return self._total_rows

    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickling state dropping live DB connections for multiprocessing worker spawn."""
        state = self.__dict__.copy()
        state["_conn"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom unpickling state restoring clean connection handle."""
        self.__dict__.update(state)
        self._conn = None

    def __getitem__(self, idx: int) -> Tuple[Any, int, Dict[str, Any]]:
        """Retrieve a single sample by index.

        Parameters
        ----------
        idx : int
            Sample index in [0, len(dataset) - 1].

        Returns
        -------
        image : PIL.Image.Image or torch.Tensor
            Decoded image (or transformed tensor if transform provided).
        label : int
            Raw binary label: 0 for Real, 1 for Fake/Synthetic.
        meta : dict
            Sample metadata containing:
            - 'path': original filename/path string
            - 'generator': human-readable generator name string (e.g. 'adm', 'real')
            - 'generator_id': integer generator code (0 to 8)
            - 'label_name': semantic label string ('real' or 'fake')
            - 'is_ai': boolean flag (True if label == 1)
        """
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._total_rows})")

        file_idx = bisect_right(self._cumulative_rows, idx)
        file_start = self._cumulative_rows[file_idx - 1] if file_idx > 0 else 0
        file_offset = idx - file_start
        file_path = self.files[file_idx]

        f_sql = file_path.replace("\\", "/")
        query = f"""
            SELECT (image).bytes, (image).path, label, generator 
            FROM '{f_sql}' 
            LIMIT 1 OFFSET {file_offset}
        """
        row = self._get_conn().execute(query).fetchone()
        if row is None:
            raise IndexError(f"Failed to read row at offset {file_offset} in {file_path}")

        raw_bytes, path_str, label, gen_id = row

        # Decode image bytes to PIL Image
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # Apply optional transforms
        if self.transform is not None:
            image = self.transform(image)

        gen_id_int = int(gen_id)
        label_int = int(label)
        gen_name = TINY_GENIMAGE_GENERATOR_MAP.get(gen_id_int, f"generator_{gen_id_int}")
        label_name = TINY_GENIMAGE_LABEL_MAP.get(label_int, f"label_{label_int}")

        meta = {
            "path": path_str,
            "generator": gen_name,
            "generator_id": gen_id_int,
            "label_name": label_name,
            "is_ai": bool(label_int == 1),
        }
        return image, label_int, meta

    def get_summary(self) -> Dict[str, Any]:
        """Compute dataset summary statistics without loading image bytes into RAM."""
        conn = duckdb.connect()
        try:
            files_sql = ", ".join(f"'{f.replace(chr(92), '/')}'" for f in self.files)
            dist_query = f"""
                SELECT label, generator, COUNT(*) as count 
                FROM read_parquet([{files_sql}]) 
                GROUP BY label, generator 
                ORDER BY label, generator
            """
            dist_rows = conn.execute(dist_query).fetchall()

            generators_breakdown: Dict[str, int] = {}
            real_count = 0
            fake_count = 0

            for lbl, gid, cnt in dist_rows:
                gname = TINY_GENIMAGE_GENERATOR_MAP.get(gid, f"generator_{gid}")
                generators_breakdown[gname] = generators_breakdown.get(gname, 0) + cnt
                if lbl == 0:
                    real_count += cnt
                else:
                    fake_count += cnt

            return {
                "total_samples": self._total_rows,
                "real_samples": real_count,
                "synthetic_samples": fake_count,
                "num_files": len(self.files),
                "generators_breakdown": generators_breakdown,
            }
        finally:
            conn.close()


def create_tiny_genimage_datasets(
    train_source: Union[str, Path, Sequence[Union[str, Path]]] = DEFAULT_TINY_GENIMAGE_TRAIN_SOURCE,
    val_source: Union[str, Path, Sequence[Union[str, Path]]] = DEFAULT_TINY_GENIMAGE_VAL_SOURCE,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    held_out_test_dir: Optional[Union[str, Path]] = None,
) -> Tuple[TinyGenImageParquetDataset, TinyGenImageParquetDataset]:
    """Construct independent training and validation TinyGenImageParquetDataset instances.

    Parameters
    ----------
    train_source : str, Path, or sequence of such
        Source path or glob for training parquet files.
    val_source : str, Path, or sequence of such
        Source path or glob for validation parquet files.
    train_transform : callable, optional
    val_transform : callable, optional
    held_out_test_dir : str or Path, optional

    Returns
    -------
    tuple of (TinyGenImageParquetDataset, TinyGenImageParquetDataset)
        Independent (train_dataset, val_dataset) instances.
    """
    train_ds = TinyGenImageParquetDataset(
        source=train_source,
        transform=train_transform,
        held_out_test_dir=held_out_test_dir,
    )
    val_ds = TinyGenImageParquetDataset(
        source=val_source,
        transform=val_transform,
        held_out_test_dir=held_out_test_dir,
    )
    return train_ds, val_ds
