"""SignalScope Data Modules."""

from src.data.parquet_dataset import (
    DEFAULT_TINY_GENIMAGE_TRAIN_SOURCE,
    DEFAULT_TINY_GENIMAGE_VAL_SOURCE,
    TINY_GENIMAGE_GENERATOR_MAP,
    TINY_GENIMAGE_LABEL_MAP,
    TinyGenImageParquetDataset,
    create_tiny_genimage_datasets,
    resolve_parquet_files,
)
from src.data.path_safety import (
    DEFAULT_PROTECTED_TEST_DIRS,
    TestSetContaminationError,
    is_path_contaminated,
    validate_path_safety,
)

__all__ = [
    "DEFAULT_PROTECTED_TEST_DIRS",
    "DEFAULT_TINY_GENIMAGE_TRAIN_SOURCE",
    "DEFAULT_TINY_GENIMAGE_VAL_SOURCE",
    "TINY_GENIMAGE_GENERATOR_MAP",
    "TINY_GENIMAGE_LABEL_MAP",
    "TestSetContaminationError",
    "TinyGenImageParquetDataset",
    "create_tiny_genimage_datasets",
    "is_path_contaminated",
    "resolve_parquet_files",
    "validate_path_safety",
]
