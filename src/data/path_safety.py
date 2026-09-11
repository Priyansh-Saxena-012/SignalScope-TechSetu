"""SignalScope Test-Set Isolation & Path Safety Validator.

Prevents training, validation, or development pipelines from contaminating
their data with the official held-out evaluation dataset.

Rules Enforced:
1. Training/development data root MUST NOT be identical to the held-out test directory.
2. Training/development data root MUST NOT be a parent/ancestor directory of the test directory
   (which would cause recursive crawling like Path.rglob to crawl into the test set).
3. Training/development data root MUST NOT be a subdirectory of the test directory.
4. Symlinks, redundant relative segments (..), and case differences are fully normalized.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Union


# Default protected test directories that must NEVER be crawled or used in development
DEFAULT_PROTECTED_TEST_DIRS: List[str] = [
    r"C:\Datasets\SignalScope\test",
]


class TestSetContaminationError(ValueError):
    """Raised when a dataset path violates test-set isolation."""
    __test__ = False


def normalize_path(p: Union[str, Path]) -> Path:
    """Normalize path for robust cross-platform comparison (case, slashes, realpath)."""
    abs_p = os.path.normcase(os.path.realpath(os.path.abspath(str(p))))
    return Path(abs_p)


def get_protected_test_dirs(
    custom_protected_paths: Optional[Union[str, Path, Iterable[Union[str, Path]]]] = None
) -> List[Path]:
    """Compile the list of protected test directories from defaults, env vars, and arguments."""
    protected: List[Path] = []

    # 1. Environment variable override/addition
    env_dir = os.environ.get("SIGNALSCOPE_HELD_OUT_TEST_DIR") or os.environ.get("HELD_OUT_TEST_DIR")
    if env_dir:
        protected.append(normalize_path(env_dir))

    # 2. Configured custom paths
    if custom_protected_paths is not None:
        if isinstance(custom_protected_paths, (str, Path)):
            protected.append(normalize_path(custom_protected_paths))
        else:
            for cp in custom_protected_paths:
                if cp:
                    protected.append(normalize_path(cp))

    # 3. Built-in defaults (e.g. C:\Datasets\SignalScope\test)
    for default_dir in DEFAULT_PROTECTED_TEST_DIRS:
        norm_default = normalize_path(default_dir)
        if norm_default not in protected:
            protected.append(norm_default)

    return protected


def is_path_contaminated(
    candidate_path: Union[str, Path],
    protected_paths: Optional[Union[str, Path, Iterable[Union[str, Path]]]] = None,
) -> bool:
    """Check if candidate_path overlaps with, contains, or is contained by any protected test directory."""
    if candidate_path is None or str(candidate_path).strip() == "":
        return False

    cand = normalize_path(candidate_path)
    protected_list = get_protected_test_dirs(protected_paths)

    for prot in protected_list:
        if cand == prot:
            return True
        try:
            if prot.is_relative_to(cand):
                return True
        except (ValueError, AttributeError):
            pass
        try:
            if cand.is_relative_to(prot):
                return True
        except (ValueError, AttributeError):
            pass

    return False


def validate_path_safety(
    candidate_path: Union[str, Path],
    protected_paths: Optional[Union[str, Path, Iterable[Union[str, Path]]]] = None,
    context_desc: str = "dataset directory",
) -> None:
    """Validate that candidate_path does not violate test-set isolation.

    Parameters
    ----------
    candidate_path : str or Path
        Candidate training or development directory path to check.
    protected_paths : str, Path, or iterable of such, optional
        Explicit protected test directories to enforce isolation against.
        Merged with environment variables and DEFAULT_PROTECTED_TEST_DIRS.
    context_desc : str, default="dataset directory"
        Descriptive context for error messages.

    Raises
    ------
    TestSetContaminationError
        If candidate_path matches, contains, or is inside a protected test directory.
    """
    if candidate_path is None or str(candidate_path).strip() == "":
        return

    cand = normalize_path(candidate_path)
    protected_list = get_protected_test_dirs(protected_paths)

    for prot in protected_list:
        relationship = None
        if cand == prot:
            relationship = "is identical to"
        else:
            try:
                if prot.is_relative_to(cand):
                    relationship = "is a parent/ancestor directory containing"
            except (ValueError, AttributeError):
                pass

            if not relationship:
                try:
                    if cand.is_relative_to(prot):
                        relationship = "is located inside"
                except (ValueError, AttributeError):
                    pass

        if relationship:
            raise TestSetContaminationError(
                f"[FATAL TEST CONTAMINATION DETECTED]\n"
                f"The specified {context_desc} violates test-set isolation:\n"
                f"  Candidate Path : '{candidate_path}' (normalized: '{cand}')\n"
                f"  Protected Test : '{prot}'\n"
                f"Reason: Candidate path {relationship} the official held-out test benchmark.\n"
                f"The 100,000-image held-out evaluation dataset must NEVER be used for training,\n"
                f"validation, model selection, fitting, or development crawling.\n"
                f"Please specify a separate, isolated training dataset directory."
            )
