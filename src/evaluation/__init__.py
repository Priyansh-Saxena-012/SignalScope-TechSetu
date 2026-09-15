"""SignalScope Evaluation Modules."""

from src.evaluation.evaluate import (
    compute_metrics,
    evaluate_with_generator_breakdown,
    format_evaluation_report,
)
from src.evaluation.robustness import (
    RobustnessCondition,
    SampleRecord,
    apply_downscale_restore,
    apply_gaussian_blur,
    apply_jpeg_compression,
    apply_photometric_edit,
    apply_screenshot_proxy,
    compute_robustness_metrics,
    get_standard_robustness_conditions,
    sample_deterministic_dev_set,
    save_robustness_artifacts,
)

__all__ = [
    "compute_metrics",
    "evaluate_with_generator_breakdown",
    "format_evaluation_report",
    "RobustnessCondition",
    "SampleRecord",
    "apply_jpeg_compression",
    "apply_downscale_restore",
    "apply_gaussian_blur",
    "apply_screenshot_proxy",
    "apply_photometric_edit",
    "get_standard_robustness_conditions",
    "sample_deterministic_dev_set",
    "compute_robustness_metrics",
    "save_robustness_artifacts",
]
