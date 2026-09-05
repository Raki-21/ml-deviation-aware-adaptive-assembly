import numpy as np

from src.config import QUALITY_WEIGHTS, TOLERANCE_LIMITS


def calculate_quality_metrics(gap: np.ndarray) -> dict:
    abs_gap = np.abs(gap)

    mean_gap = float(np.mean(abs_gap))
    max_gap = float(np.max(abs_gap))
    parallelism_error = float(abs(gap[-1] - gap[0]))
    rms_deviation = float(np.sqrt(np.mean(gap ** 2)))

    return {
        "mean_gap": mean_gap,
        "max_gap": max_gap,
        "parallelism_error": parallelism_error,
        "rms_deviation": rms_deviation,
    }


def calculate_quality_score(metrics: dict) -> float:
    score = (
        QUALITY_WEIGHTS["mean_gap"] * metrics["mean_gap"]
        + QUALITY_WEIGHTS["max_gap"] * metrics["max_gap"]
        + QUALITY_WEIGHTS["parallelism_error"] * metrics["parallelism_error"]
        + QUALITY_WEIGHTS["rms_deviation"] * metrics["rms_deviation"]
    )

    return float(score)


def evaluate_tolerance_pass(metrics: dict) -> bool:
    return (
        metrics["mean_gap"] <= TOLERANCE_LIMITS["mean_gap"]
        and metrics["max_gap"] <= TOLERANCE_LIMITS["max_gap"]
        and metrics["parallelism_error"] <= TOLERANCE_LIMITS["parallelism_error"]
    )