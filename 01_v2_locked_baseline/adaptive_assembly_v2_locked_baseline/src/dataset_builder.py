import numpy as np
import pandas as pd

from src.config import (
    DEVIATION_RANGES,
    DECISION_RANGES,
    DEFAULT_DATASET_SIZE,
    RANDOM_STATE,
)
from src.deviation_generator import generate_nominal_profile, generate_deviation_profile
from src.assembly_model import apply_assembly_correction, calculate_gap
from src.quality_metrics import (
    calculate_quality_metrics,
    calculate_quality_score,
    evaluate_tolerance_pass,
)


def sample_uniform(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    return float(rng.uniform(value_range[0], value_range[1]))


def generate_single_sample(
    case_id: int,
    rng: np.random.Generator
) -> dict:
    dev_offset = sample_uniform(rng, DEVIATION_RANGES["dev_offset"])
    dev_tilt = sample_uniform(rng, DEVIATION_RANGES["dev_tilt"])
    dev_bend = sample_uniform(rng, DEVIATION_RANGES["dev_bend"])
    dev_waviness = sample_uniform(rng, DEVIATION_RANGES["dev_waviness"])
    dev_noise = sample_uniform(rng, DEVIATION_RANGES["dev_noise"])

    z_adj = sample_uniform(rng, DECISION_RANGES["z_adj"])
    theta_adj = sample_uniform(rng, DECISION_RANGES["theta_adj"])
    locator_offset = sample_uniform(rng, DECISION_RANGES["locator_offset"])

    reference_profile = generate_nominal_profile()

    deviated_profile = generate_deviation_profile(
        dev_offset=dev_offset,
        dev_tilt=dev_tilt,
        dev_bend=dev_bend,
        dev_waviness=dev_waviness,
        dev_noise=dev_noise,
        random_seed=case_id,
    )

    corrected_profile = apply_assembly_correction(
        deviated_profile=deviated_profile,
        z_adj=z_adj,
        theta_adj=theta_adj,
        locator_offset=locator_offset,
    )

    gap = calculate_gap(reference_profile, corrected_profile)

    metrics = calculate_quality_metrics(gap)
    quality_score = calculate_quality_score(metrics)
    tolerance_pass = evaluate_tolerance_pass(metrics)

    return {
        "case_id": case_id,
        "dev_offset": dev_offset,
        "dev_tilt": dev_tilt,
        "dev_bend": dev_bend,
        "dev_waviness": dev_waviness,
        "dev_noise": dev_noise,
        "z_adj": z_adj,
        "theta_adj": theta_adj,
        "locator_offset": locator_offset,
        "mean_gap": metrics["mean_gap"],
        "max_gap": metrics["max_gap"],
        "parallelism_error": metrics["parallelism_error"],
        "rms_deviation": metrics["rms_deviation"],
        "quality_score": quality_score,
        "tolerance_pass": int(tolerance_pass),
    }


def generate_dataset(
    n_samples: int = DEFAULT_DATASET_SIZE,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    rows = []
    for case_id in range(n_samples):
        row = generate_single_sample(case_id, rng)
        rows.append(row)

    return pd.DataFrame(rows)