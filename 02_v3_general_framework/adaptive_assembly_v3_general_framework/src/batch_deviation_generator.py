"""
Batch-aware deviation dataset generator for Version 3.

This module extends the Version 2 single-case prototype into a batch-aware
dataset structure.

Version 3 extension:
- offset_dominant
- tilt_dominant
- bend_dominant
- waviness_dominant
- mixed
- twist_dominant
- fixture_drift
- disturbed_batch

Important:
twist_dominant and fixture_drift are synthetic controlled extension scenarios.
They are not claimed as measured industrial data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Basic geometry and quality settings
# -------------------------------------------------------------------------

PROFILE_LENGTH_MM = 1000.0
N_POINTS = 100

QUALITY_WEIGHTS = {
    "mean_gap": 0.30,
    "max_gap": 0.30,
    "parallelism_error": 0.30,
    "rms_deviation": 0.10,
}

ASSEMBLY_PARAMETER_RANGES = {
    "z_adj": (-2.5, 2.5),
    "theta_adj": (-1.2, 1.2),
    "locator_offset": (-1.0, 1.0),
}

SCENARIO_TYPES = [
    "offset_dominant",
    "tilt_dominant",
    "bend_dominant",
    "waviness_dominant",
    "mixed",
    "twist_dominant",
    "fixture_drift",
]

SEVERITY_LEVELS = ["low", "medium", "high"]


@dataclass
class DeviationFeatures:
    """Container for one generated deviation case."""

    dev_offset: float
    dev_tilt: float
    dev_bend: float
    dev_waviness: float
    dev_noise: float
    dev_twist: float
    fixture_drift: float


def _severity_factor(severity_level: str) -> float:
    """
    Convert severity label into a numerical multiplier.
    """

    severity_map = {
        "low": 0.45,
        "medium": 0.75,
        "high": 1.00,
    }

    if severity_level not in severity_map:
        raise ValueError(f"Unknown severity_level: {severity_level}")

    return severity_map[severity_level]


def generate_deviation_features(
    scenario_type: str,
    severity_level: str,
    disturbance_flag: str,
    rng: np.random.Generator,
) -> DeviationFeatures:
    """
    Generate physically interpretable deviation features.

    New Version 3 extension:
    - dev_twist: controlled torsion-like profile contribution
    - fixture_drift: systematic process/locator drift contribution
    """

    sf = _severity_factor(severity_level)

    # Base background deviations
    dev_offset = rng.uniform(-0.4, 0.4) * sf
    dev_tilt = rng.uniform(-0.2, 0.2) * sf
    dev_bend = rng.uniform(-0.3, 0.3) * sf
    dev_waviness = rng.uniform(0.0, 0.2) * sf
    dev_noise = rng.uniform(0.02, 0.06) * sf
    dev_twist = rng.uniform(-0.15, 0.15) * sf
    fixture_drift = rng.uniform(-0.10, 0.10) * sf

    if scenario_type == "offset_dominant":
        dev_offset = rng.uniform(-2.0, 2.0) * sf
        dev_tilt = rng.uniform(-0.15, 0.15) * sf
        dev_bend = rng.uniform(-0.25, 0.25) * sf
        dev_waviness = rng.uniform(0.0, 0.15) * sf
        dev_twist = rng.uniform(-0.10, 0.10) * sf
        fixture_drift = rng.uniform(-0.10, 0.10) * sf

    elif scenario_type == "tilt_dominant":
        dev_offset = rng.uniform(-0.4, 0.4) * sf
        dev_tilt = rng.uniform(-1.0, 1.0) * sf
        dev_bend = rng.uniform(-0.25, 0.25) * sf
        dev_waviness = rng.uniform(0.0, 0.15) * sf
        dev_twist = rng.uniform(-0.10, 0.10) * sf
        fixture_drift = rng.uniform(-0.10, 0.10) * sf

    elif scenario_type == "bend_dominant":
        dev_offset = rng.uniform(-0.4, 0.4) * sf
        dev_tilt = rng.uniform(-0.2, 0.2) * sf
        dev_bend = rng.uniform(-1.5, 1.5) * sf
        dev_waviness = rng.uniform(0.0, 0.20) * sf
        dev_twist = rng.uniform(-0.15, 0.15) * sf
        fixture_drift = rng.uniform(-0.10, 0.10) * sf

    elif scenario_type == "waviness_dominant":
        dev_offset = rng.uniform(-0.4, 0.4) * sf
        dev_tilt = rng.uniform(-0.2, 0.2) * sf
        dev_bend = rng.uniform(-0.3, 0.3) * sf
        dev_waviness = rng.uniform(0.35, 0.8) * sf
        dev_twist = rng.uniform(-0.15, 0.15) * sf
        fixture_drift = rng.uniform(-0.10, 0.10) * sf

    elif scenario_type == "mixed":
        dev_offset = rng.uniform(-1.4, 1.4) * sf
        dev_tilt = rng.uniform(-0.75, 0.75) * sf
        dev_bend = rng.uniform(-1.0, 1.0) * sf
        dev_waviness = rng.uniform(0.0, 0.6) * sf
        dev_twist = rng.uniform(-0.5, 0.5) * sf
        fixture_drift = rng.uniform(-0.30, 0.30) * sf

    elif scenario_type == "twist_dominant":
        # Synthetic controlled torsion-like deviation.
        # This represents a more complex shape mode than pure tilt/bend.
        dev_offset = rng.uniform(-0.3, 0.3) * sf
        dev_tilt = rng.uniform(-0.25, 0.25) * sf
        dev_bend = rng.uniform(-0.4, 0.4) * sf
        dev_waviness = rng.uniform(0.0, 0.25) * sf
        dev_twist = rng.uniform(-1.2, 1.2) * sf
        fixture_drift = rng.uniform(-0.10, 0.10) * sf

    elif scenario_type == "fixture_drift":
        # Synthetic controlled process/locator drift scenario.
        # This represents systematic setup-related shift, not measured real drift.
        dev_offset = rng.uniform(-0.6, 0.6) * sf
        dev_tilt = rng.uniform(-0.35, 0.35) * sf
        dev_bend = rng.uniform(-0.4, 0.4) * sf
        dev_waviness = rng.uniform(0.0, 0.25) * sf
        dev_twist = rng.uniform(-0.2, 0.2) * sf
        fixture_drift = rng.uniform(-1.0, 1.0) * sf

    elif scenario_type == "disturbed_batch":
        dev_offset = rng.uniform(1.0, 2.2) * sf * rng.choice([-1.0, 1.0])
        dev_tilt = rng.uniform(0.45, 1.2) * sf * rng.choice([-1.0, 1.0])
        dev_bend = rng.uniform(-1.4, 1.4) * sf
        dev_waviness = rng.uniform(0.25, 0.9) * sf
        dev_noise = rng.uniform(0.08, 0.18) * sf
        dev_twist = rng.uniform(-0.8, 0.8) * sf
        fixture_drift = rng.uniform(-0.8, 0.8) * sf

    else:
        raise ValueError(f"Unknown scenario_type: {scenario_type}")

    if disturbance_flag == "disturbed" and scenario_type != "disturbed_batch":
        dev_offset += rng.normal(0.0, 0.35 * sf)
        dev_tilt += rng.normal(0.0, 0.15 * sf)
        dev_noise += rng.uniform(0.03, 0.08) * sf
        fixture_drift += rng.normal(0.0, 0.20 * sf)

    return DeviationFeatures(
        dev_offset=float(dev_offset),
        dev_tilt=float(dev_tilt),
        dev_bend=float(dev_bend),
        dev_waviness=float(dev_waviness),
        dev_noise=float(max(dev_noise, 0.0)),
        dev_twist=float(dev_twist),
        fixture_drift=float(fixture_drift),
    )


def generate_deviated_profile(
    features: DeviationFeatures,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate nominal and deviated profiles.

    dev_twist is represented as an antisymmetric nonlinear profile component.
    fixture_drift is represented as a systematic localized/support-related shift.
    """

    s = np.linspace(0.0, 1.0, N_POINTS)
    z_A = np.zeros_like(s)

    tilt_rad = np.deg2rad(features.dev_tilt)
    waviness_frequency = 3
    phase = rng.uniform(0.0, 2.0 * np.pi)

    noise = rng.normal(
        loc=0.0,
        scale=features.dev_noise,
        size=N_POINTS,
    )

    twist_component = features.dev_twist * (s - 0.5) * np.sin(2.0 * np.pi * s)

    fixture_drift_component = features.fixture_drift * np.exp(
        -((s - 0.35) ** 2) / (2.0 * 0.18**2)
    )

    z_B = (
        features.dev_offset
        + np.tan(tilt_rad) * PROFILE_LENGTH_MM * (s - 0.5)
        + features.dev_bend * (4.0 * (s - 0.5) ** 2 - 1.0)
        + features.dev_waviness * np.sin(2.0 * np.pi * waviness_frequency * s + phase)
        + twist_component
        + fixture_drift_component
        + noise
    )

    return s, z_B - z_A


def apply_assembly_correction(
    s: np.ndarray,
    deviated_profile: np.ndarray,
    z_adj: float,
    theta_adj: float,
    locator_offset: float,
) -> np.ndarray:
    """
    Apply assembly correction parameters.

    Correction parameters remain intentionally limited to:
    - z_adj
    - theta_adj
    - locator_offset

    No fake additional correction parameters are added in this version.
    """

    theta_rad = np.deg2rad(theta_adj)

    locator_center = 0.5
    locator_sigma = 0.12
    locator_effect = locator_offset * np.exp(
        -((s - locator_center) ** 2) / (2.0 * locator_sigma**2)
    )

    corrected_profile = (
        deviated_profile
        + z_adj
        + np.tan(theta_rad) * PROFILE_LENGTH_MM * (s - 0.5)
        + locator_effect
    )

    return corrected_profile


def calculate_quality_metrics(gap: np.ndarray) -> Dict[str, float]:
    """
    Calculate quality metrics from gap distribution.
    """

    mean_gap = float(np.mean(np.abs(gap)))
    max_gap = float(np.max(np.abs(gap)))
    parallelism_error = float(abs(gap[-1] - gap[0]))
    rms_deviation = float(np.sqrt(np.mean(gap**2)))

    quality_score = (
        QUALITY_WEIGHTS["mean_gap"] * mean_gap
        + QUALITY_WEIGHTS["max_gap"] * max_gap
        + QUALITY_WEIGHTS["parallelism_error"] * parallelism_error
        + QUALITY_WEIGHTS["rms_deviation"] * rms_deviation
    )

    tolerance_pass = bool(
        (mean_gap <= 1.0)
        and (max_gap <= 2.5)
        and (parallelism_error <= 2.0)
    )

    return {
        "mean_gap": mean_gap,
        "max_gap": max_gap,
        "parallelism_error": parallelism_error,
        "rms_deviation": rms_deviation,
        "quality_score": float(quality_score),
        "tolerance_pass": tolerance_pass,
    }


def sample_assembly_parameters(rng: np.random.Generator) -> Dict[str, float]:
    """
    Randomly sample assembly parameters within allowed Version 3 ranges.
    """

    return {
        name: float(rng.uniform(low, high))
        for name, (low, high) in ASSEMBLY_PARAMETER_RANGES.items()
    }


def _row_from_case(
    case_id: int,
    batch_id: str,
    part_id: int,
    scenario_type: str,
    severity_level: str,
    disturbance_flag: str,
    features: DeviationFeatures,
    params: Dict[str, float],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    """
    Build one dataset row.
    """

    return {
        "case_id": case_id,
        "batch_id": batch_id,
        "part_id": part_id,
        "scenario_type": scenario_type,
        "severity_level": severity_level,
        "disturbance_flag": disturbance_flag,
        "dev_offset": features.dev_offset,
        "dev_tilt": features.dev_tilt,
        "dev_bend": features.dev_bend,
        "dev_waviness": features.dev_waviness,
        "dev_noise": features.dev_noise,
        "dev_twist": features.dev_twist,
        "fixture_drift": features.fixture_drift,
        "z_adj": params["z_adj"],
        "theta_adj": params["theta_adj"],
        "locator_offset": params["locator_offset"],
        **metrics,
    }


def create_batch_aware_dataset(
    batches_per_scenario: int = 5,
    parts_per_batch: int = 10,
    parameter_samples_per_part: int = 8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a batch-aware dataset for Version 3.

    New expected default size:
    7 normal scenario types x 3 severity levels x 5 batches x 10 parts x 8 samples
    = 8400 normal rows

    Plus disturbed batch rows:
    5 disturbed batches x 10 parts x 8 samples
    = 400 disturbed rows

    Total = 8800 rows.
    """

    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, object]] = []
    case_id = 0

    # Normal scenario batches
    for scenario_type in SCENARIO_TYPES:
        for severity_level in SEVERITY_LEVELS:
            for batch_number in range(1, batches_per_scenario + 1):
                batch_id = f"{scenario_type[:3].upper()}_{severity_level[:1].upper()}_{batch_number:03d}"

                for part_id in range(1, parts_per_batch + 1):
                    features = generate_deviation_features(
                        scenario_type=scenario_type,
                        severity_level=severity_level,
                        disturbance_flag="normal",
                        rng=rng,
                    )

                    s, deviated_profile = generate_deviated_profile(features, rng)

                    for _ in range(parameter_samples_per_part):
                        case_id += 1
                        params = sample_assembly_parameters(rng)

                        corrected_profile = apply_assembly_correction(
                            s=s,
                            deviated_profile=deviated_profile,
                            z_adj=params["z_adj"],
                            theta_adj=params["theta_adj"],
                            locator_offset=params["locator_offset"],
                        )

                        metrics = calculate_quality_metrics(corrected_profile)

                        rows.append(
                            _row_from_case(
                                case_id=case_id,
                                batch_id=batch_id,
                                part_id=part_id,
                                scenario_type=scenario_type,
                                severity_level=severity_level,
                                disturbance_flag="normal",
                                features=features,
                                params=params,
                                metrics=metrics,
                            )
                        )

    # Disturbed / crisis-like batches
    disturbed_batches = 5

    for batch_number in range(1, disturbed_batches + 1):
        batch_id = f"DIST_H_{batch_number:03d}"

        for part_id in range(1, parts_per_batch + 1):
            features = generate_deviation_features(
                scenario_type="disturbed_batch",
                severity_level="high",
                disturbance_flag="disturbed",
                rng=rng,
            )

            s, deviated_profile = generate_deviated_profile(features, rng)

            for _ in range(parameter_samples_per_part):
                case_id += 1
                params = sample_assembly_parameters(rng)

                corrected_profile = apply_assembly_correction(
                    s=s,
                    deviated_profile=deviated_profile,
                    z_adj=params["z_adj"],
                    theta_adj=params["theta_adj"],
                    locator_offset=params["locator_offset"],
                )

                metrics = calculate_quality_metrics(corrected_profile)

                rows.append(
                    _row_from_case(
                        case_id=case_id,
                        batch_id=batch_id,
                        part_id=part_id,
                        scenario_type="disturbed_batch",
                        severity_level="high",
                        disturbance_flag="disturbed",
                        features=features,
                        params=params,
                        metrics=metrics,
                    )
                )

    df = pd.DataFrame(rows)
    return df


def save_batch_aware_dataset(
    output_path: str | Path,
    batches_per_scenario: int = 5,
    parts_per_batch: int = 10,
    parameter_samples_per_part: int = 8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate and save the Version 3 batch-aware dataset.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = create_batch_aware_dataset(
        batches_per_scenario=batches_per_scenario,
        parts_per_batch=parts_per_batch,
        parameter_samples_per_part=parameter_samples_per_part,
        random_seed=random_seed,
    )

    df.to_csv(output_path, index=False)
    return df