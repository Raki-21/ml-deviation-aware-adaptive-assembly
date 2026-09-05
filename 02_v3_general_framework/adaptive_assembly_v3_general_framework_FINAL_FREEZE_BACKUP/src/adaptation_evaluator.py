"""
Improved adaptation-level evaluator for Version 3.

This version evaluates adaptation strategies using a deterministic parameter
grid and direct quality calculation from deviation features.

Version 3 extension:
- supports dev_twist
- supports fixture_drift

Strategies:
1. Nominal universal baseline
2. Global-best fixed setting
3. Batch-adaptive setting
4. Individual adaptive setting

Lower quality_score is better.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


PROFILE_LENGTH_MM = 1000.0
N_POINTS = 100

QUALITY_WEIGHTS = {
    "mean_gap": 0.30,
    "max_gap": 0.30,
    "parallelism_error": 0.30,
    "rms_deviation": 0.10,
}


def load_batch_dataset(dataset_path: str | Path) -> pd.DataFrame:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run scripts/run_05_generate_batch_dataset.py first."
        )

    return pd.read_csv(dataset_path)


def get_unique_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the sampled dataset to one row per physical part case.

    Each part has many sampled assembly parameter rows, but the deviation
    features are the same for that part.
    """

    part_columns = [
        "batch_id",
        "part_id",
        "scenario_type",
        "severity_level",
        "disturbance_flag",
        "dev_offset",
        "dev_tilt",
        "dev_bend",
        "dev_waviness",
        "dev_noise",
        "dev_twist",
        "fixture_drift",
    ]

    missing_columns = [col for col in part_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns in batch dataset: {missing_columns}. "
            "Regenerate the dataset using run_05_generate_batch_dataset.py."
        )

    unique_parts = (
        df[part_columns]
        .drop_duplicates(subset=["batch_id", "part_id"])
        .reset_index(drop=True)
    )

    unique_parts["part_case_id"] = np.arange(1, len(unique_parts) + 1)

    return unique_parts


def create_parameter_grid() -> pd.DataFrame:
    """
    Create a clean parameter grid for strategy comparison.

    The grid is intentionally moderate in size:
    z_adj: 11 values
    theta_adj: 11 values
    locator_offset: 9 values

    Total candidates = 1089.
    """

    z_values = np.linspace(-2.5, 2.5, 11)
    theta_values = np.linspace(-1.2, 1.2, 11)
    locator_values = np.linspace(-1.0, 1.0, 9)

    rows = []

    for z_adj in z_values:
        for theta_adj in theta_values:
            for locator_offset in locator_values:
                rows.append(
                    {
                        "z_adj": float(z_adj),
                        "theta_adj": float(theta_adj),
                        "locator_offset": float(locator_offset),
                    }
                )

    return pd.DataFrame(rows)


def calculate_quality_for_part(
    dev_offset: float,
    dev_tilt: float,
    dev_bend: float,
    dev_waviness: float,
    dev_noise: float,
    dev_twist: float,
    fixture_drift: float,
    z_adj: float,
    theta_adj: float,
    locator_offset: float,
) -> Dict[str, float]:
    """
    Deterministically calculate quality for one part and one parameter setting.

    This uses the same simplified physical logic as the Version 3 generator,
    but without random phase/noise regeneration. This is intentional for fair
    strategy comparison.

    dev_twist:
        torsion-like nonlinear shape component.

    fixture_drift:
        systematic localized process/locator drift contribution.
    """

    s = np.linspace(0.0, 1.0, N_POINTS)

    dev_tilt_rad = np.deg2rad(dev_tilt)
    theta_rad = np.deg2rad(theta_adj)

    waviness_frequency = 3
    phase = 0.0

    twist_component = dev_twist * (s - 0.5) * np.sin(2.0 * np.pi * s)

    fixture_drift_component = fixture_drift * np.exp(
        -((s - 0.35) ** 2) / (2.0 * 0.18**2)
    )

    base_gap = (
        dev_offset
        + np.tan(dev_tilt_rad) * PROFILE_LENGTH_MM * (s - 0.5)
        + dev_bend * (4.0 * (s - 0.5) ** 2 - 1.0)
        + dev_waviness * np.sin(2.0 * np.pi * waviness_frequency * s + phase)
        + twist_component
        + fixture_drift_component
    )

    locator_center = 0.5
    locator_sigma = 0.12
    locator_effect = locator_offset * np.exp(
        -((s - locator_center) ** 2) / (2.0 * locator_sigma**2)
    )

    corrected_gap = (
        base_gap
        + z_adj
        + np.tan(theta_rad) * PROFILE_LENGTH_MM * (s - 0.5)
        + locator_effect
    )

    mean_gap = float(np.mean(np.abs(corrected_gap)))
    max_gap = float(np.max(np.abs(corrected_gap)))
    parallelism_error = float(abs(corrected_gap[-1] - corrected_gap[0]))
    rms_deviation = float(np.sqrt(np.mean(corrected_gap**2)))

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


def evaluate_parts_with_setting(
    unique_parts: pd.DataFrame,
    z_adj: float,
    theta_adj: float,
    locator_offset: float,
    strategy: str,
) -> pd.DataFrame:
    """
    Evaluate all unique parts using one fixed parameter setting.
    """

    rows = []

    for _, part in unique_parts.iterrows():
        metrics = calculate_quality_for_part(
            dev_offset=part["dev_offset"],
            dev_tilt=part["dev_tilt"],
            dev_bend=part["dev_bend"],
            dev_waviness=part["dev_waviness"],
            dev_noise=part["dev_noise"],
            dev_twist=part["dev_twist"],
            fixture_drift=part["fixture_drift"],
            z_adj=z_adj,
            theta_adj=theta_adj,
            locator_offset=locator_offset,
        )

        rows.append(
            {
                "strategy": strategy,
                "part_case_id": part["part_case_id"],
                "batch_id": part["batch_id"],
                "part_id": part["part_id"],
                "scenario_type": part["scenario_type"],
                "severity_level": part["severity_level"],
                "disturbance_flag": part["disturbance_flag"],
                "dev_offset": part["dev_offset"],
                "dev_tilt": part["dev_tilt"],
                "dev_bend": part["dev_bend"],
                "dev_waviness": part["dev_waviness"],
                "dev_noise": part["dev_noise"],
                "dev_twist": part["dev_twist"],
                "fixture_drift": part["fixture_drift"],
                "z_adj": z_adj,
                "theta_adj": theta_adj,
                "locator_offset": locator_offset,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def find_best_setting_for_parts(
    unique_parts: pd.DataFrame,
    parameter_grid: pd.DataFrame,
) -> Tuple[Dict[str, float], float]:
    """
    Find the parameter-grid setting with the lowest mean quality score
    over the given parts.
    """

    best_setting = None
    best_mean_quality = np.inf

    for _, candidate in parameter_grid.iterrows():
        evaluated = evaluate_parts_with_setting(
            unique_parts=unique_parts,
            z_adj=candidate["z_adj"],
            theta_adj=candidate["theta_adj"],
            locator_offset=candidate["locator_offset"],
            strategy="candidate",
        )

        mean_quality = evaluated["quality_score"].mean()

        if mean_quality < best_mean_quality:
            best_mean_quality = mean_quality
            best_setting = {
                "z_adj": float(candidate["z_adj"]),
                "theta_adj": float(candidate["theta_adj"]),
                "locator_offset": float(candidate["locator_offset"]),
            }

    return best_setting, float(best_mean_quality)


def evaluate_nominal_baseline(unique_parts: pd.DataFrame) -> pd.DataFrame:
    return evaluate_parts_with_setting(
        unique_parts=unique_parts,
        z_adj=0.0,
        theta_adj=0.0,
        locator_offset=0.0,
        strategy="nominal_universal_baseline",
    )


def evaluate_global_best_fixed_setting(
    unique_parts: pd.DataFrame,
    parameter_grid: pd.DataFrame,
) -> pd.DataFrame:
    best_setting, _ = find_best_setting_for_parts(unique_parts, parameter_grid)

    return evaluate_parts_with_setting(
        unique_parts=unique_parts,
        z_adj=best_setting["z_adj"],
        theta_adj=best_setting["theta_adj"],
        locator_offset=best_setting["locator_offset"],
        strategy="global_best_fixed_setting",
    )


def evaluate_batch_adaptive_setting(
    unique_parts: pd.DataFrame,
    parameter_grid: pd.DataFrame,
) -> pd.DataFrame:
    selected_batches = []

    for batch_id, batch_parts in unique_parts.groupby("batch_id"):
        best_setting, _ = find_best_setting_for_parts(batch_parts, parameter_grid)

        evaluated = evaluate_parts_with_setting(
            unique_parts=batch_parts,
            z_adj=best_setting["z_adj"],
            theta_adj=best_setting["theta_adj"],
            locator_offset=best_setting["locator_offset"],
            strategy="batch_adaptive_setting",
        )

        selected_batches.append(evaluated)

    return pd.concat(selected_batches, ignore_index=True)


def evaluate_individual_adaptive_setting(
    unique_parts: pd.DataFrame,
    parameter_grid: pd.DataFrame,
) -> pd.DataFrame:
    selected_parts = []

    for _, part in unique_parts.iterrows():
        one_part = pd.DataFrame([part])
        best_setting, _ = find_best_setting_for_parts(one_part, parameter_grid)

        evaluated = evaluate_parts_with_setting(
            unique_parts=one_part,
            z_adj=best_setting["z_adj"],
            theta_adj=best_setting["theta_adj"],
            locator_offset=best_setting["locator_offset"],
            strategy="individual_adaptive_setting",
        )

        selected_parts.append(evaluated)

    return pd.concat(selected_parts, ignore_index=True)


def _strategy_metrics(strategy_name: str, selected_rows: pd.DataFrame) -> Dict[str, float]:
    return {
        "strategy": strategy_name,
        "mean_quality_score": float(selected_rows["quality_score"].mean()),
        "median_quality_score": float(selected_rows["quality_score"].median()),
        "std_quality_score": float(selected_rows["quality_score"].std()),
        "worst_case_quality_score": float(selected_rows["quality_score"].max()),
        "best_case_quality_score": float(selected_rows["quality_score"].min()),
        "pass_rate_percent": float(selected_rows["tolerance_pass"].mean() * 100.0),
        "evaluated_cases": int(len(selected_rows)),
    }


def compare_adaptation_levels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_parts = get_unique_parts(df)
    parameter_grid = create_parameter_grid()

    nominal = evaluate_nominal_baseline(unique_parts)
    global_best = evaluate_global_best_fixed_setting(unique_parts, parameter_grid)
    batch_adaptive = evaluate_batch_adaptive_setting(unique_parts, parameter_grid)
    individual_adaptive = evaluate_individual_adaptive_setting(unique_parts, parameter_grid)

    selected_cases = pd.concat(
        [nominal, global_best, batch_adaptive, individual_adaptive],
        ignore_index=True,
    )

    summary_rows = [
        _strategy_metrics("nominal_universal_baseline", nominal),
        _strategy_metrics("global_best_fixed_setting", global_best),
        _strategy_metrics("batch_adaptive_setting", batch_adaptive),
        _strategy_metrics("individual_adaptive_setting", individual_adaptive),
    ]

    comparison_summary = pd.DataFrame(summary_rows)

    nominal_mean = comparison_summary.loc[
        comparison_summary["strategy"] == "nominal_universal_baseline",
        "mean_quality_score",
    ].iloc[0]

    nominal_pass_rate = comparison_summary.loc[
        comparison_summary["strategy"] == "nominal_universal_baseline",
        "pass_rate_percent",
    ].iloc[0]

    comparison_summary["improvement_vs_nominal_percent"] = (
        (nominal_mean - comparison_summary["mean_quality_score"])
        / nominal_mean
        * 100.0
    )

    comparison_summary["pass_rate_improvement_vs_nominal_percentage_points"] = (
        comparison_summary["pass_rate_percent"] - nominal_pass_rate
    )

    return comparison_summary, selected_cases


def create_scenario_strategy_summary(selected_cases: pd.DataFrame) -> pd.DataFrame:
    summary = (
        selected_cases.groupby(
            ["strategy", "scenario_type", "severity_level", "disturbance_flag"],
            as_index=False,
        )
        .agg(
            evaluated_cases=("quality_score", "count"),
            mean_quality_score=("quality_score", "mean"),
            median_quality_score=("quality_score", "median"),
            std_quality_score=("quality_score", "std"),
            worst_case_quality_score=("quality_score", "max"),
            pass_rate_percent=("tolerance_pass", lambda x: float(x.mean() * 100.0)),
        )
    )

    return summary


def save_adaptation_evaluation(
    dataset_path: str | Path,
    output_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_batch_dataset(dataset_path)

    comparison_summary, selected_cases = compare_adaptation_levels(df)
    scenario_summary = create_scenario_strategy_summary(selected_cases)

    comparison_path = output_dir / "adaptation_level_comparison.csv"
    selected_cases_path = output_dir / "adaptation_selected_cases.csv"
    scenario_summary_path = output_dir / "adaptation_scenario_summary.csv"

    comparison_summary.to_csv(comparison_path, index=False)
    selected_cases.to_csv(selected_cases_path, index=False)
    scenario_summary.to_csv(scenario_summary_path, index=False)

    return comparison_summary, selected_cases, scenario_summary