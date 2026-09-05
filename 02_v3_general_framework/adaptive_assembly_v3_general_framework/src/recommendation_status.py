"""
Recommendation-status and feasibility classification for Version 3.

This module converts optimized/adaptive assembly results into an
industrial decision-support output.

Instead of only reporting pass/fail, the system classifies each case as:

1. PASS_RECOMMENDED
   The recommended parameters satisfy the defined quality limits.

2. MANUAL_REVIEW
   The part does not fully pass, but the result is close to the acceptable
   boundary and should be checked by an engineer or quality specialist.

3. REWORK_REQUIRED
   The part is not acceptable after correction, but may still be recoverable
   through additional process action, rework, or alternative setup.

4. OUT_OF_CORRECTION_RANGE
   The available correction parameters are insufficient for the deviation case.
   The part/condition should not be released using the current recommendation.

This makes the framework more realistic for industrial quality decision support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


QUALITY_COLUMN = "quality_score"


def load_selected_cases(selected_cases_path: str | Path) -> pd.DataFrame:
    """
    Load adaptation-selected cases.

    This file should be created by:
    scripts/run_06_adaptation_level_evaluation.py
    """

    selected_cases_path = Path(selected_cases_path)

    if not selected_cases_path.exists():
        raise FileNotFoundError(
            f"Selected cases file not found: {selected_cases_path}. "
            "Run scripts/run_06_adaptation_level_evaluation.py first."
        )

    return pd.read_csv(selected_cases_path)


def classify_recommendation_status(row: pd.Series) -> str:
    """
    Classify one evaluated case into an industrial recommendation status.

    The thresholds are prototype-level decision thresholds.

    Logic:
    - If the part satisfies all tolerance criteria, release the recommendation.
    - If it fails but quality score is still close to acceptable, manual review.
    - If it fails with moderate remaining error, rework is required.
    - If the remaining error is high, the current correction system is insufficient.
    """

    tolerance_pass = bool(row["tolerance_pass"])
    quality_score = float(row[QUALITY_COLUMN])
    max_gap = float(row["max_gap"])
    parallelism_error = float(row["parallelism_error"])

    if tolerance_pass:
        return "PASS_RECOMMENDED"

    # Close-to-pass region:
    # not fully acceptable, but close enough that an engineer/quality check
    # may decide whether additional local inspection is acceptable.
    if quality_score <= 1.00 and max_gap <= 3.00 and parallelism_error <= 2.50:
        return "MANUAL_REVIEW"

    # Recoverable region:
    # the correction is not sufficient for direct release, but the error is not
    # extremely large. This can be sent to rework or alternative process action.
    if quality_score <= 2.50 and max_gap <= 5.00 and parallelism_error <= 5.00:
        return "REWORK_REQUIRED"

    # Severe remaining error:
    # current correction variables are not enough.
    return "OUT_OF_CORRECTION_RANGE"


def add_recommendation_status(selected_cases: pd.DataFrame) -> pd.DataFrame:
    """
    Add recommendation_status and release_decision columns.
    """

    df = selected_cases.copy()

    df["recommendation_status"] = df.apply(classify_recommendation_status, axis=1)

    df["release_decision"] = df["recommendation_status"].map(
        {
            "PASS_RECOMMENDED": "release",
            "MANUAL_REVIEW": "hold_for_review",
            "REWORK_REQUIRED": "hold_for_rework",
            "OUT_OF_CORRECTION_RANGE": "do_not_release",
        }
    )

    return df


def create_recommendation_status_summary(status_cases: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize recommendation status by strategy.
    """

    summary = (
        status_cases.groupby(["strategy", "recommendation_status"], as_index=False)
        .agg(
            cases=("part_case_id", "count"),
            mean_quality_score=(QUALITY_COLUMN, "mean"),
            median_quality_score=(QUALITY_COLUMN, "median"),
            worst_case_quality_score=(QUALITY_COLUMN, "max"),
            mean_max_gap=("max_gap", "mean"),
            mean_parallelism_error=("parallelism_error", "mean"),
        )
    )

    total_by_strategy = (
        status_cases.groupby("strategy")["part_case_id"]
        .count()
        .rename("total_cases")
        .reset_index()
    )

    summary = summary.merge(total_by_strategy, on="strategy", how="left")
    summary["status_share_percent"] = summary["cases"] / summary["total_cases"] * 100.0

    # Nice ordering for readability
    status_order = {
        "PASS_RECOMMENDED": 1,
        "MANUAL_REVIEW": 2,
        "REWORK_REQUIRED": 3,
        "OUT_OF_CORRECTION_RANGE": 4,
    }

    strategy_order = {
        "nominal_universal_baseline": 1,
        "global_best_fixed_setting": 2,
        "batch_adaptive_setting": 3,
        "individual_adaptive_setting": 4,
    }

    summary["status_order"] = summary["recommendation_status"].map(status_order)
    summary["strategy_order"] = summary["strategy"].map(strategy_order)

    summary = (
        summary.sort_values(["strategy_order", "status_order"])
        .drop(columns=["strategy_order", "status_order"])
        .reset_index(drop=True)
    )

    return summary


def create_recommendation_status_by_scenario(status_cases: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize status distribution by strategy, scenario type, and severity.
    """

    scenario_summary = (
        status_cases.groupby(
            [
                "strategy",
                "scenario_type",
                "severity_level",
                "disturbance_flag",
                "recommendation_status",
            ],
            as_index=False,
        )
        .agg(
            cases=("part_case_id", "count"),
            mean_quality_score=(QUALITY_COLUMN, "mean"),
            worst_case_quality_score=(QUALITY_COLUMN, "max"),
        )
    )

    total_by_group = (
        status_cases.groupby(
            ["strategy", "scenario_type", "severity_level", "disturbance_flag"]
        )["part_case_id"]
        .count()
        .rename("total_group_cases")
        .reset_index()
    )

    scenario_summary = scenario_summary.merge(
        total_by_group,
        on=["strategy", "scenario_type", "severity_level", "disturbance_flag"],
        how="left",
    )

    scenario_summary["status_share_percent"] = (
        scenario_summary["cases"] / scenario_summary["total_group_cases"] * 100.0
    )

    return scenario_summary


def save_recommendation_status_analysis(
    selected_cases_path: str | Path,
    output_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load selected adaptation cases, classify recommendation status, and save tables.
    """

    selected_cases_path = Path(selected_cases_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = load_selected_cases(selected_cases_path)
    status_cases = add_recommendation_status(selected_cases)

    status_summary = create_recommendation_status_summary(status_cases)
    scenario_status_summary = create_recommendation_status_by_scenario(status_cases)

    status_cases_path = output_dir / "recommendation_status_cases.csv"
    status_summary_path = output_dir / "recommendation_status_summary.csv"
    scenario_status_summary_path = output_dir / "recommendation_status_by_scenario.csv"

    status_cases.to_csv(status_cases_path, index=False)
    status_summary.to_csv(status_summary_path, index=False)
    scenario_status_summary.to_csv(scenario_status_summary_path, index=False)

    return status_cases, status_summary, scenario_status_summary