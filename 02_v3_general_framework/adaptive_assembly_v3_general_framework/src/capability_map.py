"""
Capability map analysis for Version 3.

This module converts recommendation-status and failure-analysis results into
a correction capability map.

The purpose is to move the thesis from simple parameter optimization toward
a capability-aware adaptive assembly decision-support framework.

Main question:
For each scenario and severity level, can the current correction parameters
handle the observed part variation, and what future correction capability
would be needed when they cannot?
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


STATUS_PASS = "PASS_RECOMMENDED"
STATUS_REVIEW = "MANUAL_REVIEW"
STATUS_REWORK = "REWORK_REQUIRED"
STATUS_OUT_OF_RANGE = "OUT_OF_CORRECTION_RANGE"


def read_csv_checked(path: Path) -> pd.DataFrame:
    """
    Read a CSV file and provide a clear error message if it is missing.
    """

    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    return pd.read_csv(path)


def get_status_column(df: pd.DataFrame) -> str:
    """
    Find the recommendation-status column.

    Different earlier scripts may have saved the column with slightly different names.
    """

    possible_columns = [
        "recommendation_status",
        "status",
        "decision_status",
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    raise KeyError(
        "Could not find recommendation status column. Expected one of: "
        f"{possible_columns}. Available columns: {list(df.columns)}"
    )


def get_strategy_column(df: pd.DataFrame) -> str:
    """
    Find the strategy column.
    """

    possible_columns = [
        "strategy",
        "adaptation_strategy",
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    raise KeyError(
        "Could not find strategy column. Expected one of: "
        f"{possible_columns}. Available columns: {list(df.columns)}"
    )


def get_failure_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Find available Boolean failure metric columns.

    This is used when failure flags already exist in the input table.
    """

    candidates = {
        "mean_gap": [
            "fails_mean_gap",
            "mean_gap_failed",
            "fail_mean_gap",
        ],
        "max_gap": [
            "fails_max_gap",
            "max_gap_failed",
            "fail_max_gap",
        ],
        "parallelism_error": [
            "fails_parallelism_error",
            "parallelism_error_failed",
            "fail_parallelism_error",
        ],
    }

    found = {}

    for metric_name, possible_columns in candidates.items():
        for column in possible_columns:
            if column in df.columns:
                found[metric_name] = column
                break

    return found


def infer_dominant_failure_metric(group: pd.DataFrame) -> str:
    """
    Determine the dominant failed quality metric inside a difficult group.

    First, this function checks for explicit Boolean failure columns.
    If they are not available, it infers the failed metric from the
    quality metric values.

    Returns:
    - mean_gap
    - max_gap
    - parallelism_error
    - no_failure_detected
    - failure_metric_not_inferable
    """

    metric_columns = get_failure_metric_columns(group)

    if metric_columns:
        failure_counts = {}

        for metric_name, column in metric_columns.items():
            failure_counts[metric_name] = int(
                group[column].fillna(False).astype(bool).sum()
            )

        max_count = max(failure_counts.values())

        if max_count == 0:
            return "no_failure_detected"

        dominant_metrics = [
            metric_name
            for metric_name, count in failure_counts.items()
            if count == max_count
        ]

        return " + ".join(dominant_metrics)

    # Fallback: infer from actual quality metric values.
    # These limits must match the tolerance limits used in the quality/status logic.
    tolerance_limits = {
        "mean_gap": 1.0,
        "max_gap": 2.0,
        "parallelism_error": 1.0,
    }

    available_metric_columns = [
        metric for metric in tolerance_limits.keys() if metric in group.columns
    ]

    if not available_metric_columns:
        return "failure_metric_not_inferable"

    failure_counts = {}

    for metric in available_metric_columns:
        limit = tolerance_limits[metric]
        failure_counts[metric] = int((group[metric].abs() > limit).sum())

    max_count = max(failure_counts.values())

    if max_count == 0:
        return "no_failure_detected"

    dominant_metrics = [
        metric_name
        for metric_name, count in failure_counts.items()
        if count == max_count
    ]

    return " + ".join(dominant_metrics)


def suggest_future_capability(
    scenario_type: str,
    dominant_failure_metric: str,
    out_of_range_percent: float,
    rework_percent: float,
) -> str:
    """
    Suggest a physically meaningful future correction capability.

    This function is intentionally conservative:
    - If the group has no difficult cases, no extra capability is suggested.
    - If failures exist, the suggestion is based mainly on the limiting metric.
    """

    scenario_type_lower = str(scenario_type).lower()
    metric_lower = str(dominant_failure_metric).lower()

    if rework_percent == 0.0 and out_of_range_percent == 0.0:
        return "current correction capability sufficient for this group"

    if "no_failure_detected" in metric_lower:
        return "current correction capability sufficient for this group"

    if out_of_range_percent >= 20.0:
        return "increase correction range or add additional actuator/correction degree of freedom"

    if "parallelism" in metric_lower:
        return "add stronger angular or multi-point locator correction capability"

    if "max_gap" in metric_lower:
        return "add local shape or multi-point gap correction capability"

    if "mean_gap" in metric_lower:
        return "improve global vertical offset compensation"

    if "twist" in scenario_type_lower:
        return "add torsional or multi-point angular correction capability"

    if "fixture" in scenario_type_lower or "drift" in scenario_type_lower:
        return "add fixture calibration and drift-aware compensation"

    if "waviness" in scenario_type_lower or "bend" in scenario_type_lower:
        return "add local profile correction or flexible locator support"

    if rework_percent > 0.0:
        return "review physical correction limits and add process-specific rework strategy"

    return "current correction capability sufficient for this group"


def create_capability_map(
    recommendation_cases: pd.DataFrame,
    selected_strategy: str = "individual_adaptive_setting",
) -> pd.DataFrame:
    """
    Create a correction capability map for a selected strategy.

    The recommended default is individual_adaptive_setting because that is the
    main thesis contribution.
    """

    df = recommendation_cases.copy()

    status_column = get_status_column(df)
    strategy_column = get_strategy_column(df)

    if "scenario_type" not in df.columns:
        raise KeyError("Column 'scenario_type' is required for capability map analysis.")

    if "severity_level" not in df.columns:
        raise KeyError("Column 'severity_level' is required for capability map analysis.")

    strategy_df = df[df[strategy_column] == selected_strategy].copy()

    if strategy_df.empty:
        available_strategies = sorted(df[strategy_column].dropna().unique().tolist())
        raise ValueError(
            f"No rows found for strategy '{selected_strategy}'. "
            f"Available strategies: {available_strategies}"
        )

    rows = []

    group_columns = ["scenario_type", "severity_level"]

    for (scenario_type, severity_level), group in strategy_df.groupby(group_columns):
        total_cases = len(group)

        pass_count = int((group[status_column] == STATUS_PASS).sum())
        review_count = int((group[status_column] == STATUS_REVIEW).sum())
        rework_count = int((group[status_column] == STATUS_REWORK).sum())
        out_of_range_count = int((group[status_column] == STATUS_OUT_OF_RANGE).sum())

        pass_percent = 100.0 * pass_count / total_cases if total_cases else np.nan
        review_percent = 100.0 * review_count / total_cases if total_cases else np.nan
        rework_percent = 100.0 * rework_count / total_cases if total_cases else np.nan
        out_of_range_percent = (
            100.0 * out_of_range_count / total_cases if total_cases else np.nan
        )

        difficult_group = group[group[status_column] != STATUS_PASS].copy()

        if difficult_group.empty:
            dominant_failure_metric = "no_failure_detected"
        else:
            dominant_failure_metric = infer_dominant_failure_metric(difficult_group)

        suggested_capability = suggest_future_capability(
            scenario_type=scenario_type,
            dominant_failure_metric=dominant_failure_metric,
            out_of_range_percent=out_of_range_percent,
            rework_percent=rework_percent,
        )

        rows.append(
            {
                "strategy": selected_strategy,
                "scenario_type": scenario_type,
                "severity_level": severity_level,
                "total_cases": total_cases,
                "pass_recommended_count": pass_count,
                "pass_recommended_percent": pass_percent,
                "manual_review_count": review_count,
                "manual_review_percent": review_percent,
                "rework_required_count": rework_count,
                "rework_required_percent": rework_percent,
                "out_of_correction_range_count": out_of_range_count,
                "out_of_correction_range_percent": out_of_range_percent,
                "dominant_failure_metric": dominant_failure_metric,
                "suggested_future_capability": suggested_capability,
            }
        )

    capability_map = pd.DataFrame(rows)

    capability_map = capability_map.sort_values(
        by=["scenario_type", "severity_level"]
    ).reset_index(drop=True)

    return capability_map


def create_capability_summary(capability_map: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact summary from the correction capability map.

    The summary is intentionally conservative:
    if a scenario contains any difficult cases, the suggested future capability
    is selected from the difficult groups only, not from fully passing groups.
    """

    rows = []

    for scenario_type, group in capability_map.groupby("scenario_type"):
        total_cases = int(group["total_cases"].sum())

        pass_count = int(group["pass_recommended_count"].sum())
        review_count = int(group["manual_review_count"].sum())
        rework_count = int(group["rework_required_count"].sum())
        out_of_range_count = int(group["out_of_correction_range_count"].sum())

        pass_percent = 100.0 * pass_count / total_cases if total_cases else np.nan
        review_percent = 100.0 * review_count / total_cases if total_cases else np.nan
        rework_percent = 100.0 * rework_count / total_cases if total_cases else np.nan
        out_of_range_percent = (
            100.0 * out_of_range_count / total_cases if total_cases else np.nan
        )

        difficult_case_percent = 100.0 - pass_percent

        difficult_groups = group[
            (group["rework_required_count"] > 0)
            | (group["manual_review_count"] > 0)
            | (group["out_of_correction_range_count"] > 0)
        ].copy()

        if difficult_groups.empty:
            dominant_failure_metric = "no_failure_detected"
            suggested_future_capability = (
                "current correction capability sufficient for this scenario"
            )
        else:
            dominant_failure_metrics = (
                difficult_groups["dominant_failure_metric"]
                .replace("no_failure_detected", np.nan)
                .dropna()
                .value_counts()
            )

            if dominant_failure_metrics.empty:
                dominant_failure_metric = "failure_metric_not_inferable"
            else:
                dominant_failure_metric = dominant_failure_metrics.index[0]

            suggestions = (
                difficult_groups["suggested_future_capability"]
                .replace(
                    "current correction capability sufficient for this group",
                    np.nan,
                )
                .dropna()
                .value_counts()
            )

            if suggestions.empty:
                suggested_future_capability = (
                    "review physical correction limits and add process-specific rework strategy"
                )
            else:
                suggested_future_capability = suggestions.index[0]

        rows.append(
            {
                "scenario_type": scenario_type,
                "total_cases": total_cases,
                "pass_recommended_count": pass_count,
                "pass_recommended_percent": pass_percent,
                "manual_review_count": review_count,
                "manual_review_percent": review_percent,
                "rework_required_count": rework_count,
                "rework_required_percent": rework_percent,
                "out_of_correction_range_count": out_of_range_count,
                "out_of_correction_range_percent": out_of_range_percent,
                "difficult_case_percent": difficult_case_percent,
                "dominant_failure_metric": dominant_failure_metric,
                "suggested_future_capability": suggested_future_capability,
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=["difficult_case_percent", "scenario_type"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return summary


def save_capability_map_outputs(
    input_path: str | Path,
    output_dir: str | Path,
    selected_strategy: str = "individual_adaptive_setting",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load recommendation-status cases, create capability map and summary,
    and save them as CSV files.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendation_cases = read_csv_checked(input_path)

    capability_map = create_capability_map(
        recommendation_cases=recommendation_cases,
        selected_strategy=selected_strategy,
    )

    capability_summary = create_capability_summary(capability_map)

    capability_map_path = output_dir / "correction_capability_map.csv"
    capability_summary_path = output_dir / "correction_capability_summary.csv"

    capability_map.to_csv(capability_map_path, index=False)
    capability_summary.to_csv(capability_summary_path, index=False)

    return capability_map, capability_summary