"""
Version 3 result plots for thesis/report.

This module creates report-ready plots from the generated Version 3 result tables.

Input tables:
- adaptation_level_comparison.csv
- recommendation_status_summary.csv
- failure_analysis_by_metric.csv
- optimization_efficiency_checkpoints.csv
- model_screening_summary.csv

Output folder:
- results/plots/v3
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STRATEGY_LABELS = {
    "nominal_universal_baseline": "Nominal\nbaseline",
    "global_best_fixed_setting": "Global-best\nfixed",
    "batch_adaptive_setting": "Batch\nadaptive",
    "individual_adaptive_setting": "Individual\nadaptive",
}

STATUS_LABELS = {
    "PASS_RECOMMENDED": "Pass\nrecommended",
    "MANUAL_REVIEW": "Manual\nreview",
    "REWORK_REQUIRED": "Rework\nrequired",
    "OUT_OF_CORRECTION_RANGE": "Out of\ncorrection range",
}

METHOD_LABELS = {
    "bayesian_optimization": "Bayesian\nOptimization",
    "random_search": "Random\nSearch",
}


def read_csv_checked(path: Path) -> pd.DataFrame:
    """
    Read a CSV file and provide a clear error if it does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")

    return pd.read_csv(path)


def save_current_figure(output_path: Path) -> None:
    """
    Save current matplotlib figure with consistent settings.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_adaptation_quality_comparison(
    table_dir: Path,
    plot_dir: Path,
) -> None:
    """
    Plot mean quality score by adaptation strategy.

    Lower quality score is better.
    """

    df = read_csv_checked(table_dir / "adaptation_level_comparison.csv")
    df = df.copy()
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])

    plt.figure(figsize=(8, 5))
    plt.bar(df["strategy_label"], df["mean_quality_score"])
    plt.ylabel("Mean quality score")
    plt.xlabel("Adaptation strategy")
    plt.title("Mean Quality Score by Adaptation Strategy\n(lower is better)")
    plt.grid(axis="y", alpha=0.3)

    for index, value in enumerate(df["mean_quality_score"]):
        plt.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    save_current_figure(plot_dir / "adaptation_strategy_quality_comparison.png")


def plot_adaptation_pass_rate_comparison(
    table_dir: Path,
    plot_dir: Path,
) -> None:
    """
    Plot pass rate by adaptation strategy.

    Higher pass rate is better.
    """

    df = read_csv_checked(table_dir / "adaptation_level_comparison.csv")
    df = df.copy()
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])

    plt.figure(figsize=(8, 5))
    plt.bar(df["strategy_label"], df["pass_rate_percent"])
    plt.ylabel("Pass rate (%)")
    plt.xlabel("Adaptation strategy")
    plt.title("Pass Rate by Adaptation Strategy\n(higher is better)")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)

    for index, value in enumerate(df["pass_rate_percent"]):
        plt.text(index, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

    save_current_figure(plot_dir / "adaptation_strategy_pass_rate_comparison.png")


def plot_recommendation_status_distribution(
    table_dir: Path,
    plot_dir: Path,
) -> None:
    """
    Plot recommendation status distribution for each strategy.

    Uses a grouped bar chart with status share percentages.
    """

    df = read_csv_checked(table_dir / "recommendation_status_summary.csv")
    df = df.copy()

    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])
    df["status_label"] = (
        df["recommendation_status"].map(STATUS_LABELS).fillna(df["recommendation_status"])
    )

    pivot = df.pivot_table(
        index="strategy_label",
        columns="status_label",
        values="status_share_percent",
        fill_value=0.0,
    )

    desired_status_order = [
        "Pass\nrecommended",
        "Manual\nreview",
        "Rework\nrequired",
        "Out of\ncorrection range",
    ]

    existing_status_order = [
        status for status in desired_status_order if status in pivot.columns
    ]

    pivot = pivot[existing_status_order]

    plt.figure(figsize=(10, 6))
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Status share (%)")
    ax.set_xlabel("Adaptation strategy")
    ax.set_title("Recommendation Status Distribution by Strategy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Recommendation status", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)

    save_current_figure(plot_dir / "recommendation_status_distribution.png")


def plot_failure_metric_comparison(
    table_dir: Path,
    plot_dir: Path,
) -> None:
    """
    Plot failure causes by metric.

    This shows whether difficult cases fail due to mean gap, max gap,
    or parallelism error.
    """

    df = read_csv_checked(table_dir / "failure_analysis_by_metric.csv")
    df = df.copy()

    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])

    plot_columns = [
        "fails_mean_gap_percent",
        "fails_max_gap_percent",
        "fails_parallelism_error_percent",
    ]

    label_map = {
        "fails_mean_gap_percent": "Mean gap",
        "fails_max_gap_percent": "Max gap",
        "fails_parallelism_error_percent": "Parallelism error",
    }

    plot_df = df.set_index("strategy_label")[plot_columns]
    plot_df = plot_df.rename(columns=label_map)

    plt.figure(figsize=(10, 6))
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Share of difficult cases failing metric (%)")
    ax.set_xlabel("Adaptation strategy")
    ax.set_title("Failure Causes by Quality Metric")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Failed quality metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)

    save_current_figure(plot_dir / "failure_metric_comparison.png")


def plot_optimization_efficiency_curve(
    table_dir: Path,
    plot_dir: Path,
) -> None:
    """
    Plot optimization efficiency curve.

    Lower best quality is better.
    """

    df = read_csv_checked(table_dir / "optimization_efficiency_checkpoints.csv")
    df = df.copy()

    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])

    plt.figure(figsize=(8, 5))

    for method_label, group in df.groupby("method_label"):
        group = group.sort_values("evaluation_budget")
        plt.plot(
            group["evaluation_budget"],
            group["mean_best_quality"],
            marker="o",
            label=method_label,
        )

    plt.ylabel("Mean best quality score")
    plt.xlabel("Evaluation budget")
    plt.title("Optimization Efficiency Comparison\n(lower is better)")
    plt.grid(alpha=0.3)
    plt.legend(title="Method")

    save_current_figure(plot_dir / "optimization_efficiency_curve.png")


def plot_model_screening_rmse(
    table_dir: Path,
    plot_dir: Path,
) -> None:
    """
    Plot model comparison by RMSE.

    Lower RMSE is better.
    """

    df = read_csv_checked(table_dir / "model_screening_summary.csv")
    df = df.copy()
    df = df.sort_values("mean_RMSE", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["mean_RMSE"])
    plt.ylabel("Mean RMSE")
    plt.xlabel("Model")
    plt.title("Surrogate Model Screening by RMSE\n(lower is better)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for index, value in enumerate(df["mean_RMSE"]):
        plt.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    save_current_figure(plot_dir / "model_screening_rmse_comparison.png")


def plot_model_screening_r2(
    table_dir: Path,
    plot_dir: Path,
) -> None:
    """
    Plot model comparison by R2.

    Higher R2 is better.
    """

    df = read_csv_checked(table_dir / "model_screening_summary.csv")
    df = df.copy()
    df = df.sort_values("mean_R2", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["mean_R2"])
    plt.ylabel("Mean R²")
    plt.xlabel("Model")
    plt.title("Surrogate Model Screening by R²\n(higher is better)")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for index, value in enumerate(df["mean_R2"]):
        plt.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    save_current_figure(plot_dir / "model_screening_r2_comparison.png")


def generate_all_v3_result_plots(
    table_dir: str | Path,
    plot_dir: str | Path,
) -> None:
    """
    Generate all Version 3 result plots.
    """

    table_dir = Path(table_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_adaptation_quality_comparison(table_dir, plot_dir)
    plot_adaptation_pass_rate_comparison(table_dir, plot_dir)
    plot_recommendation_status_distribution(table_dir, plot_dir)
    plot_failure_metric_comparison(table_dir, plot_dir)
    plot_optimization_efficiency_curve(table_dir, plot_dir)
    plot_model_screening_rmse(table_dir, plot_dir)
    plot_model_screening_r2(table_dir, plot_dir)