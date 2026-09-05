"""
Run script for Version 3 batch-aware dataset generation.

This script creates:
- results/tables/batch_aware_dataset.csv
- results/tables/batch_aware_dataset_summary.csv

It does not modify the Version 2 dataset.
"""

from pathlib import Path
import sys

import pandas as pd


# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from src.batch_deviation_generator import save_batch_aware_dataset


def main() -> None:
    output_dir = PROJECT_ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "batch_aware_dataset.csv"
    summary_path = output_dir / "batch_aware_dataset_summary.csv"

    print("Generating Version 3 batch-aware dataset...")

    df = save_batch_aware_dataset(
        output_path=dataset_path,
        batches_per_scenario=5,
        parts_per_batch=10,
        parameter_samples_per_part=8,
        random_seed=42,
    )

    print(f"Dataset saved to: {dataset_path}")
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    summary = (
        df.groupby(["scenario_type", "severity_level", "disturbance_flag"])
        .agg(
            rows=("case_id", "count"),
            unique_batches=("batch_id", "nunique"),
            unique_parts=("part_id", "nunique"),
            mean_quality_score=("quality_score", "mean"),
            std_quality_score=("quality_score", "std"),
            min_quality_score=("quality_score", "min"),
            max_quality_score=("quality_score", "max"),
            pass_rate=("tolerance_pass", "mean"),
        )
        .reset_index()
    )

    summary["pass_rate"] = summary["pass_rate"] * 100.0

    summary.to_csv(summary_path, index=False)

    print(f"Summary saved to: {summary_path}")
    print("")
    print("Preview of dataset:")
    print(df.head())
    print("")
    print("Scenario summary:")
    print(summary)


if __name__ == "__main__":
    main()