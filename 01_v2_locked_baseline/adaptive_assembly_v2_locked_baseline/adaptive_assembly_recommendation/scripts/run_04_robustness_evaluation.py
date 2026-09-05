import os
import sys

# Add project root folder to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DEVIATION_RANGES
from src.surrogate_models import load_model
from src.optimization import compare_recommendation_methods


def sample_deviation_case(rng):
    """
    Create one unseen deviation case by sampling from the locked deviation ranges.
    """
    return {
        "dev_offset": float(rng.uniform(*DEVIATION_RANGES["dev_offset"])),
        "dev_tilt": float(rng.uniform(*DEVIATION_RANGES["dev_tilt"])),
        "dev_bend": float(rng.uniform(*DEVIATION_RANGES["dev_bend"])),
        "dev_waviness": float(rng.uniform(*DEVIATION_RANGES["dev_waviness"])),
        "dev_noise": float(rng.uniform(*DEVIATION_RANGES["dev_noise"])),
    }


def plot_improvement_distribution(df, output_path):
    """
    Plot the distribution of improvement percentages for Bayesian optimization.
    """
    bayes_df = df[df["method"] == "bayesian_optimization"]

    plt.figure(figsize=(8, 5))
    plt.hist(bayes_df["improvement_percent_vs_fixed"], bins=15, alpha=0.8)
    plt.xlabel("Improvement compared with fixed setting (%)")
    plt.ylabel("Number of deviation cases")
    plt.title("Robustness Evaluation: Improvement Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    model = load_model("results/models/random_forest_surrogate.joblib")

    rng = np.random.default_rng(123)

    n_cases = 30
    all_rows = []

    for case_id in range(n_cases):
        deviation_case = sample_deviation_case(rng)

        comparison_df, _ = compare_recommendation_methods(
            model=model,
            deviation_case=deviation_case,
        )

        comparison_df["test_case_id"] = case_id

        for key, value in deviation_case.items():
            comparison_df[key] = value

        all_rows.append(comparison_df)

        print(f"Completed robustness case {case_id + 1}/{n_cases}")

    results_df = pd.concat(all_rows, ignore_index=True)

    results_path = "results/tables/robustness_evaluation.csv"
    results_df.to_csv(results_path, index=False)

    summary_rows = []

    for method in results_df["method"].unique():
        method_df = results_df[results_df["method"] == method]

        summary_rows.append({
            "method": method,
            "mean_quality_score": method_df["predicted_quality_score"].mean(),
            "std_quality_score": method_df["predicted_quality_score"].std(),
            "mean_improvement_percent_vs_fixed": method_df["improvement_percent_vs_fixed"].mean(),
            "min_improvement_percent_vs_fixed": method_df["improvement_percent_vs_fixed"].min(),
            "max_improvement_percent_vs_fixed": method_df["improvement_percent_vs_fixed"].max(),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = "results/tables/robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_improvement_distribution(
        df=results_df,
        output_path="results/plots/robustness_improvement_distribution.png",
    )

    print()
    print("Robustness evaluation completed.")
    print()
    print("Summary:")
    print(summary_df)
    print()
    print(f"Saved full robustness results to: {results_path}")
    print(f"Saved robustness summary to: {summary_path}")
    print("Saved plot to: results/plots/robustness_improvement_distribution.png")


if __name__ == "__main__":
    main()