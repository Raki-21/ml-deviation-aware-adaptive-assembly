"""
Create correction difficulty ranking plot for Version 3.

Input:
- results/tables/correction_capability_summary.csv

Output:
- results/plots/v3/correction_difficulty_ranking.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "correction_capability_summary.csv"
PLOT_DIR = PROJECT_ROOT / "results" / "plots" / "v3"
PLOT_PATH = PLOT_DIR / "correction_difficulty_ranking.png"


def main() -> None:
    print("Generating correction difficulty ranking plot...")
    print(f"Input file: {TABLE_PATH}")
    print(f"Output file: {PLOT_PATH}")

    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {TABLE_PATH}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TABLE_PATH)

    df = df.sort_values("difficult_case_percent", ascending=True)

    scenario_labels = df["scenario_type"].str.replace("_", "\n")

    plt.figure(figsize=(9, 5.5))
    plt.barh(scenario_labels, df["difficult_case_percent"])

    plt.xlabel("Difficult cases after individual adaptive recommendation (%)")
    plt.ylabel("Deviation scenario")
    plt.title(
        "Correction Difficulty Ranking by Deviation Scenario\n"
        "Capability-Aware Analysis"
    )
    plt.xlim(0, max(df["difficult_case_percent"]) + 3)
    plt.grid(axis="x", alpha=0.3)

    for index, value in enumerate(df["difficult_case_percent"]):
        plt.text(
            value + 0.15,
            index,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("")
    print("Saved plot:")
    print(PLOT_PATH)
    print("")
    print("Done.")


if __name__ == "__main__":
    main()