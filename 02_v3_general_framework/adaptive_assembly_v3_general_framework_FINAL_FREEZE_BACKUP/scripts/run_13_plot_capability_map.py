"""
Create correction capability map plot for Version 3.

Input:
- results/tables/correction_capability_summary.csv

Output:
- results/plots/v3/correction_capability_map.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "correction_capability_summary.csv"
PLOT_DIR = PROJECT_ROOT / "results" / "plots" / "v3"
PLOT_PATH = PLOT_DIR / "correction_capability_map.png"


def main() -> None:
    print("Generating correction capability map plot...")
    print(f"Input file: {TABLE_PATH}")
    print(f"Output file: {PLOT_PATH}")

    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {TABLE_PATH}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TABLE_PATH)

    # Show hardest scenarios at the bottom/top clearly
    df = df.sort_values("pass_recommended_percent", ascending=True)

    scenario_labels = df["scenario_type"].str.replace("_", "\n")

    plot_df = pd.DataFrame(
        {
            "Pass recommended": df["pass_recommended_percent"],
            "Rework required": df["rework_required_percent"],
            "Manual review": df["manual_review_percent"],
            "Out of correction range": df["out_of_correction_range_percent"],
        },
        index=scenario_labels,
    )

    ax = plot_df.plot(
        kind="barh",
        stacked=True,
        figsize=(11, 6.5),
        width=0.75,
    )

    ax.set_xlabel("Share of cases (%)")
    ax.set_ylabel("Deviation scenario")
    ax.set_title(
        "Correction Capability Map by Deviation Scenario\n"
        "Individual Adaptive Recommendation"
    )

    # Extra space on right for difficult labels
    ax.set_xlim(0, 108)
    ax.grid(axis="x", alpha=0.3)

    # Move legend below plot to avoid overlap
    ax.legend(
        title="Recommendation status",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=True,
    )

    # Add readable labels
    for index, row in df.reset_index(drop=True).iterrows():
        pass_value = row["pass_recommended_percent"]
        difficult_value = row["difficult_case_percent"]

        ax.text(
            pass_value / 2,
            index,
            f"{pass_value:.1f}% pass",
            va="center",
            ha="center",
            fontsize=8,
        )

        if difficult_value > 0:
            ax.text(
                101.0,
                index,
                f"{difficult_value:.1f}% difficult",
                va="center",
                ha="left",
                fontsize=8,
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