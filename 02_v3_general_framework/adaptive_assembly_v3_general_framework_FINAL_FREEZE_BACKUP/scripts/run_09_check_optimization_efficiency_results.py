"""
Clean optimization-efficiency result checker.

This script does not import the broken optimization_efficiency module.
It reads the already generated result tables and prints the final results.

Required files:
- results/tables/optimization_efficiency_summary.csv
- results/tables/optimization_efficiency_checkpoints.csv
- results/tables/optimization_efficiency_curve.csv
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"


def main() -> None:
    summary_path = TABLE_DIR / "optimization_efficiency_summary.csv"
    checkpoints_path = TABLE_DIR / "optimization_efficiency_checkpoints.csv"
    curve_path = TABLE_DIR / "optimization_efficiency_curve.csv"

    missing_files = [
        path for path in [summary_path, checkpoints_path, curve_path] if not path.exists()
    ]

    if missing_files:
        print("Missing optimization-efficiency result files:")
        for path in missing_files:
            print(path)
        print("")
        print("The efficiency comparison must be regenerated later.")
        return

    summary = pd.read_csv(summary_path)
    checkpoints = pd.read_csv(checkpoints_path)
    curve = pd.read_csv(curve_path)

    print("Optimization efficiency summary:")
    print(summary.to_string(index=False))

    print("")
    print("Optimization efficiency checkpoints:")
    print(checkpoints.to_string(index=False))

    print("")
    print("Curve metadata check:")
    print(curve[["method", "evaluation", "scenario_type", "severity_level", "disturbance_flag"]].head(20).to_string(index=False))

    print("")
    print("Files checked successfully:")
    print(summary_path)
    print(checkpoints_path)
    print(curve_path)

    print("")
    print("Interpretation:")
    print("- Lower quality score is better.")
    print("- Bayesian Optimization should be compared against Random Search.")
    print("- Use final_mean_best_quality and checkpoint mean_best_quality for the report.")
    print("- If scenario_type contains NaN in the curve file, the numeric result is still usable, but metadata cleanup is needed later.")


if __name__ == "__main__":
    main()