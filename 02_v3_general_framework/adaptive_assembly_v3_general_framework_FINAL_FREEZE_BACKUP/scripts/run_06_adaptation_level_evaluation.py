"""
Run script for Version 3 adaptation-level evaluation.

This script compares:
1. Nominal universal baseline
2. Global-best fixed setting
3. Batch-adaptive setting
4. Individual adaptive setting

Input:
- results/tables/batch_aware_dataset.csv

Outputs:
- results/tables/adaptation_level_comparison.csv
- results/tables/adaptation_selected_cases.csv
- results/tables/adaptation_scenario_summary.csv
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from src.adaptation_evaluator import save_adaptation_evaluation


def main() -> None:
    dataset_path = PROJECT_ROOT / "results" / "tables" / "batch_aware_dataset.csv"
    output_dir = PROJECT_ROOT / "results" / "tables"

    print("Running Version 3 adaptation-level evaluation...")
    print(f"Input dataset: {dataset_path}")

    comparison_summary, selected_cases, scenario_summary = save_adaptation_evaluation(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    print("")
    print("Adaptation-level comparison:")
    print(comparison_summary)

    print("")
    print("Selected cases shape:")
    print(selected_cases.shape)

    print("")
    print("Scenario summary preview:")
    print(scenario_summary.head(20))

    print("")
    print("Saved files:")
    print(output_dir / "adaptation_level_comparison.csv")
    print(output_dir / "adaptation_selected_cases.csv")
    print(output_dir / "adaptation_scenario_summary.csv")


if __name__ == "__main__":
    main()