"""
Run correction capability map analysis.

This script creates capability-aware result tables from the recommendation-status
case file.

Input:
- results/tables/recommendation_status_cases.csv

Outputs:
- results/tables/correction_capability_map.csv
- results/tables/correction_capability_summary.csv
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from src.capability_map import save_capability_map_outputs


def main() -> None:
    input_path = PROJECT_ROOT / "results" / "tables" / "recommendation_status_cases.csv"
    output_dir = PROJECT_ROOT / "results" / "tables"

    print("Running correction capability map analysis...")
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print("Selected strategy: individual_adaptive_setting")

    capability_map, capability_summary = save_capability_map_outputs(
        input_path=input_path,
        output_dir=output_dir,
        selected_strategy="individual_adaptive_setting",
    )

    print("")
    print("Correction capability map:")
    print(capability_map.to_string(index=False))

    print("")
    print("Correction capability summary:")
    print(capability_summary.to_string(index=False))

    print("")
    print("Saved files:")
    print(output_dir / "correction_capability_map.csv")
    print(output_dir / "correction_capability_summary.csv")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()