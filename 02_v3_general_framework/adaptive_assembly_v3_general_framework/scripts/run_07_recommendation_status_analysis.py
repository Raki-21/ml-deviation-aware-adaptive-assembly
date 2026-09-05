"""
Run script for Version 3 recommendation-status analysis.

This script reads:
- results/tables/adaptation_selected_cases.csv

It creates:
- results/tables/recommendation_status_cases.csv
- results/tables/recommendation_status_summary.csv
- results/tables/recommendation_status_by_scenario.csv

Purpose:
Convert pass/fail adaptation results into industrial decision-support statuses.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from src.recommendation_status import save_recommendation_status_analysis


def main() -> None:
    selected_cases_path = (
        PROJECT_ROOT / "results" / "tables" / "adaptation_selected_cases.csv"
    )
    output_dir = PROJECT_ROOT / "results" / "tables"

    print("Running Version 3 recommendation-status analysis...")
    print(f"Input selected cases: {selected_cases_path}")

    status_cases, status_summary, scenario_status_summary = (
        save_recommendation_status_analysis(
            selected_cases_path=selected_cases_path,
            output_dir=output_dir,
        )
    )

    print("")
    print("Recommendation status summary:")
    print(status_summary.to_string(index=False))

    print("")
    print("Scenario status summary preview:")
    print(scenario_status_summary.head(30).to_string(index=False))

    print("")
    print("Saved files:")
    print(output_dir / "recommendation_status_cases.csv")
    print(output_dir / "recommendation_status_summary.csv")
    print(output_dir / "recommendation_status_by_scenario.csv")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()