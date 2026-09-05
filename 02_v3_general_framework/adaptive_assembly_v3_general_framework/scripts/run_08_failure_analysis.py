"""
Run script for Version 3 failure analysis.

Input:
- results/tables/recommendation_status_cases.csv

Outputs:
- results/tables/failure_analysis_by_strategy.csv
- results/tables/failure_analysis_by_scenario.csv
- results/tables/failure_analysis_by_metric.csv
- results/tables/individual_adaptive_failed_cases.csv
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from src.failure_analysis import save_failure_analysis


def main() -> None:
    input_path = PROJECT_ROOT / "results" / "tables" / "recommendation_status_cases.csv"
    output_dir = PROJECT_ROOT / "results" / "tables"

    print("Running Version 3 failure analysis...")
    print(f"Input file: {input_path}")

    by_strategy, by_scenario, by_metric, individual_failed = save_failure_analysis(
        recommendation_status_cases_path=input_path,
        output_dir=output_dir,
    )

    print("")
    print("Failure analysis by strategy:")
    print(by_strategy.to_string(index=False))

    print("")
    print("Failure analysis by scenario preview:")
    print(by_scenario.head(30).to_string(index=False))

    print("")
    print("Failure analysis by metric:")
    print(by_metric.to_string(index=False))

    print("")
    print("Individual adaptive failed cases:")
    print(individual_failed.to_string(index=False))

    print("")
    print("Saved files:")
    print(output_dir / "failure_analysis_by_strategy.csv")
    print(output_dir / "failure_analysis_by_scenario.csv")
    print(output_dir / "failure_analysis_by_metric.csv")
    print(output_dir / "individual_adaptive_failed_cases.csv")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()