"""
Run script for Version 3 ML model screening.

Input:
- results/tables/batch_aware_dataset.csv

Outputs:
- results/tables/model_screening_results.csv
- results/tables/model_screening_summary.csv

Purpose:
Compare candidate surrogate models and justify the final model choice.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from src.model_screening import save_model_screening_results


def main() -> None:
    dataset_path = PROJECT_ROOT / "results" / "tables" / "batch_aware_dataset.csv"
    output_dir = PROJECT_ROOT / "results" / "tables"

    print("Running Version 3 ML model screening...")
    print(f"Input dataset: {dataset_path}")
    print("This may take a few minutes.")

    results, summary = save_model_screening_results(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    print("")
    print("Detailed model screening results:")
    print(results.to_string(index=False))

    print("")
    print("Model screening summary:")
    print(summary.to_string(index=False))

    print("")
    print("Saved files:")
    print(output_dir / "model_screening_results.csv")
    print(output_dir / "model_screening_summary.csv")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()