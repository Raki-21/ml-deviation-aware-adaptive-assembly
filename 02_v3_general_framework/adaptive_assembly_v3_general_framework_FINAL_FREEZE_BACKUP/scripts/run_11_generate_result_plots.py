"""
Run script for Version 3 thesis result plots.

Input:
- results/tables/*.csv

Output:
- results/plots/v3/*.png
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from src.v3_result_plots import generate_all_v3_result_plots


def main() -> None:
    table_dir = PROJECT_ROOT / "results" / "tables"
    plot_dir = PROJECT_ROOT / "results" / "plots" / "v3"

    print("Generating Version 3 result plots...")
    print(f"Input table directory: {table_dir}")
    print(f"Output plot directory: {plot_dir}")

    generate_all_v3_result_plots(
        table_dir=table_dir,
        plot_dir=plot_dir,
    )

    print("")
    print("Generated plots:")
    for plot_path in sorted(plot_dir.glob("*.png")):
        print(plot_path)

    print("")
    print("Done.")


if __name__ == "__main__":
    main()