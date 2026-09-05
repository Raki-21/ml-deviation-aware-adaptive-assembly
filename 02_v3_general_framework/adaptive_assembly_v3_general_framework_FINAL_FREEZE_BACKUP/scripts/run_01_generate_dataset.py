import os
import sys

# Add project root folder to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset_builder import generate_dataset
from src.config import DEFAULT_DATASET_SIZE


def main():
    os.makedirs("data/processed", exist_ok=True)

    df = generate_dataset(n_samples=DEFAULT_DATASET_SIZE)

    output_path = "data/processed/assembly_deviation_dataset.csv"
    df.to_csv(output_path, index=False)

    print("Dataset generated successfully.")
    print(f"Rows: {len(df)}")
    print(f"Saved to: {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()