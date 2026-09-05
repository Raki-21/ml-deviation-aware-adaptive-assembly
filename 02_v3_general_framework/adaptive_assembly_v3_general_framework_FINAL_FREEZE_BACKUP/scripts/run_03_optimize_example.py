import os
import sys

# Add project root folder to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.surrogate_models import load_model
from src.optimization import compare_recommendation_methods
from src.visualization import plot_optimization_convergence


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    model = load_model("results/models/random_forest_surrogate.joblib")

    # Example unseen deviation case
    deviation_case = {
        "dev_offset": 1.2,
        "dev_tilt": 0.6,
        "dev_bend": -0.8,
        "dev_waviness": 0.4,
        "dev_noise": 0.05,
    }

    comparison_df, convergence_df = compare_recommendation_methods(
        model=model,
        deviation_case=deviation_case,
    )

    comparison_path = "results/tables/recommendation_example.csv"
    convergence_path = "results/tables/optimization_convergence_data.csv"

    comparison_df.to_csv(comparison_path, index=False)
    convergence_df.to_csv(convergence_path, index=False)

    plot_optimization_convergence(
        convergence_df=convergence_df,
        output_path="results/plots/optimization_convergence.png",
    )

    print("Optimization example completed.")
    print()
    print("Deviation case used:")
    for key, value in deviation_case.items():
        print(f"{key}: {value}")

    print()
    print("Recommendation comparison:")
    print(comparison_df)

    print()
    print(f"Saved recommendation table to: {comparison_path}")
    print(f"Saved convergence data to: {convergence_path}")
    print("Saved convergence plot to: results/plots/optimization_convergence.png")


if __name__ == "__main__":
    main()