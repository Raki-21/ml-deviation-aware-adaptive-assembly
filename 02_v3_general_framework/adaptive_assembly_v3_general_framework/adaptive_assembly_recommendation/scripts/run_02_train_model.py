import os
import sys

# Add project root folder to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.surrogate_models import train_models, save_model, FEATURE_COLUMNS
from src.visualization import plot_predicted_vs_actual, plot_feature_importance


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    dataset_path = "data/processed/assembly_deviation_dataset.csv"
    df = pd.read_csv(dataset_path)

    trained_models, metrics_df = train_models(df)

    metrics_path = "results/tables/model_metrics_table.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("Model training completed.")
    print(metrics_df)

    # Save Random Forest as the main surrogate model
    rf_model = trained_models["random_forest"]["model"]
    save_model(rf_model, "results/models/random_forest_surrogate.joblib")

    # Save predicted-vs-actual plot
    plot_predicted_vs_actual(
        y_test=trained_models["random_forest"]["y_test"],
        y_pred=trained_models["random_forest"]["y_pred"],
        output_path="results/plots/predicted_vs_actual_random_forest.png",
    )

    # Save feature-importance plot
    plot_feature_importance(
        model=rf_model,
        feature_names=FEATURE_COLUMNS,
        output_path="results/plots/feature_importance_random_forest.png",
    )

    print("Saved model, metrics table, and plots.")
    print("Saved model to: results/models/random_forest_surrogate.joblib")
    print("Saved metrics to: results/tables/model_metrics_table.csv")
    print("Saved plots to: results/plots/")


if __name__ == "__main__":
    main()