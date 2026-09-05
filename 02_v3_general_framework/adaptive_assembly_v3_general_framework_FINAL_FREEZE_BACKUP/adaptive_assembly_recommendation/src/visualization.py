import matplotlib.pyplot as plt
import pandas as pd


def plot_predicted_vs_actual(y_test, y_pred, output_path: str):
    """
    Plot predicted quality score against actual quality score.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)

    min_value = min(min(y_test), min(y_pred))
    max_value = max(max(y_test), max(y_pred))

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual quality score")
    plt.ylabel("Predicted quality score")
    plt.title("Predicted vs Actual Quality Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(model, feature_names, output_path: str):
    """
    Plot feature importance for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print("Selected model does not provide feature importance.")
        return

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_optimization_convergence(convergence_df: pd.DataFrame, output_path: str):
    """
    Plot optimization convergence for random search and Bayesian optimization.
    """
    plt.figure(figsize=(8, 5))

    for method in convergence_df["method"].unique():
        subset = convergence_df[convergence_df["method"] == method]
        plt.plot(subset["iteration"], subset["best_score"], label=method)

    plt.xlabel("Iteration")
    plt.ylabel("Best predicted quality score")
    plt.title("Optimization Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()