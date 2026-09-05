import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# VERSION 3.2 - SURROGATE MODEL DIAGNOSTICS
# ============================================================
#
# Purpose:
#
# Validate the best model from the model-screening experiment
# using predictions on completely unseen assemblies.
#
# Checks:
#
#   1. Automatically identify best model
#   2. Predicted vs actual quality
#   3. Residual distribution
#   4. Error percentiles
#   5. Error by component stage
#   6. Error for low / medium / high-quality cases
#
# This must be completed before Bayesian Optimization.
# ============================================================


SCREENING_FILE = (
    "results/tables/"
    "v3_2_ml_model_screening.csv"
)

PREDICTION_FILE = (
    "data/processed/"
    "v3_2_ml_test_predictions.csv"
)

OUTPUT_TABLE = (
    "results/tables/"
    "v3_2_best_surrogate_diagnostics.csv"
)

PLOT_DIR = (
    "results/plots"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    SCREENING_FILE,
    PREDICTION_FILE,
]


missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    print(
        "\nMissing required files:"
    )

    for file in missing_files:
        print(file)

    raise SystemExit(
        "\nRun screen_ml_models_v3_2.py first."
    )


os.makedirs(
    PLOT_DIR,
    exist_ok=True,
)

os.makedirs(
    "results/tables",
    exist_ok=True,
)


# ============================================================
# LOAD RESULTS
# ============================================================

screening_df = pd.read_csv(
    SCREENING_FILE
)

prediction_df = pd.read_csv(
    PREDICTION_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SURROGATE DIAGNOSTICS"
)

print(
    "============================================================"
)


# ============================================================
# IDENTIFY BEST MODEL
# ============================================================

screening_df = (
    screening_df
    .sort_values(
        by=[
            "R2",
            "RMSE",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


best_row = screening_df.iloc[0]

best_model_name = (
    best_row[
        "model"
    ]
)


prediction_column = (
    "pred_"
    + best_model_name
    .lower()
    .replace(
        " ",
        "_",
    )
)


print(
    f"\nBest model : "
    f"{best_model_name}"
)

print(
    f"MAE        : "
    f"{best_row['MAE']:.5f}"
)

print(
    f"RMSE       : "
    f"{best_row['RMSE']:.5f}"
)

print(
    f"R²         : "
    f"{best_row['R2']:.5f}"
)


if prediction_column not in prediction_df.columns:

    raise ValueError(
        f"\nPrediction column not found: "
        f"{prediction_column}"
    )


# ============================================================
# ACTUAL / PREDICTED
# ============================================================

actual = prediction_df[
    "actual_quality"
].to_numpy()


predicted = prediction_df[
    prediction_column
].to_numpy()


residual = (
    actual
    - predicted
)


absolute_error = np.abs(
    residual
)


prediction_df[
    "best_model_prediction"
] = predicted


prediction_df[
    "residual"
] = residual


prediction_df[
    "absolute_error"
] = absolute_error


# ============================================================
# BASIC ERROR DIAGNOSTICS
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ERROR DIAGNOSTICS"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Mean residual       : "
    f"{np.mean(residual):.5f}"
)

print(
    f"Residual std        : "
    f"{np.std(residual):.5f}"
)

print(
    f"Median abs. error   : "
    f"{np.median(absolute_error):.5f}"
)

print(
    f"90th percentile AE  : "
    f"{np.quantile(absolute_error, 0.90):.5f}"
)

print(
    f"95th percentile AE  : "
    f"{np.quantile(absolute_error, 0.95):.5f}"
)

print(
    f"99th percentile AE  : "
    f"{np.quantile(absolute_error, 0.99):.5f}"
)

print(
    f"Maximum abs. error  : "
    f"{np.max(absolute_error):.5f}"
)


# ============================================================
# ERROR BY COMPONENT STAGE
# ============================================================

stage_diagnostics = (
    prediction_df
    .groupby(
        "component_index"
    )
    .agg(
        sample_count=(
            "absolute_error",
            "count",
        ),

        mean_absolute_error=(
            "absolute_error",
            "mean",
        ),

        median_absolute_error=(
            "absolute_error",
            "median",
        ),

        p95_absolute_error=(
            "absolute_error",
            lambda x:
                x.quantile(0.95),
        ),

        mean_residual=(
            "residual",
            "mean",
        ),
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ERROR BY COMPONENT STAGE"
)

print(
    "------------------------------------------------------------"
)


print(
    stage_diagnostics.round(5)
)


# ============================================================
# QUALITY-RANGE DIAGNOSTICS
# ============================================================
#
# Divide true quality into approximate low / medium / high
# ranges using test-set thirds.
# ============================================================

q33 = prediction_df[
    "actual_quality"
].quantile(
    1.0 / 3.0
)

q67 = prediction_df[
    "actual_quality"
].quantile(
    2.0 / 3.0
)


prediction_df[
    "quality_region"
] = pd.cut(
    prediction_df[
        "actual_quality"
    ],
    bins=[
        -np.inf,
        q33,
        q67,
        np.inf,
    ],
    labels=[
        "low_quality_score",
        "medium_quality_score",
        "high_quality_score",
    ],
)


region_diagnostics = (
    prediction_df
    .groupby(
        "quality_region",
        observed=True,
    )
    .agg(
        sample_count=(
            "absolute_error",
            "count",
        ),

        mean_actual_quality=(
            "actual_quality",
            "mean",
        ),

        mean_absolute_error=(
            "absolute_error",
            "mean",
        ),

        p95_absolute_error=(
            "absolute_error",
            lambda x:
                x.quantile(0.95),
        ),

        mean_residual=(
            "residual",
            "mean",
        ),
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ERROR BY QUALITY REGION"
)

print(
    "------------------------------------------------------------"
)


print(
    region_diagnostics.round(5)
)


# ============================================================
# SAVE DIAGNOSTIC TABLE
# ============================================================

overall_diagnostics = pd.DataFrame(
    {
        "metric": [
            "best_model",
            "MAE",
            "RMSE",
            "R2",
            "mean_residual",
            "residual_std",
            "median_absolute_error",
            "p90_absolute_error",
            "p95_absolute_error",
            "p99_absolute_error",
            "maximum_absolute_error",
        ],

        "value": [
            best_model_name,
            best_row["MAE"],
            best_row["RMSE"],
            best_row["R2"],
            np.mean(residual),
            np.std(residual),
            np.median(absolute_error),
            np.quantile(
                absolute_error,
                0.90,
            ),
            np.quantile(
                absolute_error,
                0.95,
            ),
            np.quantile(
                absolute_error,
                0.99,
            ),
            np.max(
                absolute_error
            ),
        ],
    }
)


overall_diagnostics.to_csv(
    OUTPUT_TABLE,
    index=False,
)


stage_diagnostics.to_csv(
    "results/tables/"
    "v3_2_surrogate_error_by_stage.csv"
)


region_diagnostics.to_csv(
    "results/tables/"
    "v3_2_surrogate_error_by_quality_region.csv"
)


prediction_df.to_csv(
    "data/processed/"
    "v3_2_best_surrogate_diagnostic_predictions.csv",
    index=False,
)


# ============================================================
# PLOT 1 - PREDICTED VS ACTUAL
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.scatter(
    actual,
    predicted,
    alpha=0.20,
    s=12,
)


minimum = min(
    actual.min(),
    predicted.min(),
)

maximum = max(
    actual.max(),
    predicted.max(),
)


plt.plot(
    [
        minimum,
        maximum,
    ],
    [
        minimum,
        maximum,
    ],
    linestyle="--",
)


plt.xlabel(
    "Actual Quality Score"
)

plt.ylabel(
    "Predicted Quality Score"
)

plt.title(
    f"V3.2 Predicted vs Actual Quality\n"
    f"{best_model_name}"
)

plt.tight_layout()


plt.savefig(
    "results/plots/"
    "v3_2_predicted_vs_actual.png",
    dpi=300,
)


plt.close()


# ============================================================
# PLOT 2 - RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(7, 5)
)


plt.hist(
    residual,
    bins=60,
)


plt.axvline(
    0.0,
    linestyle="--",
)


plt.xlabel(
    "Residual (Actual - Predicted)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    f"V3.2 Surrogate Residual Distribution\n"
    f"{best_model_name}"
)

plt.tight_layout()


plt.savefig(
    "results/plots/"
    "v3_2_surrogate_residual_distribution.png",
    dpi=300,
)


plt.close()


# ============================================================
# PLOT 3 - ERROR BY COMPONENT STAGE
# ============================================================

plt.figure(
    figsize=(7, 5)
)


plt.plot(
    stage_diagnostics.index,
    stage_diagnostics[
        "mean_absolute_error"
    ],
    marker="o",
)


plt.xlabel(
    "Component Index"
)

plt.ylabel(
    "Mean Absolute Error"
)

plt.title(
    "V3.2 Surrogate Error Across Sequential Assembly"
)

plt.xticks(
    stage_diagnostics.index
)

plt.tight_layout()


plt.savefig(
    "results/plots/"
    "v3_2_surrogate_error_by_component_stage.png",
    dpi=300,
)


plt.close()


# ============================================================
# FINISH
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "FILES CREATED"
)

print(
    "------------------------------------------------------------"
)


print(
    OUTPUT_TABLE
)

print(
    "results/tables/"
    "v3_2_surrogate_error_by_stage.csv"
)

print(
    "results/tables/"
    "v3_2_surrogate_error_by_quality_region.csv"
)

print(
    "results/plots/"
    "v3_2_predicted_vs_actual.png"
)

print(
    "results/plots/"
    "v3_2_surrogate_residual_distribution.png"
)

print(
    "results/plots/"
    "v3_2_surrogate_error_by_component_stage.png"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SURROGATE DIAGNOSTICS COMPLETED"
)

print(
    "============================================================"
)