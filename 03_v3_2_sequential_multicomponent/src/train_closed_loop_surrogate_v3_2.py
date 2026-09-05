import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    GroupShuffleSplit,
)


# ============================================================
# VERSION 3.2
# CLOSED-LOOP SURROGATE RETRAINING + FAIR COMPARISON
# ============================================================
#
# PURPOSE
#
# Diagnosis showed:
#
#   Surrogate error increases through sequential assembly.
#
# Main hypothesis:
#
#   The original RF was trained mainly on uncorrected states
#   and therefore suffered from closed-loop distribution shift.
#
# This experiment:
#
#   1. Uses the augmented closed-loop dataset.
#   2. Preserves assembly-wise train/test separation.
#   3. Uses the SAME holdout assemblies as the original model.
#   4. Evaluates the original RF on closed-loop test states.
#   5. Trains a new closed-loop RF.
#   6. Trains Gradient Boosting as nonlinear reference.
#   7. Evaluates error separately at Components 1 -> 5.
#
# IMPORTANT COMPUTATIONAL CHANGE:
#
# To keep Windows / VS Code responsive:
#
#   Random Forest:
#       200 trees
#       n_jobs = 2
#
#   Gradient Boosting:
#       180 estimators
#
# The methodology is unchanged.
# Only computational load is reduced.
#
# The original V3.2 model is NOT overwritten.
# ============================================================


# ============================================================
# PATHS
# ============================================================

CLOSED_LOOP_FILE = (
    "data/processed/"
    "v3_2_closed_loop_ml_training_dataset.csv"
)

ORIGINAL_TRAINING_FILE = (
    "data/processed/"
    "v3_2_ml_training_dataset.csv"
)

ORIGINAL_MODEL_FILE = (
    "models/"
    "v3_2_best_quality_surrogate.joblib"
)

ORIGINAL_FEATURE_FILE = (
    "models/"
    "v3_2_surrogate_features.txt"
)


NEW_MODEL_FILE = (
    "models/"
    "v3_2_closed_loop_quality_surrogate.joblib"
)

NEW_FEATURE_FILE = (
    "models/"
    "v3_2_closed_loop_surrogate_features.txt"
)


RESULT_FILE = (
    "results/tables/"
    "v3_2_closed_loop_surrogate_comparison.csv"
)

STAGE_FILE = (
    "results/tables/"
    "v3_2_closed_loop_surrogate_error_by_stage.csv"
)

TRAJECTORY_FILE = (
    "results/tables/"
    "v3_2_closed_loop_surrogate_error_by_trajectory.csv"
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

required_files = [
    CLOSED_LOOP_FILE,
    ORIGINAL_TRAINING_FILE,
    ORIGINAL_MODEL_FILE,
    ORIGINAL_FEATURE_FILE,
]


missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    print(
        "\nERROR - Required files missing:"
    )

    for file in missing_files:
        print(file)

    raise SystemExit(
        "\nComplete the closed-loop training-data generation first."
    )


os.makedirs(
    "models",
    exist_ok=True,
)

os.makedirs(
    "results/tables",
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

closed_df = pd.read_csv(
    CLOSED_LOOP_FILE
)

original_df = pd.read_csv(
    ORIGINAL_TRAINING_FILE
)


with open(
    ORIGINAL_FEATURE_FILE,
    "r",
    encoding="utf-8",
) as file:

    FEATURE_COLUMNS = [
        line.strip()
        for line in file
        if line.strip()
    ]


TARGET_COLUMN = (
    "target_quality_score"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CLOSED-LOOP SURROGATE RETRAINING"
)

print(
    "============================================================"
)


print(
    f"\nClosed-loop rows : "
    f"{len(closed_df)}"
)

print(
    f"Assemblies       : "
    f"{closed_df['assembly_id'].nunique()}"
)

print(
    f"Model features   : "
    f"{len(FEATURE_COLUMNS)}"
)


# ============================================================
# FEATURE CHECK
# ============================================================

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in closed_df.columns
]


if missing_features:

    raise ValueError(
        "\nMissing closed-loop features:\n"
        f"{missing_features}"
    )


# ============================================================
# RECONSTRUCT ORIGINAL ASSEMBLY-WISE HOLDOUT
# ============================================================

original_X = original_df[
    FEATURE_COLUMNS
]

original_y = original_df[
    TARGET_COLUMN
]

original_groups = original_df[
    "assembly_id"
]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)


original_train_indices, original_test_indices = next(
    splitter.split(
        original_X,
        original_y,
        groups=original_groups,
    )
)


train_assembly_ids = set(
    original_groups.iloc[
        original_train_indices
    ].unique()
)


test_assembly_ids = set(
    original_groups.iloc[
        original_test_indices
    ].unique()
)


overlap = (
    train_assembly_ids
    .intersection(
        test_assembly_ids
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ASSEMBLY HOLDOUT"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Training assemblies : "
    f"{len(train_assembly_ids)}"
)

print(
    f"Test assemblies     : "
    f"{len(test_assembly_ids)}"
)

print(
    f"Overlap             : "
    f"{len(overlap)}"
)


if len(overlap) != 0:

    raise RuntimeError(
        "Assembly leakage detected."
    )


print(
    "PASS - Original unseen-assembly holdout preserved."
)


# ============================================================
# APPLY SAME SPLIT TO CLOSED-LOOP DATA
# ============================================================

train_df = closed_df[
    closed_df[
        "assembly_id"
    ].isin(
        train_assembly_ids
    )
].copy()


test_df = closed_df[
    closed_df[
        "assembly_id"
    ].isin(
        test_assembly_ids
    )
].copy()


X_train = train_df[
    FEATURE_COLUMNS
]

y_train = train_df[
    TARGET_COLUMN
]


X_test = test_df[
    FEATURE_COLUMNS
]

y_test = test_df[
    TARGET_COLUMN
]


print(
    f"\nClosed-loop training rows : "
    f"{len(train_df)}"
)

print(
    f"Closed-loop test rows     : "
    f"{len(test_df)}"
)


# ============================================================
# METRIC HELPER
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    residual = (
        np.asarray(actual)
        -
        np.asarray(predicted)
    )

    absolute_error = np.abs(
        residual
    )


    return {
        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),

        "mean_signed_error":
            float(
                np.mean(
                    residual
                )
            ),

        "median_absolute_error":
            float(
                np.median(
                    absolute_error
                )
            ),

        "p95_absolute_error":
            float(
                np.quantile(
                    absolute_error,
                    0.95,
                )
            ),

        "p99_absolute_error":
            float(
                np.quantile(
                    absolute_error,
                    0.99,
                )
            ),
    }


# ============================================================
# 1. ORIGINAL RF ON CLOSED-LOOP TEST STATES
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "1. ORIGINAL RF ON CLOSED-LOOP STATES"
)

print(
    "------------------------------------------------------------"
)


original_model = joblib.load(
    ORIGINAL_MODEL_FILE
)


start = time.time()


original_predictions = (
    original_model.predict(
        X_test
    )
)


original_time = (
    time.time()
    - start
)


original_metrics = (
    calculate_metrics(
        y_test,
        original_predictions,
    )
)


print(
    f"MAE  = "
    f"{original_metrics['MAE']:.5f}"
)

print(
    f"RMSE = "
    f"{original_metrics['RMSE']:.5f}"
)

print(
    f"R²   = "
    f"{original_metrics['R2']:.5f}"
)

print(
    f"P95 AE = "
    f"{original_metrics['p95_absolute_error']:.5f}"
)


# ============================================================
# 2. NEW CLOSED-LOOP RANDOM FOREST
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "2. TRAIN NEW CLOSED-LOOP RANDOM FOREST"
)

print(
    "------------------------------------------------------------"
)

print(
    "\nUsing 200 trees and 2 CPU cores "
    "to keep the computer responsive."
)


closed_loop_rf = (
    RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=2,
    )
)


start = time.time()


closed_loop_rf.fit(
    X_train,
    y_train,
)


# ------------------------------------------------------------
# SAVE RF IMMEDIATELY AFTER TRAINING
# ------------------------------------------------------------
#
# If a later stage fails, the expensive RF training is already
# preserved on disk.
# ------------------------------------------------------------

joblib.dump(
    closed_loop_rf,
    NEW_MODEL_FILE,
)


with open(
    NEW_FEATURE_FILE,
    "w",
    encoding="utf-8",
) as file:

    for feature in FEATURE_COLUMNS:

        file.write(
            feature
            + "\n"
        )


print(
    "\nClosed-loop RF saved successfully:"
)

print(
    NEW_MODEL_FILE
)


rf_predictions = (
    closed_loop_rf.predict(
        X_test
    )
)


rf_time = (
    time.time()
    - start
)


rf_metrics = (
    calculate_metrics(
        y_test,
        rf_predictions,
    )
)


print(
    f"\nMAE  = "
    f"{rf_metrics['MAE']:.5f}"
)

print(
    f"RMSE = "
    f"{rf_metrics['RMSE']:.5f}"
)

print(
    f"R²   = "
    f"{rf_metrics['R2']:.5f}"
)

print(
    f"P95 AE = "
    f"{rf_metrics['p95_absolute_error']:.5f}"
)

print(
    f"Time = "
    f"{rf_time:.2f} sec"
)


# ============================================================
# 3. GRADIENT BOOSTING REFERENCE
# ============================================================

print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "3. TRAIN GRADIENT BOOSTING REFERENCE"
)

print(
    "------------------------------------------------------------"
)


gradient_boosting = (
    GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
)


start = time.time()


gradient_boosting.fit(
    X_train,
    y_train,
)


gb_predictions = (
    gradient_boosting.predict(
        X_test
    )
)


gb_time = (
    time.time()
    - start
)


gb_metrics = (
    calculate_metrics(
        y_test,
        gb_predictions,
    )
)


print(
    f"MAE  = "
    f"{gb_metrics['MAE']:.5f}"
)

print(
    f"RMSE = "
    f"{gb_metrics['RMSE']:.5f}"
)

print(
    f"R²   = "
    f"{gb_metrics['R2']:.5f}"
)

print(
    f"P95 AE = "
    f"{gb_metrics['p95_absolute_error']:.5f}"
)

print(
    f"Time = "
    f"{gb_time:.2f} sec"
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "model":
                "Original RF",

            **original_metrics,

            "runtime_sec":
                original_time,
        },

        {
            "model":
                "Closed-loop RF",

            **rf_metrics,

            "runtime_sec":
                rf_time,
        },

        {
            "model":
                "Closed-loop Gradient Boosting",

            **gb_metrics,

            "runtime_sec":
                gb_time,
        },
    ]
)


comparison_df = (
    comparison_df
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


# ============================================================
# STAGE-WISE COMPARISON
# ============================================================

prediction_df = test_df[
    [
        "assembly_id",
        "component_index",
        "trajectory_type",
        "candidate_source",
        TARGET_COLUMN,
    ]
].copy()


prediction_df[
    "original_rf_prediction"
] = (
    original_predictions
)


prediction_df[
    "closed_loop_rf_prediction"
] = (
    rf_predictions
)


prediction_df[
    "gradient_boosting_prediction"
] = (
    gb_predictions
)


prediction_df[
    "original_rf_abs_error"
] = np.abs(
    prediction_df[
        TARGET_COLUMN
    ]
    -
    prediction_df[
        "original_rf_prediction"
    ]
)


prediction_df[
    "closed_loop_rf_abs_error"
] = np.abs(
    prediction_df[
        TARGET_COLUMN
    ]
    -
    prediction_df[
        "closed_loop_rf_prediction"
    ]
)


prediction_df[
    "gradient_boosting_abs_error"
] = np.abs(
    prediction_df[
        TARGET_COLUMN
    ]
    -
    prediction_df[
        "gradient_boosting_prediction"
    ]
)


stage_summary = (
    prediction_df
    .groupby(
        "component_index"
    )
    .agg(
        samples=(
            TARGET_COLUMN,
            "count",
        ),

        original_rf_mae=(
            "original_rf_abs_error",
            "mean",
        ),

        closed_loop_rf_mae=(
            "closed_loop_rf_abs_error",
            "mean",
        ),

        gradient_boosting_mae=(
            "gradient_boosting_abs_error",
            "mean",
        ),

        original_rf_p95=(
            "original_rf_abs_error",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),

        closed_loop_rf_p95=(
            "closed_loop_rf_abs_error",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),
    )
)


stage_summary[
    "rf_mae_improvement_percent"
] = (
    (
        stage_summary[
            "original_rf_mae"
        ]
        -
        stage_summary[
            "closed_loop_rf_mae"
        ]
    )
    /
    stage_summary[
        "original_rf_mae"
    ]
    * 100.0
)


# ============================================================
# TRAJECTORY-WISE COMPARISON
# ============================================================

trajectory_summary = (
    prediction_df
    .groupby(
        "trajectory_type"
    )
    .agg(
        samples=(
            TARGET_COLUMN,
            "count",
        ),

        original_rf_mae=(
            "original_rf_abs_error",
            "mean",
        ),

        closed_loop_rf_mae=(
            "closed_loop_rf_abs_error",
            "mean",
        ),

        original_rf_p95=(
            "original_rf_abs_error",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),

        closed_loop_rf_p95=(
            "closed_loop_rf_abs_error",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),
    )
)


trajectory_summary[
    "mae_improvement_percent"
] = (
    (
        trajectory_summary[
            "original_rf_mae"
        ]
        -
        trajectory_summary[
            "closed_loop_rf_mae"
        ]
    )
    /
    trajectory_summary[
        "original_rf_mae"
    ]
    * 100.0
)


# ============================================================
# SAVE TABLES
# ============================================================

comparison_df.to_csv(
    RESULT_FILE,
    index=False,
)


stage_summary.to_csv(
    STAGE_FILE
)


trajectory_summary.to_csv(
    TRAJECTORY_FILE
)


prediction_df.to_csv(
    "data/processed/"
    "v3_2_closed_loop_surrogate_predictions.csv",
    index=False,
)


# ============================================================
# IMPORTANT FIX-1 METRICS
# ============================================================

overall_mae_improvement = (
    (
        original_metrics[
            "MAE"
        ]
        -
        rf_metrics[
            "MAE"
        ]
    )
    /
    original_metrics[
        "MAE"
    ]
    * 100.0
)


original_stage5_mae = float(
    stage_summary.loc[
        5,
        "original_rf_mae",
    ]
)


new_stage5_mae = float(
    stage_summary.loc[
        5,
        "closed_loop_rf_mae",
    ]
)


stage5_improvement = (
    (
        original_stage5_mae
        -
        new_stage5_mae
    )
    /
    original_stage5_mae
    * 100.0
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CLOSED-LOOP SURROGATE COMPARISON"
)

print(
    "============================================================\n"
)


print(
    comparison_df.round(5)
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
    stage_summary.round(5)
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ERROR BY TRAJECTORY TYPE"
)

print(
    "------------------------------------------------------------"
)


print(
    trajectory_summary.round(5)
)


print(
    "\n"
    "============================================================"
)

print(
    "FIX 1 EVALUATION"
)

print(
    "============================================================"
)


print(
    f"\nOverall RF MAE improvement : "
    f"{overall_mae_improvement:.2f}%"
)


print(
    f"Original RF Component-5 MAE : "
    f"{original_stage5_mae:.5f}"
)


print(
    f"Closed-loop RF Component-5 MAE : "
    f"{new_stage5_mae:.5f}"
)


print(
    f"Component-5 MAE improvement : "
    f"{stage5_improvement:.2f}%"
)


if (
    rf_metrics[
        "MAE"
    ]
    <
    original_metrics[
        "MAE"
    ]
):

    print(
        "\nPASS - Closed-loop training reduced "
        "overall prediction error."
    )

else:

    print(
        "\nWARNING - Closed-loop training did not "
        "reduce overall prediction error."
    )


if (
    new_stage5_mae
    <
    original_stage5_mae
):

    print(
        "PASS - Closed-loop training improved "
        "late-stage sequential prediction."
    )

else:

    print(
        "WARNING - Component-5 prediction "
        "did not improve."
    )


print(
    "\nSaved new model:"
)

print(
    NEW_MODEL_FILE
)


print(
    "\nSaved comparison:"
)

print(
    RESULT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CLOSED-LOOP SURROGATE RETRAINING COMPLETED"
)

print(
    "============================================================"
)