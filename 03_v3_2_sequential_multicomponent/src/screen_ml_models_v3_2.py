import os
import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    GroupShuffleSplit,
)

import joblib


# ============================================================
# VERSION 3.2 - ML MODEL SCREENING
# ============================================================
#
# Research question:
#
# Which surrogate model best predicts the resulting
# assembly quality in the NEW sequential, state-aware
# multi-component framework?
#
#
# IMPORTANT VALIDATION IMPROVEMENT:
#
# We split by assembly_id rather than randomly by rows.
#
# Why?
#
# Each assembly/component state has many candidate corrections.
# If nearly identical physical states appear in both train
# and test sets, performance could look artificially high.
#
# Group splitting ensures that complete assemblies remain
# either in training or testing.
#
#
# Models compared:
#
#   1. Linear Regression
#   2. Ridge Regression
#   3. Random Forest
#   4. Gradient Boosting
#   5. SVR
#
#
# Metrics:
#
#   MAE
#   RMSE
#   R²
#
# Lower MAE/RMSE = better
# Higher R²      = better
# ============================================================


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "v3_2_ml_training_dataset.csv"
)

RESULT_FILE = (
    "results/tables/"
    "v3_2_ml_model_screening.csv"
)

PREDICTION_FILE = (
    "data/processed/"
    "v3_2_ml_test_predictions.csv"
)

MODEL_DIR = (
    "models"
)


# ------------------------------------------------------------
# CHECK INPUT
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        "\nV3.2 ML training dataset not found.\n"
        "Run generate_ml_training_data_v3_2.py first."
    )


os.makedirs(
    "results/tables",
    exist_ok=True,
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 ML MODEL SCREENING"
)

print(
    "============================================================"
)


print(
    f"\nDataset rows    : {len(df)}"
)

print(
    f"Dataset columns : {len(df.columns)}"
)

print(
    f"Assemblies      : "
    f"{df['assembly_id'].nunique()}"
)


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = (
    "target_quality_score"
)


# ============================================================
# FEATURE SET
# ============================================================
#
# We deliberately use interpretable numerical features.
#
# IDs and target columns are excluded.
# ============================================================

FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # Current assembly state
    # --------------------------------------------------------

    "state_mean_gap",
    "state_max_gap",
    "state_parallelism",
    "state_rms",
    "state_quality",
    "state_signed_mean",
    "state_signed_end_difference",
    "state_estimated_angle_deg",

    # --------------------------------------------------------
    # Incoming component deviation
    # --------------------------------------------------------

    "offset_mm",
    "tilt_deg",
    "bend_mm",
    "waviness_mm",
    "twist_mm",
    "local_bump_mm",

    "component_profile_rms_mm",
    "component_parallelism_mm",

    "n_active_modes",

    # --------------------------------------------------------
    # Batch/process condition
    # --------------------------------------------------------

    "batch_offset_bias_mm",
    "batch_angular_bias_deg",
    "fixture_drift_mm",
    "variation_multiplier",

    # --------------------------------------------------------
    # Candidate correction
    # --------------------------------------------------------

    "z_adj",
    "theta_adj",
    "locator_offset",

    "correction_utilization",

    # --------------------------------------------------------
    # Sequential context
    # --------------------------------------------------------

    "component_index",
]


# ------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in df.columns
]


if missing_features:

    raise ValueError(
        "\nMissing required ML features:\n"
        f"{missing_features}"
    )


# ============================================================
# X / y / GROUPS
# ============================================================

X = df[
    FEATURE_COLUMNS
].copy()


y = df[
    TARGET_COLUMN
].copy()


groups = df[
    "assembly_id"
].copy()


# ============================================================
# ASSEMBLY-WISE TRAIN / TEST SPLIT
# ============================================================
#
# 80% assemblies for training
# 20% completely unseen assemblies for testing
# ============================================================

group_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)


train_indices, test_indices = next(
    group_splitter.split(
        X,
        y,
        groups=groups,
    )
)


X_train = X.iloc[
    train_indices
].copy()


X_test = X.iloc[
    test_indices
].copy()


y_train = y.iloc[
    train_indices
].copy()


y_test = y.iloc[
    test_indices
].copy()


train_assemblies = set(
    groups.iloc[
        train_indices
    ]
)


test_assemblies = set(
    groups.iloc[
        test_indices
    ]
)


# ------------------------------------------------------------
# LEAKAGE CHECK
# ------------------------------------------------------------

overlap = (
    train_assemblies
    .intersection(
        test_assemblies
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ASSEMBLY-WISE TRAIN / TEST SPLIT"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Training rows       : {len(X_train)}"
)

print(
    f"Test rows           : {len(X_test)}"
)

print(
    f"Training assemblies : {len(train_assemblies)}"
)

print(
    f"Test assemblies     : {len(test_assemblies)}"
)

print(
    f"Assembly overlap    : {len(overlap)}"
)


if len(overlap) == 0:

    print(
        "PASS - No assembly leakage between train and test."
    )

else:

    raise RuntimeError(
        "Assembly leakage detected."
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================


models = {


    # --------------------------------------------------------
    # 1. LINEAR REGRESSION
    # --------------------------------------------------------

    "Linear Regression":

        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),


    # --------------------------------------------------------
    # 2. RIDGE REGRESSION
    # --------------------------------------------------------

    "Ridge Regression":

        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "model",
                    Ridge(
                        alpha=1.0
                    ),
                ),
            ]
        ),


    # --------------------------------------------------------
    # 3. RANDOM FOREST
    # --------------------------------------------------------

    "Random Forest":

        RandomForestRegressor(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),


    # --------------------------------------------------------
    # 4. GRADIENT BOOSTING
    # --------------------------------------------------------

    "Gradient Boosting":

        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),


    # --------------------------------------------------------
    # 5. SVR
    # --------------------------------------------------------

    "SVR":

        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "model",
                    SVR(
                        kernel="rbf",
                        C=10.0,
                        epsilon=0.05,
                        gamma="scale",
                    ),
                ),
            ]
        ),
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

prediction_table = pd.DataFrame(
    {
        "assembly_id":
            df.iloc[
                test_indices
            ][
                "assembly_id"
            ].values,

        "component_index":
            df.iloc[
                test_indices
            ][
                "component_index"
            ].values,

        "candidate_id":
            df.iloc[
                test_indices
            ][
                "candidate_id"
            ].values,

        "actual_quality":
            y_test.values,
    }
)


best_model_name = None
best_model = None
best_r2 = -np.inf


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "MODEL TRAINING"
)

print(
    "------------------------------------------------------------\n"
)


for model_name, model in models.items():

    print(
        f"Training: {model_name}"
    )


    start_time = time.time()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    elapsed_time = (
        time.time()
        - start_time
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )


    mse = mean_squared_error(
        y_test,
        predictions,
    )


    rmse = np.sqrt(
        mse
    )


    r2 = r2_score(
        y_test,
        predictions,
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append(
        {
            "model":
                model_name,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2,

            "training_and_prediction_time_sec":
                elapsed_time,
        }
    )


    prediction_column = (
        "pred_"
        + model_name
        .lower()
        .replace(
            " ",
            "_",
        )
    )


    prediction_table[
        prediction_column
    ] = predictions


    print(
        f"  MAE  = {mae:.5f}"
    )

    print(
        f"  RMSE = {rmse:.5f}"
    )

    print(
        f"  R²   = {r2:.5f}"
    )

    print(
        f"  Time = {elapsed_time:.2f} sec"
    )


    # --------------------------------------------------------
    # TRACK BEST MODEL
    # --------------------------------------------------------

    if r2 > best_r2:

        best_r2 = r2

        best_model_name = (
            model_name
        )

        best_model = (
            model
        )


    print(
        "----------------------------------------------"
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
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
# SAVE BEST MODEL
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR
    + "/"
    + "v3_2_best_quality_surrogate.joblib"
)


joblib.dump(
    best_model,
    BEST_MODEL_PATH,
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_file = (
    MODEL_DIR
    + "/"
    + "v3_2_surrogate_features.txt"
)


with open(
    feature_file,
    "w",
    encoding="utf-8",
) as file:

    for feature in FEATURE_COLUMNS:

        file.write(
            feature
            + "\n"
        )


# ============================================================
# SAVE TABLES
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False,
)


prediction_table.to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL MODEL SCREENING RESULTS"
)

print(
    "============================================================\n"
)


print(
    results_df.round(5)
)


print(
    "\nBEST MODEL"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Selected model : "
    f"{best_model_name}"
)

print(
    f"Best test R²   : "
    f"{best_r2:.5f}"
)


print(
    "\nSaved best model:"
)

print(
    BEST_MODEL_PATH
)


print(
    "\nSaved feature definition:"
)

print(
    feature_file
)


print(
    "\nSaved model comparison:"
)

print(
    RESULT_FILE
)


print(
    "\nSaved test predictions:"
)

print(
    PREDICTION_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 ML MODEL SCREENING COMPLETED"
)

print(
    "============================================================"
)