"""
Phase 8 model training:
Objective-Aligned Residual Learning.

Purpose
-------
Train and evaluate two machine-learning model families for predicting
the residual correction remaining after Composite-Q:

    delta_u = u_best_reference - u_CompositeQ

Models
------
1. Random Forest
2. Gradient Boosting

Three independent regressors are trained for each model family:

    delta z
    delta theta
    delta locator

Important interpretation rule
-----------------------------
The Phase 8 target distribution is highly sparse because Composite-Q
already matches the stronger numerical reference on most decisions.

Therefore, model prediction errors are compared against a zero-residual
baseline. Low prediction error alone is not interpreted as evidence of
useful machine-learning value.

The decisive test remains the later closed-loop comparison of:

    Composite-Q
    Composite-Q + RF residual
    Composite-Q + GB residual

No Version 3.3 or Phase 1-7 files are modified.
"""

from pathlib import Path
import sys
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


# ============================================================
# PHASE 8 CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase8_config import (  # noqa: E402
    TRAINING_DATA_DIR,
    DEVELOPMENT_DATA_DIR,
    MODEL_DIR,
    TABLES_DIR,
    RF_N_ESTIMATORS,
    RF_MIN_SAMPLES_LEAF,
    RF_MAX_FEATURES,
    RF_N_JOBS,
    RF_RANDOM_STATE,
    GB_N_ESTIMATORS,
    GB_LEARNING_RATE,
    GB_MAX_DEPTH,
    GB_MIN_SAMPLES_LEAF,
    GB_RANDOM_STATE,
    ensure_phase8_directories,
)


# ============================================================
# INPUT FILES
# ============================================================

TRAINING_FILE = (
    TRAINING_DATA_DIR
    / "phase8_residual_learning_training.csv"
)

DEVELOPMENT_FILE = (
    DEVELOPMENT_DATA_DIR
    / "phase8_residual_learning_development.csv"
)

FEATURE_FILE = (
    TRAINING_DATA_DIR
    / "phase8_observable_features.txt"
)


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_METRICS_FILE = (
    TABLES_DIR
    / "phase8_residual_model_metrics.csv"
)

DEVELOPMENT_PREDICTIONS_FILE = (
    TABLES_DIR
    / "phase8_residual_model_development_predictions.csv"
)

MODEL_MANIFEST_FILE = (
    TABLES_DIR
    / "phase8_residual_model_manifest.csv"
)


# ============================================================
# TARGETS
# ============================================================

TARGET_COLUMNS = {
    "delta_z": "target_delta_z_mm",
    "delta_theta": "target_delta_theta_deg",
    "delta_locator": "target_delta_locator_mm",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_feature_columns():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Observable feature file not found:\n{FEATURE_FILE}"
        )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        columns = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not columns:
        raise RuntimeError(
            "Observable feature list is empty."
        )

    return columns


def load_datasets():
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAINING_FILE}"
        )

    if not DEVELOPMENT_FILE.exists():
        raise FileNotFoundError(
            f"Development dataset not found:\n{DEVELOPMENT_FILE}"
        )

    train_df = pd.read_csv(
        TRAINING_FILE
    )

    development_df = pd.read_csv(
        DEVELOPMENT_FILE
    )

    return train_df, development_df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_datasets(
    train_df,
    development_df,
    feature_columns,
):
    required_columns = (
        list(feature_columns)
        +
        list(TARGET_COLUMNS.values())
        +
        [
            "assembly_id",
            "assembly_index",
            "component_index",
        ]
    )

    for column in required_columns:
        if column not in train_df.columns:
            raise RuntimeError(
                f"Missing training column: {column}"
            )

        if column not in development_df.columns:
            raise RuntimeError(
                f"Missing development column: {column}"
            )

    train_ids = set(
        train_df["assembly_id"]
    )

    development_ids = set(
        development_df["assembly_id"]
    )

    overlap = (
        train_ids
        &
        development_ids
    )

    if overlap:
        raise RuntimeError(
            "Training/development assembly overlap detected."
        )

    train_matrix = (
        train_df[
            feature_columns
        ].to_numpy(
            dtype=float
        )
    )

    development_matrix = (
        development_df[
            feature_columns
        ].to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        train_matrix
    ).all():
        raise RuntimeError(
            "Non-finite training features detected."
        )

    if not np.isfinite(
        development_matrix
    ).all():
        raise RuntimeError(
            "Non-finite development features detected."
        )

    for target_column in TARGET_COLUMNS.values():
        if not np.isfinite(
            train_df[
                target_column
            ].to_numpy(
                dtype=float
            )
        ).all():
            raise RuntimeError(
                f"Non-finite training target detected: {target_column}"
            )

        if not np.isfinite(
            development_df[
                target_column
            ].to_numpy(
                dtype=float
            )
        ).all():
            raise RuntimeError(
                f"Non-finite development target detected: {target_column}"
            )


# ============================================================
# METRICS
# ============================================================

def safe_r2(
    y_true,
    y_pred,
):
    """
    Return R² if the development target has non-zero variance.

    If the target is constant, R² is not meaningful and NaN is returned.
    """

    if np.isclose(
        np.var(y_true),
        0.0,
        atol=1e-30,
    ):
        return np.nan

    return float(
        r2_score(
            y_true,
            y_pred,
        )
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    r2 = safe_r2(
        y_true,
        y_pred,
    )

    return {
        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,
    }


# ============================================================
# MODEL FACTORIES
# ============================================================

def create_random_forest():
    return RandomForestRegressor(
        n_estimators=
            RF_N_ESTIMATORS,

        min_samples_leaf=
            RF_MIN_SAMPLES_LEAF,

        max_features=
            RF_MAX_FEATURES,

        n_jobs=
            RF_N_JOBS,

        random_state=
            RF_RANDOM_STATE,
    )


def create_gradient_boosting():
    return GradientBoostingRegressor(
        n_estimators=
            GB_N_ESTIMATORS,

        learning_rate=
            GB_LEARNING_RATE,

        max_depth=
            GB_MAX_DEPTH,

        min_samples_leaf=
            GB_MIN_SAMPLES_LEAF,

        random_state=
            GB_RANDOM_STATE,

        loss=
            "squared_error",
    )


# ============================================================
# TRAIN ONE MODEL FAMILY
# ============================================================

def train_model_family(
    family_name,
    model_factory,
    X_train,
    X_development,
    train_df,
    development_df,
):
    models = {}
    predictions = {}
    metrics_rows = []

    for target_name, target_column in TARGET_COLUMNS.items():
        print(
            f"\nTraining {family_name} "
            f"for {target_name} ..."
        )

        y_train = (
            train_df[
                target_column
            ].to_numpy(
                dtype=float
            )
        )

        y_development = (
            development_df[
                target_column
            ].to_numpy(
                dtype=float
            )
        )

        model = (
            model_factory()
        )

        start_time = (
            time.time()
        )

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = float(
            time.time()
            -
            start_time
        )

        y_pred = (
            model.predict(
                X_development
            )
        )

        models[
            target_name
        ] = model

        predictions[
            target_name
        ] = y_pred

        model_metrics = (
            calculate_metrics(
                y_development,
                y_pred,
            )
        )

        # ----------------------------------------------------
        # ZERO-RESIDUAL BASELINE
        # ----------------------------------------------------

        zero_prediction = (
            np.zeros_like(
                y_development,
                dtype=float,
            )
        )

        zero_metrics = (
            calculate_metrics(
                y_development,
                zero_prediction,
            )
        )

        # ----------------------------------------------------
        # NON-ZERO TARGET SUBSET
        #
        # Descriptive only. This is useful because the majority
        # of Phase 8 residual targets are expected to be zero.
        # ----------------------------------------------------

        nonzero_mask = (
            np.abs(
                y_development
            )
            >
            1e-12
        )

        nonzero_count = int(
            nonzero_mask.sum()
        )

        if nonzero_count > 0:
            nonzero_model_metrics = (
                calculate_metrics(
                    y_development[
                        nonzero_mask
                    ],
                    y_pred[
                        nonzero_mask
                    ],
                )
            )

            nonzero_zero_metrics = (
                calculate_metrics(
                    y_development[
                        nonzero_mask
                    ],
                    zero_prediction[
                        nonzero_mask
                    ],
                )
            )

        else:
            nonzero_model_metrics = {
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
            }

            nonzero_zero_metrics = {
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
            }

        metrics_rows.append(
            {
                "model_family":
                    family_name,

                "target":
                    target_name,

                "development_rows":
                    len(
                        y_development
                    ),

                "nonzero_development_targets":
                    nonzero_count,

                "nonzero_development_percent":
                    float(
                        100.0
                        *
                        nonzero_count
                        /
                        len(
                            y_development
                        )
                    ),

                "model_mae":
                    model_metrics[
                        "mae"
                    ],

                "model_rmse":
                    model_metrics[
                        "rmse"
                    ],

                "model_r2":
                    model_metrics[
                        "r2"
                    ],

                "zero_baseline_mae":
                    zero_metrics[
                        "mae"
                    ],

                "zero_baseline_rmse":
                    zero_metrics[
                        "rmse"
                    ],

                "zero_baseline_r2":
                    zero_metrics[
                        "r2"
                    ],

                "mae_improvement_over_zero_percent":
                    float(
                        100.0
                        *
                        (
                            zero_metrics[
                                "mae"
                            ]
                            -
                            model_metrics[
                                "mae"
                            ]
                        )
                        /
                        max(
                            zero_metrics[
                                "mae"
                            ],
                            1e-30,
                        )
                    ),

                "rmse_improvement_over_zero_percent":
                    float(
                        100.0
                        *
                        (
                            zero_metrics[
                                "rmse"
                            ]
                            -
                            model_metrics[
                                "rmse"
                            ]
                        )
                        /
                        max(
                            zero_metrics[
                                "rmse"
                            ],
                            1e-30,
                        )
                    ),

                "nonzero_subset_model_mae":
                    nonzero_model_metrics[
                        "mae"
                    ],

                "nonzero_subset_zero_mae":
                    nonzero_zero_metrics[
                        "mae"
                    ],

                "training_seconds":
                    training_seconds,
            }
        )

        print(
            f"  Model MAE          : "
            f"{model_metrics['mae']:.10f}"
        )

        print(
            f"  Zero baseline MAE  : "
            f"{zero_metrics['mae']:.10f}"
        )

        print(
            f"  Model RMSE         : "
            f"{model_metrics['rmse']:.10f}"
        )

        print(
            f"  Zero baseline RMSE : "
            f"{zero_metrics['rmse']:.10f}"
        )

        print(
            f"  R2                 : "
            f"{model_metrics['r2']}"
        )

        print(
            f"  Non-zero targets   : "
            f"{nonzero_count}/{len(y_development)}"
        )

    return (
        models,
        predictions,
        metrics_rows,
    )


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(
    family_name,
    models,
):
    manifest_rows = []

    family_slug = (
        family_name
        .lower()
        .replace(
            " ",
            "_",
        )
    )

    for target_name, model in models.items():
        model_file = (
            MODEL_DIR
            /
            (
                f"phase8_{family_slug}_"
                f"{target_name}.joblib"
            )
        )

        joblib.dump(
            model,
            model_file,
        )

        manifest_rows.append(
            {
                "model_family":
                    family_name,

                "target":
                    target_name,

                "model_file":
                    str(
                        model_file
                    ),
            }
        )

    return manifest_rows


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_phase8_directories()

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8: RESIDUAL MODEL TRAINING"
    )

    print(
        "============================================================"
    )

    feature_columns = (
        load_feature_columns()
    )

    train_df, development_df = (
        load_datasets()
    )

    validate_datasets(
        train_df,
        development_df,
        feature_columns,
    )

    X_train = (
        train_df[
            feature_columns
        ].to_numpy(
            dtype=float
        )
    )

    X_development = (
        development_df[
            feature_columns
        ].to_numpy(
            dtype=float
        )
    )

    print(
        f"\nTraining rows          : {len(train_df)}"
    )

    print(
        f"Development rows       : {len(development_df)}"
    )

    print(
        f"Observable features    : {len(feature_columns)}"
    )

    print(
        f"Targets                : {len(TARGET_COLUMNS)}"
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    (
        rf_models,
        rf_predictions,
        rf_metrics,
    ) = train_model_family(
        family_name=
            "Random Forest",

        model_factory=
            create_random_forest,

        X_train=
            X_train,

        X_development=
            X_development,

        train_df=
            train_df,

        development_df=
            development_df,
    )

    # ========================================================
    # GRADIENT BOOSTING
    # ========================================================

    (
        gb_models,
        gb_predictions,
        gb_metrics,
    ) = train_model_family(
        family_name=
            "Gradient Boosting",

        model_factory=
            create_gradient_boosting,

        X_train=
            X_train,

        X_development=
            X_development,

        train_df=
            train_df,

        development_df=
            development_df,
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    manifest_rows = []

    manifest_rows.extend(
        save_models(
            "Random Forest",
            rf_models,
        )
    )

    manifest_rows.extend(
        save_models(
            "Gradient Boosting",
            gb_models,
        )
    )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_df.to_csv(
        MODEL_MANIFEST_FILE,
        index=False,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_df = pd.DataFrame(
        rf_metrics
        +
        gb_metrics
    )

    metrics_df.to_csv(
        MODEL_METRICS_FILE,
        index=False,
    )

    # ========================================================
    # SAVE DEVELOPMENT PREDICTIONS
    # ========================================================

    prediction_df = development_df[
        [
            "assembly_id",
            "assembly_index",
            "component_index",
        ]
    ].copy()

    for target_name, target_column in TARGET_COLUMNS.items():
        prediction_df[
            f"actual_{target_name}"
        ] = (
            development_df[
                target_column
            ].to_numpy(
                dtype=float
            )
        )

        prediction_df[
            f"rf_pred_{target_name}"
        ] = (
            rf_predictions[
                target_name
            ]
        )

        prediction_df[
            f"gb_pred_{target_name}"
        ] = (
            gb_predictions[
                target_name
            ]
        )

    prediction_df.to_csv(
        DEVELOPMENT_PREDICTIONS_FILE,
        index=False,
    )

    # ========================================================
    # FINAL PRINT
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 MODEL-TRAINING SUMMARY"
    )

    print(
        "============================================================"
    )

    display_columns = [
        "model_family",
        "target",
        "model_mae",
        "zero_baseline_mae",
        "mae_improvement_over_zero_percent",
        "model_rmse",
        "zero_baseline_rmse",
        "rmse_improvement_over_zero_percent",
        "model_r2",
        "nonzero_development_targets",
    ]

    print(
        "\n"
        +
        metrics_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved model metrics:"
    )

    print(
        MODEL_METRICS_FILE
    )

    print(
        "\nSaved development predictions:"
    )

    print(
        DEVELOPMENT_PREDICTIONS_FILE
    )

    print(
        "\nSaved model manifest:"
    )

    print(
        MODEL_MANIFEST_FILE
    )

    print(
        "\nPASS: Random Forest and Gradient Boosting residual models "
        "trained and evaluated on the independent development set."
    )

    print(
        "\nImportant:"
    )

    print(
        "Prediction metrics do not establish controller-level benefit."
    )

    print(
        "Closed-loop Composite-Q versus Composite-Q+ML evaluation "
        "is required next."
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 RESIDUAL MODEL TRAINING COMPLETED"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()