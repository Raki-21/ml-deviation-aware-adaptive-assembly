"""
V3.3 residual-learning pilot model training.

Purpose
-------
Test whether the nonlinear correction residual beyond analytical
LSQ can be learned from observable-only features.

Target:

    delta_u =
        [
            delta_z,
            delta_theta,
            delta_locator
        ]

where:

    u_reference = u_LSQ + delta_u

This is a prediction pilot only.

It does NOT yet prove closed-loop hybrid improvement.
Closed-loop hybrid validation is a later mandatory step.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent


V33_ROOT = (
    SCRIPT_DIR
    .parent
)


DATA_DIR = (
    V33_ROOT
    / "data"
    / "residual_learning_full"
)


MODEL_DIR = (
    V33_ROOT
    / "models"
    / "residual_learning_full"
)


RESULTS_DIR = (
    V33_ROOT
    / "results"
    / "residual_learning_full"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_FILE = (
    DATA_DIR
    / "v33_residual_learning_full_train.csv"
)


VALIDATION_FILE = (
    DATA_DIR
    / "v33_residual_learning_full_development.csv"
)


FEATURE_FILE = (
    DATA_DIR
    / "v33_residual_learning_full_observable_features.txt"
)


MODEL_FILE = (
    MODEL_DIR
    / "v33_residual_rf_full.joblib"
)


PREDICTION_FILE = (
    RESULTS_DIR
    / "v33_residual_rf_full_development_predictions.csv"
)


CV_FILE = (
    RESULTS_DIR
    / "v33_residual_rf_full_group_cv.csv"
)


SUMMARY_FILE = (
    RESULTS_DIR
    / "v33_residual_rf_full_summary.csv"
)


IMPORTANCE_FILE = (
    RESULTS_DIR
    / "v33_residual_rf_full_feature_importance.csv"
)


# ============================================================
# TARGETS
# ============================================================

TARGET_COLUMNS = [
    "target_delta_z_mm",
    "target_delta_theta_deg",
    "target_delta_locator_mm",
]


TARGET_LABELS = {
    "target_delta_z_mm":
        "delta_z_mm",

    "target_delta_theta_deg":
        "delta_theta_deg",

    "target_delta_locator_mm":
        "delta_locator_mm",
}


# ============================================================
# RANDOM FOREST SETTINGS
# ============================================================

RANDOM_SEED = 20260908


RF_PARAMETERS = {
    "n_estimators":
        500,

    "max_depth":
        None,

    "min_samples_leaf":
        2,

    "max_features":
        "sqrt",

    "random_state":
        RANDOM_SEED,

    "n_jobs":
        2,
}


# ============================================================
# HELPERS
# ============================================================

def load_feature_columns():
    """
    Read the observable-only feature list.
    """

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        feature_columns = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return feature_columns


def calculate_metrics(
    y_true,
    y_pred,
    target_name,
):
    """
    Calculate target-specific regression metrics.
    """

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


    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )


    mean_abs_target = float(
        np.mean(
            np.abs(
                y_true
            )
        )
    )


    zero_prediction_mae = float(
        mean_absolute_error(
            y_true,
            np.zeros_like(
                y_true
            ),
        )
    )


    relative_mae_percent = float(
        100.0
        *
        mae
        /
        max(
            mean_abs_target,
            1e-12,
        )
    )


    improvement_vs_zero_percent = float(
        100.0
        *
        (
            zero_prediction_mae
            -
            mae
        )
        /
        max(
            zero_prediction_mae,
            1e-12,
        )
    )


    return {
        "target":
            target_name,

        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,

        "mean_abs_target":
            mean_abs_target,

        "zero_prediction_mae":
            zero_prediction_mae,

        "relative_mae_percent":
            relative_mae_percent,

        "improvement_vs_zero_percent":
            improvement_vs_zero_percent,
    }


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 RESIDUAL-LEARNING FULL TRAINING"
)

print(
    "============================================================"
)


if not TRAIN_FILE.exists():

    raise FileNotFoundError(
        f"\nTraining file not found:\n{TRAIN_FILE}"
    )


if not VALIDATION_FILE.exists():

    raise FileNotFoundError(
        f"\nValidation file not found:\n{VALIDATION_FILE}"
    )


if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"\nFeature file not found:\n{FEATURE_FILE}"
    )


train_df = pd.read_csv(
    TRAIN_FILE
)


validation_df = pd.read_csv(
    VALIDATION_FILE
)


FEATURE_COLUMNS = (
    load_feature_columns()
)


print(
    f"\nTraining rows           : {len(train_df)}"
)


print(
    f"Validation rows         : {len(validation_df)}"
)


print(
    f"Observable features     : {len(FEATURE_COLUMNS)}"
)


print(
    f"Residual targets        : {len(TARGET_COLUMNS)}"
)


# ============================================================
# BASIC DATA CHECKS
# ============================================================

missing_train_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in train_df.columns
]


missing_validation_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in validation_df.columns
]


missing_targets = [
    column
    for column in TARGET_COLUMNS
    if (
        column not in train_df.columns
        or
        column not in validation_df.columns
    )
]


if missing_train_features:

    raise ValueError(
        "\nMissing training features:\n"
        +
        "\n".join(
            missing_train_features
        )
    )


if missing_validation_features:

    raise ValueError(
        "\nMissing validation features:\n"
        +
        "\n".join(
            missing_validation_features
        )
    )


if missing_targets:

    raise ValueError(
        "\nMissing residual targets:\n"
        +
        "\n".join(
            missing_targets
        )
    )


train_ids = set(
    train_df[
        "assembly_id"
    ]
)


validation_ids = set(
    validation_df[
        "assembly_id"
    ]
)


assembly_overlap = (
    train_ids
    &
    validation_ids
)


if assembly_overlap:

    raise RuntimeError(
        "\nTrain/validation assembly overlap detected."
    )


print(
    "\nAssembly overlap        : 0"
)


# ============================================================
# MATRICES
# ============================================================

X_train = (
    train_df[
        FEATURE_COLUMNS
    ]
    .to_numpy(
        dtype=float
    )
)


Y_train = (
    train_df[
        TARGET_COLUMNS
    ]
    .to_numpy(
        dtype=float
    )
)


X_validation = (
    validation_df[
        FEATURE_COLUMNS
    ]
    .to_numpy(
        dtype=float
    )
)


Y_validation = (
    validation_df[
        TARGET_COLUMNS
    ]
    .to_numpy(
        dtype=float
    )
)


groups = (
    train_df[
        "assembly_id"
    ]
    .astype(
        str
    )
    .to_numpy()
)


# ============================================================
# 5-FOLD ASSEMBLY-GROUPED CROSS VALIDATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "ASSEMBLY-GROUPED CROSS VALIDATION"
)

print(
    "============================================================"
)


group_kfold = GroupKFold(
    n_splits=5
)


cv_rows = []


for fold_index, (
    train_index,
    test_index,
) in enumerate(
    group_kfold.split(
        X_train,
        Y_train,
        groups=groups,
    ),
    start=1,
):

    fold_model = RandomForestRegressor(
        **RF_PARAMETERS
    )


    fold_model.fit(
        X_train[
            train_index
        ],
        Y_train[
            train_index
        ],
    )


    fold_prediction = (
        fold_model.predict(
            X_train[
                test_index
            ]
        )
    )


    for target_index, target_column in enumerate(
        TARGET_COLUMNS
    ):

        metrics = calculate_metrics(
            y_true=Y_train[
                test_index,
                target_index,
            ],
            y_pred=fold_prediction[
                :,
                target_index,
            ],
            target_name=TARGET_LABELS[
                target_column
            ],
        )


        metrics[
            "fold"
        ] = fold_index


        metrics[
            "test_rows"
        ] = len(
            test_index
        )


        metrics[
            "test_assemblies"
        ] = len(
            set(
                groups[
                    test_index
                ]
            )
        )


        cv_rows.append(
            metrics
        )


    print(
        f"Completed fold {fold_index}/5"
    )


cv_df = pd.DataFrame(
    cv_rows
)


cv_df.to_csv(
    CV_FILE,
    index=False,
)


# ============================================================
# TRAIN FINAL FULL MODEL
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL FULL MODEL"
)

print(
    "============================================================"
)


model = RandomForestRegressor(
    **RF_PARAMETERS
)


model.fit(
    X_train,
    Y_train,
)


joblib.dump(
    model,
    MODEL_FILE,
)


validation_prediction = (
    model.predict(
        X_validation
    )
)


# ============================================================
# HOLDOUT VALIDATION METRICS
# ============================================================

validation_metric_rows = []


for target_index, target_column in enumerate(
    TARGET_COLUMNS
):

    metrics = calculate_metrics(
        y_true=Y_validation[
            :,
            target_index,
        ],
        y_pred=validation_prediction[
            :,
            target_index,
        ],
        target_name=TARGET_LABELS[
            target_column
        ],
    )


    validation_metric_rows.append(
        metrics
    )


validation_metrics_df = pd.DataFrame(
    validation_metric_rows
)


# ============================================================
# VECTOR-LEVEL RESIDUAL DIAGNOSTICS
# ============================================================

true_vector_norm = np.linalg.norm(
    Y_validation,
    axis=1,
)


prediction_error_vector = (
    validation_prediction
    -
    Y_validation
)


prediction_error_norm = np.linalg.norm(
    prediction_error_vector,
    axis=1,
)


mean_true_vector_norm = float(
    np.mean(
        true_vector_norm
    )
)


mean_prediction_error_norm = float(
    np.mean(
        prediction_error_norm
    )
)


median_prediction_error_norm = float(
    np.median(
        prediction_error_norm
    )
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_df = validation_df[
    [
        "assembly_id",
        "assembly_index",
        "component_index",
        "lsq_true_quality",
        "reference_true_quality",
        "reference_gain_over_lsq",
        "reference_relative_gain_percent",
    ]
].copy()


for target_index, target_column in enumerate(
    TARGET_COLUMNS
):

    short_name = TARGET_LABELS[
        target_column
    ]


    prediction_df[
        f"true_{short_name}"
    ] = (
        Y_validation[
            :,
            target_index
        ]
    )


    prediction_df[
        f"predicted_{short_name}"
    ] = (
        validation_prediction[
            :,
            target_index
        ]
    )


    prediction_df[
        f"error_{short_name}"
    ] = (
        validation_prediction[
            :,
            target_index
        ]
        -
        Y_validation[
            :,
            target_index
        ]
    )


prediction_df[
    "true_residual_vector_norm"
] = true_vector_norm


prediction_df[
    "prediction_error_vector_norm"
] = prediction_error_norm


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature":
            FEATURE_COLUMNS,

        "importance":
            model.feature_importances_,
    }
)


importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
).reset_index(
    drop=True
)


importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False,
)


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

cv_summary = (
    cv_df
    .groupby(
        "target"
    )
    .agg(
        mean_mae=(
            "mae",
            "mean",
        ),

        std_mae=(
            "mae",
            "std",
        ),

        mean_r2=(
            "r2",
            "mean",
        ),

        std_r2=(
            "r2",
            "std",
        ),

        mean_improvement_vs_zero_percent=(
            "improvement_vs_zero_percent",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []


for _, row in validation_metrics_df.iterrows():

    summary_rows.append(
        {
            "scope":
                "independent_validation",

            "target":
                row[
                    "target"
                ],

            "metric":
                "mae",

            "value":
                row[
                    "mae"
                ],
        }
    )


    summary_rows.append(
        {
            "scope":
                "independent_validation",

            "target":
                row[
                    "target"
                ],

            "metric":
                "rmse",

            "value":
                row[
                    "rmse"
                ],
        }
    )


    summary_rows.append(
        {
            "scope":
                "independent_validation",

            "target":
                row[
                    "target"
                ],

            "metric":
                "r2",

            "value":
                row[
                    "r2"
                ],
        }
    )


    summary_rows.append(
        {
            "scope":
                "independent_validation",

            "target":
                row[
                    "target"
                ],

            "metric":
                "improvement_vs_zero_percent",

            "value":
                row[
                    "improvement_vs_zero_percent"
                ],
        }
    )


for _, row in cv_summary.iterrows():

    summary_rows.append(
        {
            "scope":
                "grouped_cross_validation",

            "target":
                row[
                    "target"
                ],

            "metric":
                "mean_mae",

            "value":
                row[
                    "mean_mae"
                ],
        }
    )


    summary_rows.append(
        {
            "scope":
                "grouped_cross_validation",

            "target":
                row[
                    "target"
                ],

            "metric":
                "mean_r2",

            "value":
                row[
                    "mean_r2"
                ],
        }
    )


summary_rows.extend(
    [
        {
            "scope":
                "dataset",

            "target":
                "all",

            "metric":
                "training_rows",

            "value":
                len(
                    train_df
                ),
        },

        {
            "scope":
                "dataset",

            "target":
                "all",

            "metric":
                "validation_rows",

            "value":
                len(
                    validation_df
                ),
        },

        {
            "scope":
                "dataset",

            "target":
                "all",

            "metric":
                "observable_feature_count",

            "value":
                len(
                    FEATURE_COLUMNS
                ),
        },

        {
            "scope":
                "vector",

            "target":
                "all",

            "metric":
                "mean_true_residual_vector_norm",

            "value":
                mean_true_vector_norm,
        },

        {
            "scope":
                "vector",

            "target":
                "all",

            "metric":
                "mean_prediction_error_vector_norm",

            "value":
                mean_prediction_error_norm,
        },

        {
            "scope":
                "vector",

            "target":
                "all",

            "metric":
                "median_prediction_error_vector_norm",

            "value":
                median_prediction_error_norm,
        },
    ]
)


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# PRINT VALIDATION RESULTS
# ============================================================

print(
    "\nIndependent validation results:"
)


for _, row in validation_metrics_df.iterrows():

    print(
        "\n"
        f"{row['target']}"
    )


    print(
        f"  MAE                   : "
        f"{row['mae']:.6f}"
    )


    print(
        f"  RMSE                  : "
        f"{row['rmse']:.6f}"
    )


    print(
        f"  R2                    : "
        f"{row['r2']:.4f}"
    )


    print(
        f"  Mean |target|         : "
        f"{row['mean_abs_target']:.6f}"
    )


    print(
        f"  Zero-residual MAE     : "
        f"{row['zero_prediction_mae']:.6f}"
    )


    print(
        f"  Improvement vs zero   : "
        f"{row['improvement_vs_zero_percent']:.2f}%"
    )


print(
    "\nVector diagnostics:"
)


print(
    f"Mean true residual norm       : "
    f"{mean_true_vector_norm:.6f}"
)


print(
    f"Mean prediction-error norm    : "
    f"{mean_prediction_error_norm:.6f}"
)


print(
    f"Median prediction-error norm  : "
    f"{median_prediction_error_norm:.6f}"
)


print(
    "\nGrouped CV mean results:"
)


for _, row in cv_summary.iterrows():

    print(
        f"\n{row['target']}"
    )


    print(
        f"  Mean MAE              : "
        f"{row['mean_mae']:.6f}"
    )


    print(
        f"  Mean R2               : "
        f"{row['mean_r2']:.4f}"
    )


    print(
        f"  Improvement vs zero   : "
        f"{row['mean_improvement_vs_zero_percent']:.2f}%"
    )


# ============================================================
# PILOT LEARNABILITY GATE
# ============================================================

validation_improvements = (
    validation_metrics_df[
        "improvement_vs_zero_percent"
    ]
    .to_numpy(
        dtype=float
    )
)


n_targets_better_than_zero = int(
    np.sum(
        validation_improvements
        >
        0.0
    )
)


mean_improvement_vs_zero = float(
    np.mean(
        validation_improvements
    )
)


print(
    "\n"
    "============================================================"
)

print(
    "FULL RESIDUAL ML LEARNABILITY GATE"
)

print(
    "============================================================"
)


print(
    f"\nTargets better than zero-residual baseline: "
    f"{n_targets_better_than_zero}/3"
)


print(
    f"Mean improvement vs zero-residual baseline : "
    f"{mean_improvement_vs_zero:.2f}%"
)


if (
    n_targets_better_than_zero
    ==
    3
    and
    mean_improvement_vs_zero
    >=
    20.0
):

    print(
        "\nSTRONG PASS:"
    )

    print(
        "All three residual components show meaningful "
        "observable-feature predictability."
    )


elif (
    n_targets_better_than_zero
    >=
    2
    and
    mean_improvement_vs_zero
    >
    0.0
):

    print(
        "\nPASS:"
    )

    print(
        "The residual contains learnable structure, "
        "but prediction quality is uneven across targets."
    )


else:

    print(
        "\nWEAK / FAIL:"
    )

    print(
        "The present residual target is not reliably "
        "predictable from the observable feature set."
    )


print(
    "\nImportant:"
)


print(
    "This gate evaluates residual predictability only."
)


print(
    "It does NOT yet prove that applying the predicted "
    "residual improves closed-loop assembly quality."
)


print(
    "\nSaved model:"
)


print(
    MODEL_FILE
)


print(
    "\nSaved predictions:"
)


print(
    PREDICTION_FILE
)


print(
    "\nSaved grouped CV:"
)


print(
    CV_FILE
)


print(
    "\nSaved summary:"
)


print(
    SUMMARY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 RESIDUAL-LEARNING FULL TRAINING COMPLETED"
)

print(
    "============================================================"
)
