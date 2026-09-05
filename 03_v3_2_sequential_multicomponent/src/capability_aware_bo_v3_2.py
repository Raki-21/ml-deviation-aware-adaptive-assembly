import os
import joblib
import numpy as np
import pandas as pd

from skopt import gp_minimize
from skopt.space import Real


# ============================================================
# VERSION 3.2
# CAPABILITY-AWARE / SAFER BAYESIAN OPTIMIZATION
# ============================================================
#
# PURPOSE
#
# Closed-loop diagnosis identified two important weaknesses:
#
#   1. surrogate distribution shift
#      -> addressed through closed-loop surrogate retraining
#
#   2. aggressive recommendations near correction limits
#      -> addressed here
#
#
# The optimizer no longer minimizes only:
#
#       predicted quality
#
# It minimizes:
#
#       J = predicted_quality
#           + capability_penalty
#
#
# The penalty is deliberately mild at moderate utilization
# and rises strongly as correction utilization approaches 1.0.
#
# This does NOT prohibit boundary use.
# It simply requires a clear predicted quality benefit before
# the optimizer chooses an aggressive correction.
# ============================================================


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = (
    "models/"
    "v3_2_closed_loop_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_closed_loop_surrogate_features.txt"
)

TEST_DATA_FILE = (
    "data/processed/"
    "v3_2_closed_loop_ml_training_dataset.csv"
)

OUTPUT_FILE = (
    "results/tables/"
    "v3_2_capability_aware_bo_smoke_test.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    MODEL_FILE,
    FEATURE_FILE,
    TEST_DATA_FILE,
]


missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing_files:

    print(
        "\nERROR - Required files are missing:"
    )

    for path in missing_files:
        print(path)

    raise SystemExit(
        "\nComplete Fix 1 first."
    )


# ============================================================
# LOAD MODEL + FEATURES
# ============================================================

model = joblib.load(
    MODEL_FILE
)


with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8",
) as file:

    FEATURE_COLUMNS = [
        line.strip()
        for line in file
        if line.strip()
    ]


df = pd.read_csv(
    TEST_DATA_FILE
)


# ============================================================
# CORRECTION CAPABILITY
# ============================================================

Z_LIMIT = 2.5
THETA_LIMIT = 1.2
LOCATOR_LIMIT = 1.0


SEARCH_SPACE = [
    Real(
        -Z_LIMIT,
        Z_LIMIT,
        name="z_adj",
    ),

    Real(
        -THETA_LIMIT,
        THETA_LIMIT,
        name="theta_adj",
    ),

    Real(
        -LOCATOR_LIMIT,
        LOCATOR_LIMIT,
        name="locator_offset",
    ),
]


# ============================================================
# CAPABILITY PENALTY SETTINGS
# ============================================================
#
# Utilization:
#
# U = max(
#       |z| / z_limit,
#       |theta| / theta_limit,
#       |locator| / locator_limit
#     )
#
#
# Behavior:
#
#   U <= 0.70
#       no penalty
#
#   0.70 < U <= 0.90
#       mild penalty
#
#   U > 0.90
#       increasingly strong penalty
#
#
# These values are engineering prototype assumptions.
# They are NOT industrial standards.
# ============================================================

SAFE_UTILIZATION = 0.70

HIGH_UTILIZATION = 0.90

MILD_PENALTY_WEIGHT = 0.10

HIGH_PENALTY_WEIGHT = 0.50


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_utilization(
    z_adj,
    theta_adj,
    locator_offset,
):

    z_util = (
        abs(z_adj)
        / Z_LIMIT
    )

    theta_util = (
        abs(theta_adj)
        / THETA_LIMIT
    )

    locator_util = (
        abs(locator_offset)
        / LOCATOR_LIMIT
    )


    return float(
        max(
            z_util,
            theta_util,
            locator_util,
        )
    )


# ============================================================
# CAPABILITY PENALTY
# ============================================================

def capability_penalty(
    utilization,
):
    """
    Piecewise smooth-ish penalty.

    Moderate corrections remain nearly unpenalized.

    Corrections near physical capability require a stronger
    quality benefit to be selected.
    """

    if utilization <= SAFE_UTILIZATION:

        return 0.0


    if utilization <= HIGH_UTILIZATION:

        normalized = (
            utilization
            -
            SAFE_UTILIZATION
        ) / (
            HIGH_UTILIZATION
            -
            SAFE_UTILIZATION
        )


        return float(
            MILD_PENALTY_WEIGHT
            * normalized ** 2
        )


    normalized_high = (
        utilization
        -
        HIGH_UTILIZATION
    ) / (
        1.0
        -
        HIGH_UTILIZATION
    )


    return float(
        MILD_PENALTY_WEIGHT
        +
        HIGH_PENALTY_WEIGHT
        * normalized_high ** 2
    )


# ============================================================
# BUILD FEATURE VECTOR
# ============================================================

def build_candidate_features(
    base_row,
    z_adj,
    theta_adj,
    locator_offset,
):

    utilization = (
        calculate_utilization(
            z_adj,
            theta_adj,
            locator_offset,
        )
    )


    candidate = {}


    for feature in FEATURE_COLUMNS:

        if feature == "z_adj":

            candidate[
                feature
            ] = z_adj


        elif feature == "theta_adj":

            candidate[
                feature
            ] = theta_adj


        elif feature == "locator_offset":

            candidate[
                feature
            ] = locator_offset


        elif feature == "correction_utilization":

            candidate[
                feature
            ] = utilization


        else:

            candidate[
                feature
            ] = base_row[
                feature
            ]


    X = pd.DataFrame(
        [
            candidate
        ]
    )


    return X[
        FEATURE_COLUMNS
    ]


# ============================================================
# PREDICT QUALITY
# ============================================================

def predict_quality(
    base_row,
    z_adj,
    theta_adj,
    locator_offset,
):

    X = build_candidate_features(
        base_row=base_row,
        z_adj=z_adj,
        theta_adj=theta_adj,
        locator_offset=locator_offset,
    )


    prediction = model.predict(
        X
    )[0]


    return float(
        prediction
    )


# ============================================================
# STANDARD BO
# ============================================================

def recommend_standard_bo(
    base_row,
    seed,
    n_calls=30,
):

    def objective(
        values
    ):

        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = values


        return predict_quality(
            base_row,
            z_adj,
            theta_adj,
            locator_offset,
        )


    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=n_calls,
        n_initial_points=10,
        acq_func="EI",
        random_state=seed,
    )


    z_adj = float(
        result.x[0]
    )

    theta_adj = float(
        result.x[1]
    )

    locator_offset = float(
        result.x[2]
    )


    utilization = (
        calculate_utilization(
            z_adj,
            theta_adj,
            locator_offset,
        )
    )


    return {
        "z_adj":
            z_adj,

        "theta_adj":
            theta_adj,

        "locator_offset":
            locator_offset,

        "predicted_quality":
            float(
                result.fun
            ),

        "utilization":
            utilization,

        "penalty":
            0.0,

        "objective":
            float(
                result.fun
            ),
    }


# ============================================================
# CAPABILITY-AWARE BO
# ============================================================

def recommend_safe_bo(
    base_row,
    seed,
    n_calls=30,
):

    def objective(
        values
    ):

        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = values


        predicted_quality = (
            predict_quality(
                base_row,
                z_adj,
                theta_adj,
                locator_offset,
            )
        )


        utilization = (
            calculate_utilization(
                z_adj,
                theta_adj,
                locator_offset,
            )
        )


        penalty = (
            capability_penalty(
                utilization
            )
        )


        return (
            predicted_quality
            +
            penalty
        )


    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=n_calls,
        n_initial_points=10,
        acq_func="EI",
        random_state=seed,
    )


    z_adj = float(
        result.x[0]
    )

    theta_adj = float(
        result.x[1]
    )

    locator_offset = float(
        result.x[2]
    )


    predicted_quality = (
        predict_quality(
            base_row,
            z_adj,
            theta_adj,
            locator_offset,
        )
    )


    utilization = (
        calculate_utilization(
            z_adj,
            theta_adj,
            locator_offset,
        )
    )


    penalty = (
        capability_penalty(
            utilization
        )
    )


    return {
        "z_adj":
            z_adj,

        "theta_adj":
            theta_adj,

        "locator_offset":
            locator_offset,

        "predicted_quality":
            predicted_quality,

        "utilization":
            utilization,

        "penalty":
            penalty,

        "objective":
            (
                predicted_quality
                +
                penalty
            ),
    }


# ============================================================
# REPRESENTATIVE TEST STATES
# ============================================================
#
# We first run only 10 states.
#
# Two representative states are taken from each component
# stage.
#
# This tests whether the safer objective actually reduces
# aggressive capability use before we run another expensive
# sequential pilot.
# ============================================================

test_rows = []


for component_index in range(
    1,
    6,
):

    subset = df[
        df[
            "component_index"
        ]
        == component_index
    ]


    # Prefer non-zero trajectory states because Fix 2 is
    # specifically aimed at adaptive closed-loop operation.

    adaptive_subset = subset[
        subset[
            "trajectory_type"
        ]
        != "zero"
    ]


    if len(adaptive_subset) >= 2:

        subset = adaptive_subset


    row_a = subset.iloc[
        len(subset)
        // 3
    ]


    row_b = subset.iloc[
        2
        * len(subset)
        // 3
    ]


    test_rows.extend(
        [
            row_a,
            row_b,
        ]
    )


# ============================================================
# TEST
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CAPABILITY-AWARE BO SMOKE TEST"
)

print(
    "============================================================"
)


print(
    f"\nTest states : "
    f"{len(test_rows)}"
)


results = []


for test_case, row in enumerate(
    test_rows,
    start=1,
):


    standard = (
        recommend_standard_bo(
            base_row=row,
            seed=(
                1000
                +
                test_case
            ),
            n_calls=30,
        )
    )


    safe = (
        recommend_safe_bo(
            base_row=row,
            seed=(
                1000
                +
                test_case
            ),
            n_calls=30,
        )
    )


    utilization_reduction = (
        (
            standard[
                "utilization"
            ]
            -
            safe[
                "utilization"
            ]
        )
        * 100.0
    )


    quality_difference = (
        safe[
            "predicted_quality"
        ]
        -
        standard[
            "predicted_quality"
        ]
    )


    print(
        "\n"
        "------------------------------------------------------------"
    )


    print(
        f"TEST CASE {test_case}"
    )


    print(
        f"Component index : "
        f"{int(row['component_index'])}"
    )


    print(
        f"Trajectory      : "
        f"{row['trajectory_type']}"
    )


    print(
        "\nSTANDARD BO"
    )


    print(
        f"Predicted Q     : "
        f"{standard['predicted_quality']:.4f}"
    )


    print(
        f"Utilization     : "
        f"{standard['utilization']:.3f}"
    )


    print(
        "\nCAPABILITY-AWARE BO"
    )


    print(
        f"Predicted Q     : "
        f"{safe['predicted_quality']:.4f}"
    )


    print(
        f"Utilization     : "
        f"{safe['utilization']:.3f}"
    )


    print(
        f"Penalty         : "
        f"{safe['penalty']:.4f}"
    )


    print(
        f"\nUtilization reduction : "
        f"{utilization_reduction:.2f} percentage points"
    )


    print(
        f"Predicted quality cost : "
        f"{quality_difference:.4f}"
    )


    results.append(
        {
            "test_case":
                test_case,

            "component_index":
                int(
                    row[
                        "component_index"
                    ]
                ),

            "trajectory_type":
                row[
                    "trajectory_type"
                ],

            "standard_predicted_quality":
                standard[
                    "predicted_quality"
                ],

            "standard_utilization":
                standard[
                    "utilization"
                ],

            "safe_predicted_quality":
                safe[
                    "predicted_quality"
                ],

            "safe_utilization":
                safe[
                    "utilization"
                ],

            "safe_penalty":
                safe[
                    "penalty"
                ],

            "utilization_reduction_percentage_points":
                utilization_reduction,

            "predicted_quality_cost":
                quality_difference,

            "standard_z_adj":
                standard[
                    "z_adj"
                ],

            "standard_theta_adj":
                standard[
                    "theta_adj"
                ],

            "standard_locator_offset":
                standard[
                    "locator_offset"
                ],

            "safe_z_adj":
                safe[
                    "z_adj"
                ],

            "safe_theta_adj":
                safe[
                    "theta_adj"
                ],

            "safe_locator_offset":
                safe[
                    "locator_offset"
                ],
        }
    )


# ============================================================
# SUMMARY
# ============================================================

result_df = pd.DataFrame(
    results
)


os.makedirs(
    "results/tables",
    exist_ok=True,
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


standard_near_limit = (
    result_df[
        "standard_utilization"
    ]
    >= 0.90
).mean() * 100.0


safe_near_limit = (
    result_df[
        "safe_utilization"
    ]
    >= 0.90
).mean() * 100.0


mean_standard_utilization = (
    result_df[
        "standard_utilization"
    ]
    .mean()
)


mean_safe_utilization = (
    result_df[
        "safe_utilization"
    ]
    .mean()
)


mean_quality_cost = (
    result_df[
        "predicted_quality_cost"
    ]
    .mean()
)


print(
    "\n"
    "============================================================"
)

print(
    "FIX 2 SMOKE-TEST SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nStandard BO mean utilization : "
    f"{mean_standard_utilization:.3f}"
)


print(
    f"Safe BO mean utilization     : "
    f"{mean_safe_utilization:.3f}"
)


print(
    f"\nStandard BO >=90% capability : "
    f"{standard_near_limit:.2f}%"
)


print(
    f"Safe BO >=90% capability     : "
    f"{safe_near_limit:.2f}%"
)


print(
    f"\nMean predicted quality cost  : "
    f"{mean_quality_cost:.4f}"
)


if (
    mean_safe_utilization
    <
    mean_standard_utilization
):

    print(
        "\nPASS - Capability-aware BO reduced "
        "average correction utilization."
    )

else:

    print(
        "\nWARNING - Capability-aware BO did not "
        "reduce average utilization."
    )


if (
    safe_near_limit
    <
    standard_near_limit
):

    print(
        "PASS - Capability-aware BO reduced "
        "near-limit recommendations."
    )

else:

    print(
        "NOTE - Near-limit recommendation rate "
        "did not decrease."
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CAPABILITY-AWARE BO SMOKE TEST COMPLETED"
)

print(
    "============================================================"
)