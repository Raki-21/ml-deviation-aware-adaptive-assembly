import os
import joblib
import numpy as np
import pandas as pd

from skopt import gp_minimize
from skopt.space import Real


# ============================================================
# VERSION 3.2 - BAYESIAN OPTIMIZER
# ============================================================
#
# Purpose:
#
# Use the selected V3.2 ML surrogate to recommend:
#
#   z_adj
#   theta_adj
#   locator_offset
#
# for a given sequential assembly state.
#
#
# IMPORTANT:
#
# Bayesian Optimization minimizes the PREDICTED quality score.
#
# The recommendation is NOT accepted as final proof yet.
#
# In the next stage, every BO recommendation will be applied
# to the actual assembly simulator and independently verified.
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = (
    "models/"
    "v3_2_best_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_surrogate_features.txt"
)

ML_DATA_FILE = (
    "data/processed/"
    "v3_2_ml_training_dataset.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    MODEL_FILE,
    FEATURE_FILE,
    ML_DATA_FILE,
]


missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    print(
        "\nERROR - Missing required files:"
    )

    for file in missing_files:
        print(file)

    raise SystemExit(
        "\nComplete the ML screening stage first."
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# LOAD EXACT FEATURE ORDER
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    ML_DATA_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 BAYESIAN OPTIMIZER"
)

print(
    "============================================================"
)


print(
    f"\nLoaded surrogate model:"
)

print(
    MODEL_FILE
)


print(
    f"\nNumber of model features:"
    f" {len(FEATURE_COLUMNS)}"
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
# BUILD ONE ML INPUT ROW
# ============================================================

def build_candidate_features(
    base_row,
    z_adj,
    theta_adj,
    locator_offset,
):
    """
    Keep the current assembly state, component deviation and
    process condition unchanged.

    Only replace the candidate correction parameters.
    """

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

            candidate[
                feature
            ] = max(
                z_util,
                theta_util,
                locator_util,
            )


        else:

            candidate[
                feature
            ] = base_row[
                feature
            ]


    candidate_df = pd.DataFrame(
        [
            candidate
        ]
    )


    candidate_df = candidate_df[
        FEATURE_COLUMNS
    ]


    return candidate_df


# ============================================================
# PREDICT QUALITY
# ============================================================

def predict_quality(
    base_row,
    z_adj,
    theta_adj,
    locator_offset,
):

    X_candidate = (
        build_candidate_features(
            base_row=base_row,
            z_adj=z_adj,
            theta_adj=theta_adj,
            locator_offset=locator_offset,
        )
    )


    prediction = model.predict(
        X_candidate
    )[0]


    return float(
        prediction
    )


# ============================================================
# BAYESIAN OPTIMIZATION FUNCTION
# ============================================================

def recommend_correction(
    base_row,
    n_calls=30,
    random_state=42,
):
    """
    Run Bayesian Optimization over the three available
    correction parameters.

    Returns the correction with minimum surrogate-predicted
    quality score.
    """

    evaluation_counter = {
        "count": 0
    }


    def objective(
        correction_values
    ):

        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = correction_values


        predicted_quality = (
            predict_quality(
                base_row=base_row,
                z_adj=z_adj,
                theta_adj=theta_adj,
                locator_offset=locator_offset,
            )
        )


        evaluation_counter[
            "count"
        ] += 1


        return predicted_quality


    result = gp_minimize(

        func=objective,

        dimensions=SEARCH_SPACE,

        n_calls=n_calls,

        n_initial_points=10,

        acq_func="EI",

        random_state=random_state,
    )


    best_z = float(
        result.x[0]
    )

    best_theta = float(
        result.x[1]
    )

    best_locator = float(
        result.x[2]
    )

    best_predicted_quality = float(
        result.fun
    )


    z_util = (
        abs(best_z)
        / Z_LIMIT
    )

    theta_util = (
        abs(best_theta)
        / THETA_LIMIT
    )

    locator_util = (
        abs(best_locator)
        / LOCATOR_LIMIT
    )


    max_utilization = max(
        z_util,
        theta_util,
        locator_util,
    )


    return {

        "z_adj":
            best_z,

        "theta_adj":
            best_theta,

        "locator_offset":
            best_locator,

        "predicted_quality":
            best_predicted_quality,

        "correction_utilization":
            max_utilization,

        "n_evaluations":
            evaluation_counter[
                "count"
            ],
    }


# ============================================================
# SMOKE TEST
# ============================================================
#
# First test only five representative component states.
#
# DO NOT run all 5000 sequential decisions yet.
# ============================================================


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "BAYESIAN OPTIMIZATION SMOKE TEST"
)

print(
    "------------------------------------------------------------"
)


# One representative component state from each assembly stage
test_rows = []


for component_index in range(
    1,
    6,
):

    candidates = df[
        df[
            "component_index"
        ]
        == component_index
    ]


    test_rows.append(
        candidates.iloc[
            len(candidates)
            // 2
        ]
    )


test_results = []


for test_number, row in enumerate(
    test_rows,
    start=1,
):


    # --------------------------------------------------------
    # QUALITY WITH ZERO CORRECTION
    # --------------------------------------------------------

    zero_quality = (
        predict_quality(
            base_row=row,
            z_adj=0.0,
            theta_adj=0.0,
            locator_offset=0.0,
        )
    )


    # --------------------------------------------------------
    # BAYESIAN OPTIMIZATION
    # --------------------------------------------------------

    recommendation = (
        recommend_correction(
            base_row=row,
            n_calls=30,
            random_state=(
                100
                + test_number
            ),
        )
    )


    improvement = (
        (
            zero_quality
            -
            recommendation[
                "predicted_quality"
            ]
        )
        /
        max(
            abs(
                zero_quality
            ),
            1e-9,
        )
        * 100.0
    )


    print(
        f"\nTEST CASE {test_number}"
    )


    print(
        f"Component index      : "
        f"{int(row['component_index'])}"
    )


    print(
        f"Zero-correction Q    : "
        f"{zero_quality:.4f}"
    )


    print(
        f"BO predicted Q       : "
        f"{recommendation['predicted_quality']:.4f}"
    )


    print(
        f"Predicted improvement: "
        f"{improvement:.2f}%"
    )


    print(
        f"z_adj                : "
        f"{recommendation['z_adj']:.4f} mm"
    )


    print(
        f"theta_adj            : "
        f"{recommendation['theta_adj']:.5f} deg"
    )


    print(
        f"locator_offset       : "
        f"{recommendation['locator_offset']:.4f} mm"
    )


    print(
        f"Capability utilization: "
        f"{recommendation['correction_utilization']:.3f}"
    )


    print(
        f"BO evaluations       : "
        f"{recommendation['n_evaluations']}"
    )


    test_results.append(
        {
            "test_case":
                test_number,

            "assembly_id":
                int(
                    row[
                        "assembly_id"
                    ]
                ),

            "component_index":
                int(
                    row[
                        "component_index"
                    ]
                ),

            "zero_correction_quality":
                zero_quality,

            "bo_predicted_quality":
                recommendation[
                    "predicted_quality"
                ],

            "predicted_improvement_percent":
                improvement,

            "z_adj":
                recommendation[
                    "z_adj"
                ],

            "theta_adj":
                recommendation[
                    "theta_adj"
                ],

            "locator_offset":
                recommendation[
                    "locator_offset"
                ],

            "correction_utilization":
                recommendation[
                    "correction_utilization"
                ],
        }
    )


# ============================================================
# SAVE SMOKE-TEST RESULTS
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True,
)


test_result_df = pd.DataFrame(
    test_results
)


OUTPUT_FILE = (
    "results/tables/"
    "v3_2_bo_smoke_test.csv"
)


test_result_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "BAYESIAN OPTIMIZATION SMOKE TEST COMPLETED"
)

print(
    "============================================================"
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)