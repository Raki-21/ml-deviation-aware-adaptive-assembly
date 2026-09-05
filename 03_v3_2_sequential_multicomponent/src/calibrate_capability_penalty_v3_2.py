import os
import joblib
import numpy as np
import pandas as pd

from skopt import gp_minimize
from skopt.space import Real


# ============================================================
# V3.2
# CAPABILITY PENALTY CALIBRATION
# ============================================================
#
# PURPOSE
#
# The first capability-aware BO smoke test showed:
#
#   - only a very small reduction in mean utilization
#   - no reduction in >=90% capability recommendations
#   - noticeable predicted-quality cost
#
# Therefore, instead of arbitrarily strengthening one penalty,
# this script compares several candidate penalty formulations
# on the SAME representative states.
#
# This is a calibration experiment only.
#
# The final selected penalty must still be verified using the
# actual sequential assembly simulator afterwards.
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

DATA_FILE = (
    "data/processed/"
    "v3_2_closed_loop_ml_training_dataset.csv"
)

DETAIL_OUTPUT = (
    "results/tables/"
    "v3_2_capability_penalty_calibration_details.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_capability_penalty_calibration_summary.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    MODEL_FILE,
    FEATURE_FILE,
    DATA_FILE,
]

missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]

if missing_files:

    print(
        "\nERROR - Missing required files:"
    )

    for path in missing_files:
        print(path)

    raise SystemExit(
        "\nRequired Fix 1 files are not available."
    )


# ============================================================
# LOAD MODEL + DATA
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CAPABILITY PENALTY CALIBRATION"
)

print(
    "============================================================"
)


print(
    "\nLoading closed-loop RF..."
)

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


print(
    "Loading closed-loop training states..."
)

df = pd.read_csv(
    DATA_FILE
)


# ============================================================
# ORIGINAL CORRECTION LIMITS
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
# OPTIMIZER SETTINGS
# ============================================================

N_CALLS = 30

N_INITIAL_POINTS = 10


# ============================================================
# PENALTY CONFIGURATIONS
# ============================================================
#
# We test four capability-aware alternatives.
#
# P0:
#   Standard BO.
#
# P1:
#   Original mild threshold penalty.
#
# P2:
#   Continuous correction-effort penalty +
#   stronger near-boundary penalty.
#
# P3:
#   Slightly stronger version of P2.
#
# P4:
#   Conservative near-boundary barrier.
#
#
# We do NOT automatically choose the strongest penalty.
#
# The goal is the best trade-off between:
#
#   predicted quality
#   correction utilization
#   near-boundary frequency
#
# ============================================================

PENALTY_CONFIGS = {

    "P0_STANDARD": {
        "effort_weight": 0.00,
        "barrier_start": 1.00,
        "barrier_weight": 0.00,
    },

    "P1_MILD": {
        "effort_weight": 0.03,
        "barrier_start": 0.85,
        "barrier_weight": 0.20,
    },

    "P2_BALANCED": {
        "effort_weight": 0.05,
        "barrier_start": 0.85,
        "barrier_weight": 0.50,
    },

    "P3_STRONG": {
        "effort_weight": 0.08,
        "barrier_start": 0.80,
        "barrier_weight": 0.75,
    },

    "P4_BARRIER": {
        "effort_weight": 0.04,
        "barrier_start": 0.80,
        "barrier_weight": 1.25,
    },
}


# ============================================================
# UTILIZATION
# ============================================================

def calculate_normalized_corrections(
    z_adj,
    theta_adj,
    locator_offset,
):

    z_norm = (
        abs(z_adj)
        / Z_LIMIT
    )

    theta_norm = (
        abs(theta_adj)
        / THETA_LIMIT
    )

    locator_norm = (
        abs(locator_offset)
        / LOCATOR_LIMIT
    )

    return (
        float(z_norm),
        float(theta_norm),
        float(locator_norm),
    )


def calculate_utilization(
    z_adj,
    theta_adj,
    locator_offset,
):

    (
        z_norm,
        theta_norm,
        locator_norm,
    ) = calculate_normalized_corrections(
        z_adj,
        theta_adj,
        locator_offset,
    )

    return float(
        max(
            z_norm,
            theta_norm,
            locator_norm,
        )
    )


def calculate_effort(
    z_adj,
    theta_adj,
    locator_offset,
):
    """
    RMS normalized correction effort.

    This represents how strongly the three available
    correction degrees of freedom are being used overall.

    It is deliberately separate from max utilization.
    """

    (
        z_norm,
        theta_norm,
        locator_norm,
    ) = calculate_normalized_corrections(
        z_adj,
        theta_adj,
        locator_offset,
    )

    effort = np.sqrt(
        (
            z_norm ** 2
            +
            theta_norm ** 2
            +
            locator_norm ** 2
        )
        / 3.0
    )

    return float(
        effort
    )


# ============================================================
# CAPABILITY PENALTY
# ============================================================

def calculate_penalty(
    z_adj,
    theta_adj,
    locator_offset,
    config,
):

    utilization = calculate_utilization(
        z_adj,
        theta_adj,
        locator_offset,
    )

    effort = calculate_effort(
        z_adj,
        theta_adj,
        locator_offset,
    )


    # --------------------------------------------------------
    # 1. Small continuous effort penalty
    # --------------------------------------------------------

    effort_penalty = (
        config[
            "effort_weight"
        ]
        *
        effort ** 2
    )


    # --------------------------------------------------------
    # 2. Nonlinear near-boundary barrier
    # --------------------------------------------------------

    barrier_start = config[
        "barrier_start"
    ]

    barrier_weight = config[
        "barrier_weight"
    ]


    if (
        utilization
        <=
        barrier_start
    ):

        boundary_penalty = 0.0

    else:

        normalized_excess = (
            utilization
            -
            barrier_start
        ) / (
            1.0
            -
            barrier_start
        )

        boundary_penalty = (
            barrier_weight
            *
            normalized_excess ** 3
        )


    total_penalty = (
        effort_penalty
        +
        boundary_penalty
    )


    return float(
        total_penalty
    )


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def build_candidate_features(
    base_row,
    z_adj,
    theta_adj,
    locator_offset,
):

    utilization = calculate_utilization(
        z_adj,
        theta_adj,
        locator_offset,
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


    candidate_df = pd.DataFrame(
        [
            candidate
        ]
    )


    return candidate_df[
        FEATURE_COLUMNS
    ]


# ============================================================
# QUALITY PREDICTION
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
# OPTIMIZE ONE STATE
# ============================================================

def optimize_state(
    base_row,
    config_name,
    config,
    seed,
):

    def objective(
        values
    ):

        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = values


        predicted_quality = predict_quality(
            base_row,
            z_adj,
            theta_adj,
            locator_offset,
        )


        penalty = calculate_penalty(
            z_adj,
            theta_adj,
            locator_offset,
            config,
        )


        return (
            predicted_quality
            +
            penalty
        )


    result = gp_minimize(

        func=objective,

        dimensions=SEARCH_SPACE,

        n_calls=N_CALLS,

        n_initial_points=N_INITIAL_POINTS,

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


    predicted_quality = predict_quality(
        base_row,
        z_adj,
        theta_adj,
        locator_offset,
    )


    utilization = calculate_utilization(
        z_adj,
        theta_adj,
        locator_offset,
    )


    effort = calculate_effort(
        z_adj,
        theta_adj,
        locator_offset,
    )


    penalty = calculate_penalty(
        z_adj,
        theta_adj,
        locator_offset,
        config,
    )


    return {

        "config":
            config_name,

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

        "effort":
            effort,

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
# BUILD BALANCED TEST STATES
# ============================================================
#
# 30 states total:
#
#   5 component stages
#   x
#   6 states per stage
#
# We intentionally sample across trajectory types.
#
# ============================================================

RNG = np.random.default_rng(
    42
)


test_rows = []


for component_index in range(
    1,
    6,
):

    stage_df = df[
        df[
            "component_index"
        ]
        ==
        component_index
    ].copy()


    available_trajectories = [
        trajectory
        for trajectory in [
            "heuristic",
            "partial",
            "random_feasible",
            "zero",
        ]
        if trajectory
        in stage_df[
            "trajectory_type"
        ].unique()
    ]


    selected_stage_rows = []


    # Try to obtain diversity across trajectory classes first.

    for trajectory in available_trajectories:

        subset = stage_df[
            stage_df[
                "trajectory_type"
            ]
            ==
            trajectory
        ]


        if len(subset) > 0:

            random_index = int(
                RNG.integers(
                    0,
                    len(subset),
                )
            )

            selected_stage_rows.append(
                subset.iloc[
                    random_index
                ]
            )


    # Fill remaining slots to reach 6 states per component.

    while len(
        selected_stage_rows
    ) < 6:

        random_index = int(
            RNG.integers(
                0,
                len(stage_df),
            )
        )

        selected_stage_rows.append(
            stage_df.iloc[
                random_index
            ]
        )


    selected_stage_rows = (
        selected_stage_rows[
            :6
        ]
    )


    test_rows.extend(
        selected_stage_rows
    )


print(
    f"\nCalibration states : "
    f"{len(test_rows)}"
)

print(
    f"Penalty profiles   : "
    f"{len(PENALTY_CONFIGS)}"
)

print(
    f"BO calls/profile   : "
    f"{N_CALLS}"
)


# ============================================================
# CALIBRATION EXPERIMENT
# ============================================================

detail_results = []


for state_number, row in enumerate(
    test_rows,
    start=1,
):


    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        f"STATE "
        f"{state_number:02d}"
        f"/"
        f"{len(test_rows)}"
        f" | component "
        f"{int(row['component_index'])}"
        f" | trajectory "
        f"{row['trajectory_type']}"
    )


    # IMPORTANT:
    # same seed for all penalty configurations on one state
    # -> fairer BO comparison

    state_seed = (
        5000
        +
        state_number
    )


    state_outputs = {}


    for (
        config_name,
        config,
    ) in PENALTY_CONFIGS.items():


        output = optimize_state(
            base_row=row,
            config_name=config_name,
            config=config,
            seed=state_seed,
        )


        state_outputs[
            config_name
        ] = output


        print(
            f"{config_name:14s}"
            f" | Q="
            f"{output['predicted_quality']:.4f}"
            f" | U="
            f"{output['utilization']:.3f}"
            f" | effort="
            f"{output['effort']:.3f}"
        )


    standard = state_outputs[
        "P0_STANDARD"
    ]


    for (
        config_name,
        output,
    ) in state_outputs.items():


        quality_cost = (
            output[
                "predicted_quality"
            ]
            -
            standard[
                "predicted_quality"
            ]
        )


        utilization_reduction = (
            standard[
                "utilization"
            ]
            -
            output[
                "utilization"
            ]
        )


        detail_results.append(
            {

                "state_number":
                    state_number,

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

                "config":
                    config_name,

                "predicted_quality":
                    output[
                        "predicted_quality"
                    ],

                "utilization":
                    output[
                        "utilization"
                    ],

                "effort":
                    output[
                        "effort"
                    ],

                "penalty":
                    output[
                        "penalty"
                    ],

                "objective":
                    output[
                        "objective"
                    ],

                "quality_cost_vs_standard":
                    quality_cost,

                "utilization_reduction_vs_standard":
                    utilization_reduction,

                "z_adj":
                    output[
                        "z_adj"
                    ],

                "theta_adj":
                    output[
                        "theta_adj"
                    ],

                "locator_offset":
                    output[
                        "locator_offset"
                    ],
            }
        )


# ============================================================
# SAVE DETAIL TABLE
# ============================================================

detail_df = pd.DataFrame(
    detail_results
)


os.makedirs(
    "results/tables",
    exist_ok=True,
)


detail_df.to_csv(
    DETAIL_OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []


for config_name in PENALTY_CONFIGS.keys():


    subset = detail_df[
        detail_df[
            "config"
        ]
        ==
        config_name
    ]


    mean_quality = (
        subset[
            "predicted_quality"
        ]
        .mean()
    )


    median_quality = (
        subset[
            "predicted_quality"
        ]
        .median()
    )


    mean_utilization = (
        subset[
            "utilization"
        ]
        .mean()
    )


    median_utilization = (
        subset[
            "utilization"
        ]
        .median()
    )


    p90_utilization = (
        subset[
            "utilization"
        ]
        .quantile(
            0.90
        )
    )


    near_limit_rate = (
        subset[
            "utilization"
        ]
        .ge(
            0.90
        )
        .mean()
        *
        100.0
    )


    full_limit_rate = (
        subset[
            "utilization"
        ]
        .ge(
            0.99
        )
        .mean()
        *
        100.0
    )


    mean_effort = (
        subset[
            "effort"
        ]
        .mean()
    )


    mean_quality_cost = (
        subset[
            "quality_cost_vs_standard"
        ]
        .mean()
    )


    mean_utilization_reduction = (
        subset[
            "utilization_reduction_vs_standard"
        ]
        .mean()
    )


    summary_rows.append(
        {

            "config":
                config_name,

            "mean_predicted_quality":
                mean_quality,

            "median_predicted_quality":
                median_quality,

            "mean_utilization":
                mean_utilization,

            "median_utilization":
                median_utilization,

            "p90_utilization":
                p90_utilization,

            "near_limit_rate_ge_90_percent":
                near_limit_rate,

            "full_limit_rate_ge_99_percent":
                full_limit_rate,

            "mean_effort":
                mean_effort,

            "mean_quality_cost_vs_standard":
                mean_quality_cost,

            "mean_utilization_reduction_vs_standard":
                mean_utilization_reduction,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "CAPABILITY PENALTY CALIBRATION SUMMARY"
)

print(
    "============================================================"
)


display_columns = [

    "config",

    "mean_predicted_quality",

    "mean_utilization",

    "p90_utilization",

    "near_limit_rate_ge_90_percent",

    "full_limit_rate_ge_99_percent",

    "mean_quality_cost_vs_standard",
]


print(
    "\n"
    +
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# PRACTICAL SCREENING
# ============================================================
#
# We intentionally do NOT declare one configuration the final
# optimizer solely from predicted-quality data.
#
# We first screen for candidates that:
#
# 1. reduce mean utilization,
# 2. reduce or preserve near-limit frequency,
# 3. do not impose excessive predicted-quality degradation.
#
# The final decision comes from actual simulator validation.
# ============================================================

standard_row = summary_df[
    summary_df[
        "config"
    ]
    ==
    "P0_STANDARD"
].iloc[0]


candidate_df = summary_df[
    summary_df[
        "config"
    ]
    !=
    "P0_STANDARD"
].copy()


candidate_df[
    "reduced_mean_utilization"
] = (
    candidate_df[
        "mean_utilization"
    ]
    <
    standard_row[
        "mean_utilization"
    ]
)


candidate_df[
    "reduced_near_limit_rate"
] = (
    candidate_df[
        "near_limit_rate_ge_90_percent"
    ]
    <
    standard_row[
        "near_limit_rate_ge_90_percent"
    ]
)


candidate_df[
    "quality_cost_reasonable"
] = (
    candidate_df[
        "mean_quality_cost_vs_standard"
    ]
    <=
    0.10
)


candidate_df[
    "screening_pass"
] = (

    candidate_df[
        "reduced_mean_utilization"
    ]

    &

    candidate_df[
        "reduced_near_limit_rate"
    ]

    &

    candidate_df[
        "quality_cost_reasonable"
    ]
)


print(
    "\n"
    "============================================================"
)

print(
    "SCREENING RESULT"
)

print(
    "============================================================"
)


passing = candidate_df[
    candidate_df[
        "screening_pass"
    ]
]


if len(
    passing
) > 0:

    print(
        "\nCandidate profiles passed the initial "
        "surrogate-based screening:"
    )

    for config_name in passing[
        "config"
    ]:

        print(
            f"  - {config_name}"
        )


    # Rank passing candidates.
    #
    # Primary:
    # lowest near-limit frequency
    #
    # Secondary:
    # lower utilization
    #
    # Tertiary:
    # lower quality cost

    ranked = passing.sort_values(

        by=[
            "near_limit_rate_ge_90_percent",
            "mean_utilization",
            "mean_quality_cost_vs_standard",
        ],

        ascending=[
            True,
            True,
            True,
        ],
    )


    provisional_best = ranked.iloc[
        0
    ][
        "config"
    ]


    print(
        "\nPROVISIONAL BEST PROFILE:"
    )

    print(
        provisional_best
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "This is NOT yet the final optimizer."
    )

    print(
        "It must next be tested in the actual sequential "
        "assembly simulator."
    )


else:

    print(
        "\nNO PROFILE PASSED ALL SCREENING CRITERIA."
    )

    print(
        "Do not run the expensive sequential pilot yet."
    )

    print(
        "The capability objective requires another redesign."
    )


# ============================================================
# SAVE
# ============================================================

print(
    "\nSaved:"
)

print(
    DETAIL_OUTPUT
)

print(
    SUMMARY_OUTPUT
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CAPABILITY PENALTY CALIBRATION COMPLETED"
)

print(
    "============================================================"
)