import os
import time
import joblib
import numpy as np
import pandas as pd

from sequential_assembly import (
    s,
    PROFILE_LENGTH_MM,
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
    correction_profile,
    update_assembly_state,
    calculate_quality_metrics,
)


# ============================================================
# VERSION 3.2
# EQUAL-BUDGET SEARCH ABLATION
# ============================================================
#
# RESEARCH QUESTION
#
# Does Bayesian refinement actually add value compared with
# a purely structured RF search when BOTH methods receive
# the same surrogate-evaluation budget?
#
#
# METHOD A:
#
#   STRUCTURED RF SEARCH
#
#   20 physically meaningful RF-evaluated candidates:
#
#       1 zero
#       1 informed
#       8 local-informed
#       6 moderate-random
#       4 broad-random
#
#       TOTAL = 20
#
#
# METHOD B:
#
#   EXISTING WARM-START BO
#
#       8 structured initial candidates
#       12 Bayesian refinement evaluations
#
#       TOTAL = 20
#
#
# The existing 100-assembly warm-start BO results are reused.
#
# Therefore this script only runs Method A and compares it
# directly against the saved warm-start BO results.
#
#
# IMPORTANT
#
# This is an ablation study.
#
# It determines whether Bayesian refinement provides useful
# additional value beyond physically structured RF search.
# ============================================================


# ============================================================
# PATHS
# ============================================================

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

MODEL_FILE = (
    "models/"
    "v3_2_profile_aware_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_profile_aware_surrogate_features.txt"
)

WARMSTART_ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_warmstart_bo_100_assembly_results.csv"
)

ORIGINAL_PILOT_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_assembly_results.csv"
)


COMPONENT_OUTPUT = (
    "data/processed/"
    "v3_2_equal_budget_structured20_component_results.csv"
)

ASSEMBLY_OUTPUT = (
    "data/processed/"
    "v3_2_equal_budget_structured20_assembly_results.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_equal_budget_search_ablation_summary.csv"
)

STAGE_OUTPUT = (
    "results/tables/"
    "v3_2_equal_budget_structured20_by_stage.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    MODEL_FILE,
    FEATURE_FILE,
    WARMSTART_ASSEMBLY_FILE,
    ORIGINAL_PILOT_FILE,
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
        "\nComplete previous V3.2 experiments first."
    )


# ============================================================
# SETTINGS
# ============================================================

TOTAL_EVALUATIONS = 20

N_COMPONENTS = 5

RANDOM_SEED = 20260825


# ============================================================
# CORRECTION LIMITS
# ============================================================

Z_LIMIT = 2.5

THETA_LIMIT = 1.2

LOCATOR_LIMIT = 1.0


# ============================================================
# PROFILE FEATURES
# ============================================================

PROFILE_SAMPLE_INDICES = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
]


PROFILE_SAMPLE_NAMES = [
    "state_profile_p00",
    "state_profile_p10",
    "state_profile_p20",
    "state_profile_p30",
    "state_profile_p40",
    "state_profile_p50",
    "state_profile_p60",
    "state_profile_p70",
    "state_profile_p80",
    "state_profile_p90",
    "state_profile_p100",
]


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 EQUAL-BUDGET SEARCH ABLATION"
)

print(
    "============================================================"
)


component_df = pd.read_csv(
    COMPONENT_FILE
)


warmstart_df = pd.read_csv(
    WARMSTART_ASSEMBLY_FILE
)


original_df = pd.read_csv(
    ORIGINAL_PILOT_FILE
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


selected_assemblies = sorted(
    warmstart_df[
        "assembly_id"
    ]
    .unique()
    .tolist()
)


print(
    f"\nAssemblies                  : "
    f"{len(selected_assemblies)}"
)

print(
    f"Components / assembly       : "
    f"{N_COMPONENTS}"
)

print(
    f"Structured evaluations      : "
    f"{TOTAL_EVALUATIONS}"
)

print(
    "Warm-start BO evaluations   : 20"
)

print(
    "\nBoth search strategies therefore receive "
    "the same surrogate-evaluation budget."
)


# ============================================================
# COMPONENT PROFILE
# ============================================================

def reconstruct_component_profile(
    row
):

    profile = np.zeros_like(
        s
    )


    if row["offset_mm"] != 0.0:

        profile += deviation_offset(
            offset_mm=row[
                "offset_mm"
            ]
        )


    if row["tilt_deg"] != 0.0:

        profile += deviation_tilt(
            angle_deg=row[
                "tilt_deg"
            ]
        )


    if row["bend_mm"] != 0.0:

        profile += deviation_bend(
            amplitude_mm=row[
                "bend_mm"
            ]
        )


    if row["waviness_mm"] != 0.0:

        profile += deviation_waviness(
            amplitude_mm=row[
                "waviness_mm"
            ],
            waves=3,
        )


    if row["twist_mm"] != 0.0:

        profile += deviation_twist(
            amplitude_mm=row[
                "twist_mm"
            ]
        )


    if row["local_bump_mm"] != 0.0:

        profile += deviation_local_bump(
            amplitude_mm=row[
                "local_bump_mm"
            ],
            sigma=0.12,
        )


    return profile


# ============================================================
# BATCH DISTURBANCE
# ============================================================

def reconstruct_batch_disturbance(
    row
):

    offset_profile = np.full_like(
        s,
        row[
            "batch_offset_bias_mm"
        ],
    )


    angular_profile = (
        np.tan(
            np.deg2rad(
                row[
                    "batch_angular_bias_deg"
                ]
            )
        )
        *
        PROFILE_LENGTH_MM
        *
        (
            s
            -
            0.5
        )
    )


    fixture_sigma = 0.18


    fixture_profile = (
        row[
            "fixture_drift_mm"
        ]
        *
        np.exp(
            -(
                (
                    s
                    -
                    0.5
                )
                ** 2
            )
            /
            (
                2.0
                *
                fixture_sigma ** 2
            )
        )
    )


    return (
        offset_profile
        +
        angular_profile
        +
        fixture_profile
    )


# ============================================================
# STATE FEATURES
# ============================================================

def calculate_state_features(
    state
):

    metrics = calculate_quality_metrics(
        state
    )


    signed_mean = float(
        np.mean(
            state
        )
    )


    signed_end_difference = float(
        state[-1]
        -
        state[0]
    )


    estimated_angle_deg = float(
        np.rad2deg(
            np.arctan(
                signed_end_difference
                /
                PROFILE_LENGTH_MM
            )
        )
    )


    features = {

        "state_mean_gap":
            metrics[
                "mean_gap"
            ],

        "state_max_gap":
            metrics[
                "max_gap"
            ],

        "state_parallelism":
            metrics[
                "parallelism_error"
            ],

        "state_rms":
            metrics[
                "rms_deviation"
            ],

        "state_quality":
            metrics[
                "quality_score"
            ],

        "state_signed_mean":
            signed_mean,

        "state_signed_end_difference":
            signed_end_difference,

        "state_estimated_angle_deg":
            estimated_angle_deg,
    }


    for (
        name,
        index,
    ) in zip(
        PROFILE_SAMPLE_NAMES,
        PROFILE_SAMPLE_INDICES,
    ):

        features[
            name
        ] = float(
            state[
                index
            ]
        )


    return features


# ============================================================
# CORRECTION UTILIZATION
# ============================================================

def calculate_utilization(
    z_adj,
    theta_adj,
    locator_offset,
):

    return float(
        max(
            abs(
                z_adj
            )
            /
            Z_LIMIT,

            abs(
                theta_adj
            )
            /
            THETA_LIMIT,

            abs(
                locator_offset
            )
            /
            LOCATOR_LIMIT,
        )
    )


# ============================================================
# CLIP
# ============================================================

def clip_correction(
    z_adj,
    theta_adj,
    locator_offset,
):

    return (

        float(
            np.clip(
                z_adj,
                -Z_LIMIT,
                Z_LIMIT,
            )
        ),

        float(
            np.clip(
                theta_adj,
                -THETA_LIMIT,
                THETA_LIMIT,
            )
        ),

        float(
            np.clip(
                locator_offset,
                -LOCATOR_LIMIT,
                LOCATOR_LIMIT,
            )
        ),
    )


# ============================================================
# INFORMED CORRECTION
# ============================================================

def informed_correction(
    state,
    row,
):

    signed_mean = float(
        np.mean(
            state
        )
    )


    signed_difference = float(
        state[-1]
        -
        state[0]
    )


    state_angle = float(
        np.rad2deg(
            np.arctan(
                signed_difference
                /
                PROFILE_LENGTH_MM
            )
        )
    )


    z_adj = -(
        row[
            "offset_mm"
        ]
        +
        row[
            "batch_offset_bias_mm"
        ]
        +
        0.50
        *
        signed_mean
    )


    theta_adj = -(
        row[
            "tilt_deg"
        ]
        +
        row[
            "batch_angular_bias_deg"
        ]
        +
        0.50
        *
        state_angle
    )


    locator_offset = -(
        row[
            "local_bump_mm"
        ]
        +
        row[
            "fixture_drift_mm"
        ]
    )


    return clip_correction(
        z_adj,
        theta_adj,
        locator_offset,
    )


# ============================================================
# GENERATE EXACTLY 20 STRUCTURED CANDIDATES
# ============================================================

def generate_structured_candidates(
    state,
    row,
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    candidates = []


    # --------------------------------------------------------
    # 1. ZERO
    # --------------------------------------------------------

    candidates.append(
        {
            "z_adj": 0.0,
            "theta_adj": 0.0,
            "locator_offset": 0.0,
            "source": "zero",
        }
    )


    # --------------------------------------------------------
    # 2. INFORMED
    # --------------------------------------------------------

    (
        informed_z,
        informed_theta,
        informed_locator,
    ) = informed_correction(
        state,
        row,
    )


    candidates.append(
        {
            "z_adj":
                informed_z,

            "theta_adj":
                informed_theta,

            "locator_offset":
                informed_locator,

            "source":
                "informed",
        }
    )


    # --------------------------------------------------------
    # 3. LOCAL INFORMED - 8
    # --------------------------------------------------------

    for _ in range(
        8
    ):


        (
            z_adj,
            theta_adj,
            locator_offset,
        ) = clip_correction(

            informed_z
            +
            rng.normal(
                0.0,
                0.30,
            ),

            informed_theta
            +
            rng.normal(
                0.0,
                0.12,
            ),

            informed_locator
            +
            rng.normal(
                0.0,
                0.15,
            ),
        )


        candidates.append(
            {
                "z_adj":
                    z_adj,

                "theta_adj":
                    theta_adj,

                "locator_offset":
                    locator_offset,

                "source":
                    "local_informed",
            }
        )


    # --------------------------------------------------------
    # 4. MODERATE RANDOM - 6
    # --------------------------------------------------------

    for _ in range(
        6
    ):


        candidates.append(
            {
                "z_adj":
                    float(
                        rng.uniform(
                            -0.60
                            *
                            Z_LIMIT,

                            0.60
                            *
                            Z_LIMIT,
                        )
                    ),

                "theta_adj":
                    float(
                        rng.uniform(
                            -0.60
                            *
                            THETA_LIMIT,

                            0.60
                            *
                            THETA_LIMIT,
                        )
                    ),

                "locator_offset":
                    float(
                        rng.uniform(
                            -0.60
                            *
                            LOCATOR_LIMIT,

                            0.60
                            *
                            LOCATOR_LIMIT,
                        )
                    ),

                "source":
                    "moderate_random",
            }
        )


    # --------------------------------------------------------
    # 5. BROAD RANDOM - 4
    # --------------------------------------------------------

    for _ in range(
        4
    ):


        candidates.append(
            {
                "z_adj":
                    float(
                        rng.uniform(
                            -Z_LIMIT,
                            Z_LIMIT,
                        )
                    ),

                "theta_adj":
                    float(
                        rng.uniform(
                            -THETA_LIMIT,
                            THETA_LIMIT,
                        )
                    ),

                "locator_offset":
                    float(
                        rng.uniform(
                            -LOCATOR_LIMIT,
                            LOCATOR_LIMIT,
                        )
                    ),

                "source":
                    "broad_random",
            }
        )


    if len(
        candidates
    ) != TOTAL_EVALUATIONS:

        raise RuntimeError(
            f"Expected {TOTAL_EVALUATIONS} candidates, "
            f"found {len(candidates)}."
        )


    return candidates


# ============================================================
# MODEL INPUT
# ============================================================

def build_model_input(
    row,
    state,
    candidates,
):

    state_features = calculate_state_features(
        state
    )


    records = []


    for candidate in candidates:


        z_adj = candidate[
            "z_adj"
        ]

        theta_adj = candidate[
            "theta_adj"
        ]

        locator_offset = candidate[
            "locator_offset"
        ]


        record = {

            **state_features,


            "offset_mm":
                row[
                    "offset_mm"
                ],

            "tilt_deg":
                row[
                    "tilt_deg"
                ],

            "bend_mm":
                row[
                    "bend_mm"
                ],

            "waviness_mm":
                row[
                    "waviness_mm"
                ],

            "twist_mm":
                row[
                    "twist_mm"
                ],

            "local_bump_mm":
                row[
                    "local_bump_mm"
                ],

            "component_profile_rms_mm":
                row[
                    "component_profile_rms_mm"
                ],

            "component_parallelism_mm":
                row[
                    "component_parallelism_mm"
                ],

            "n_active_modes":
                row[
                    "n_active_modes"
                ],


            "batch_offset_bias_mm":
                row[
                    "batch_offset_bias_mm"
                ],

            "batch_angular_bias_deg":
                row[
                    "batch_angular_bias_deg"
                ],

            "fixture_drift_mm":
                row[
                    "fixture_drift_mm"
                ],

            "variation_multiplier":
                row[
                    "variation_multiplier"
                ],


            "z_adj":
                z_adj,

            "theta_adj":
                theta_adj,

            "locator_offset":
                locator_offset,

            "correction_utilization":
                calculate_utilization(
                    z_adj,
                    theta_adj,
                    locator_offset,
                ),

            "component_index":
                row[
                    "component_index"
                ],
        }


        records.append(
            record
        )


    X = pd.DataFrame(
        records
    )


    return X[
        FEATURE_COLUMNS
    ]


# ============================================================
# ACTUAL SIMULATOR
# ============================================================

def simulate_candidate(
    state,
    component_profile,
    batch_disturbance,
    candidate,
):

    correction = correction_profile(

        z_adj_mm=(
            candidate[
                "z_adj"
            ]
        ),

        theta_adj_deg=(
            candidate[
                "theta_adj"
            ]
        ),

        locator_offset_mm=(
            candidate[
                "locator_offset"
            ]
        ),
    )


    new_state = update_assembly_state(

        previous_state=state,

        component_deviation=(
            component_profile
        ),

        fixture_drift=(
            batch_disturbance
        ),

        correction=correction,
    )


    metrics = calculate_quality_metrics(
        new_state
    )


    return (
        new_state,
        metrics,
    )


# ============================================================
# RUN
# ============================================================

component_records = []

assembly_records = []


start_time = time.time()


for (
    assembly_counter,
    assembly_id,
) in enumerate(
    selected_assemblies,
    start=1,
):


    assembly_rows = (
        component_df[
            component_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .sort_values(
            "component_index"
        )
    )


    state = create_initial_state()


    warmstart_row = (
        warmstart_df[
            warmstart_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .iloc[
            0
        ]
    )


    original_row = (
        original_df[
            original_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .iloc[
            0
        ]
    )


    batch_condition = (
        assembly_rows.iloc[
            0
        ][
            "batch_condition"
        ]
    )


    for _, row in assembly_rows.iterrows():


        component_index = int(
            row[
                "component_index"
            ]
        )


        component_profile = (
            reconstruct_component_profile(
                row
            )
        )


        batch_disturbance = (
            reconstruct_batch_disturbance(
                row
            )
        )


        seed = (
            RANDOM_SEED
            +
            int(
                assembly_id
            )
            *
            10
            +
            component_index
        )


        candidates = (
            generate_structured_candidates(
                state=state,
                row=row,
                seed=seed,
            )
        )


        X = build_model_input(
            row=row,
            state=state,
            candidates=candidates,
        )


        predictions = model.predict(
            X
        )


        selected_index = int(
            np.argmin(
                predictions
            )
        )


        selected_candidate = candidates[
            selected_index
        ]


        # ----------------------------------------------------
        # Verify all 20 candidates with simulator
        # ----------------------------------------------------

        actual_qualities = []

        actual_states = []

        actual_metrics = []


        for candidate in candidates:


            (
                candidate_state,
                candidate_metrics,
            ) = simulate_candidate(

                state=state,

                component_profile=(
                    component_profile
                ),

                batch_disturbance=(
                    batch_disturbance
                ),

                candidate=candidate,
            )


            actual_states.append(
                candidate_state
            )


            actual_metrics.append(
                candidate_metrics
            )


            actual_qualities.append(
                float(
                    candidate_metrics[
                        "quality_score"
                    ]
                )
            )


        actual_qualities = np.asarray(
            actual_qualities
        )


        true_best_index = int(
            np.argmin(
                actual_qualities
            )
        )


        true_order = np.argsort(
            actual_qualities
        )


        selected_true_rank = int(
            np.where(
                true_order
                ==
                selected_index
            )[0][0]
        ) + 1


        selected_actual_quality = float(
            actual_qualities[
                selected_index
            ]
        )


        true_best_quality = float(
            actual_qualities[
                true_best_index
            ]
        )


        regret = (
            selected_actual_quality
            -
            true_best_quality
        )


        zero_quality = float(
            actual_qualities[
                0
            ]
        )


        selected_metrics = (
            actual_metrics[
                selected_index
            ]
        )


        selected_utilization = (
            calculate_utilization(

                selected_candidate[
                    "z_adj"
                ],

                selected_candidate[
                    "theta_adj"
                ],

                selected_candidate[
                    "locator_offset"
                ],
            )
        )


        component_records.append(
            {

                "assembly_id":
                    assembly_id,

                "batch_condition":
                    batch_condition,

                "component_index":
                    component_index,

                "severity":
                    row[
                        "severity"
                    ],

                "scenario_type":
                    row[
                        "scenario_type"
                    ],

                "predicted_quality":
                    float(
                        predictions[
                            selected_index
                        ]
                    ),

                "actual_quality":
                    selected_actual_quality,

                "prediction_absolute_error":
                    abs(
                        selected_actual_quality
                        -
                        float(
                            predictions[
                                selected_index
                            ]
                        )
                    ),

                "true_best_of_20_quality":
                    true_best_quality,

                "absolute_regret":
                    regret,

                "exact_best_hit":
                    selected_index
                    ==
                    true_best_index,

                "top3_hit":
                    selected_true_rank
                    <=
                    3,

                "selected_true_rank":
                    selected_true_rank,

                "selected_beats_zero":
                    selected_actual_quality
                    <
                    zero_quality,

                "selected_source":
                    selected_candidate[
                        "source"
                    ],

                "z_adj":
                    selected_candidate[
                        "z_adj"
                    ],

                "theta_adj":
                    selected_candidate[
                        "theta_adj"
                    ],

                "locator_offset":
                    selected_candidate[
                        "locator_offset"
                    ],

                "correction_utilization":
                    selected_utilization,

                "mean_gap":
                    selected_metrics[
                        "mean_gap"
                    ],

                "max_gap":
                    selected_metrics[
                        "max_gap"
                    ],

                "parallelism":
                    selected_metrics[
                        "parallelism_error"
                    ],

                "rms":
                    selected_metrics[
                        "rms_deviation"
                    ],
            }
        )


        # ----------------------------------------------------
        # Advance using RF selected candidate
        # ----------------------------------------------------

        state = actual_states[
            selected_index
        ]


    # ========================================================
    # FINAL ASSEMBLY RESULT
    # ========================================================

    final_metrics = calculate_quality_metrics(
        state
    )


    structured20_final = float(
        final_metrics[
            "quality_score"
        ]
    )


    warmstart_final = float(
        warmstart_row[
            "warmstart_bo_final_quality"
        ]
    )


    zero_final = float(
        original_row[
            "zero_final_quality"
        ]
    )


    random_final = float(
        original_row[
            "random_final_quality"
        ]
    )


    old_bo_final = float(
        original_row[
            "bo_final_quality"
        ]
    )


    assembly_records.append(
        {

            "assembly_id":
                assembly_id,

            "batch_condition":
                batch_condition,

            "zero_final_quality":
                zero_final,

            "random_final_quality":
                random_final,

            "old_bo_final_quality":
                old_bo_final,

            "structured20_final_quality":
                structured20_final,

            "warmstart_bo_final_quality":
                warmstart_final,

            "structured20_beats_zero":
                structured20_final
                <
                zero_final,

            "structured20_beats_random":
                structured20_final
                <
                random_final,

            "structured20_beats_old_bo":
                structured20_final
                <
                old_bo_final,

            "warmstart_beats_structured20":
                warmstart_final
                <
                structured20_final,

            "structured20_beats_warmstart":
                structured20_final
                <
                warmstart_final,

            "difference_warmstart_minus_structured20":
                warmstart_final
                -
                structured20_final,
        }
    )


    if assembly_counter % 10 == 0:


        elapsed = (
            time.time()
            -
            start_time
        )


        print(
            f"Completed "
            f"{assembly_counter}/"
            f"{len(selected_assemblies)}"
            f" | elapsed = "
            f"{elapsed / 60.0:.2f} min"
        )


        pd.DataFrame(
            component_records
        ).to_csv(
            COMPONENT_OUTPUT,
            index=False,
        )


        pd.DataFrame(
            assembly_records
        ).to_csv(
            ASSEMBLY_OUTPUT,
            index=False,
        )


# ============================================================
# DATAFRAMES
# ============================================================

component_result_df = pd.DataFrame(
    component_records
)


assembly_result_df = pd.DataFrame(
    assembly_records
)


# ============================================================
# SYSTEM-LEVEL METRICS
# ============================================================

structured20_mean = (
    assembly_result_df[
        "structured20_final_quality"
    ]
    .mean()
)


warmstart_mean = (
    assembly_result_df[
        "warmstart_bo_final_quality"
    ]
    .mean()
)


zero_mean = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .mean()
)


structured20_win_zero = (
    assembly_result_df[
        "structured20_beats_zero"
    ]
    .mean()
    *
    100.0
)


structured20_win_random = (
    assembly_result_df[
        "structured20_beats_random"
    ]
    .mean()
    *
    100.0
)


structured20_win_oldbo = (
    assembly_result_df[
        "structured20_beats_old_bo"
    ]
    .mean()
    *
    100.0
)


warmstart_win_structured = (
    assembly_result_df[
        "warmstart_beats_structured20"
    ]
    .mean()
    *
    100.0
)


structured_win_warmstart = (
    assembly_result_df[
        "structured20_beats_warmstart"
    ]
    .mean()
    *
    100.0
)


mean_difference = (
    assembly_result_df[
        "difference_warmstart_minus_structured20"
    ]
    .mean()
)


median_difference = (
    assembly_result_df[
        "difference_warmstart_minus_structured20"
    ]
    .median()
)


# ============================================================
# ONLINE METRICS
# ============================================================

exact_hit = (
    component_result_df[
        "exact_best_hit"
    ]
    .mean()
    *
    100.0
)


top3_hit = (
    component_result_df[
        "top3_hit"
    ]
    .mean()
    *
    100.0
)


mean_regret = (
    component_result_df[
        "absolute_regret"
    ]
    .mean()
)


p95_regret = (
    component_result_df[
        "absolute_regret"
    ]
    .quantile(
        0.95
    )
)


mean_prediction_error = (
    component_result_df[
        "prediction_absolute_error"
    ]
    .mean()
)


mean_utilization = (
    component_result_df[
        "correction_utilization"
    ]
    .mean()
)


near_limit = (
    component_result_df[
        "correction_utilization"
    ]
    .ge(
        0.90
    )
    .mean()
    *
    100.0
)


# ============================================================
# STAGE TABLE
# ============================================================

stage_rows = []


for component_index in range(
    1,
    6,
):


    subset = component_result_df[
        component_result_df[
            "component_index"
        ]
        ==
        component_index
    ]


    stage_rows.append(
        {

            "component_index":
                component_index,

            "mean_actual_quality":
                subset[
                    "actual_quality"
                ]
                .mean(),

            "mean_prediction_abs_error":
                subset[
                    "prediction_absolute_error"
                ]
                .mean(),

            "exact_best_hit_rate_percent":
                subset[
                    "exact_best_hit"
                ]
                .mean()
                *
                100.0,

            "top3_hit_rate_percent":
                subset[
                    "top3_hit"
                ]
                .mean()
                *
                100.0,

            "mean_regret":
                subset[
                    "absolute_regret"
                ]
                .mean(),

            "selected_beats_zero_percent":
                subset[
                    "selected_beats_zero"
                ]
                .mean()
                *
                100.0,

            "mean_utilization":
                subset[
                    "correction_utilization"
                ]
                .mean(),
        }
    )


stage_df = pd.DataFrame(
    stage_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    {

        "metric": [

            "zero_mean_final_quality",

            "structured20_mean_final_quality",

            "warmstart_bo_mean_final_quality",

            "structured20_win_vs_zero_percent",

            "structured20_win_vs_random_percent",

            "structured20_win_vs_old_bo_percent",

            "warmstart_win_vs_structured20_percent",

            "structured20_win_vs_warmstart_percent",

            "mean_warmstart_minus_structured20_quality",

            "median_warmstart_minus_structured20_quality",

            "structured20_mean_prediction_abs_error",

            "structured20_exact_best_hit_rate_percent",

            "structured20_top3_hit_rate_percent",

            "structured20_mean_regret",

            "structured20_p95_regret",

            "structured20_mean_utilization",

            "structured20_near_limit_rate_percent",
        ],

        "value": [

            zero_mean,

            structured20_mean,

            warmstart_mean,

            structured20_win_zero,

            structured20_win_random,

            structured20_win_oldbo,

            warmstart_win_structured,

            structured_win_warmstart,

            mean_difference,

            median_difference,

            mean_prediction_error,

            exact_hit,

            top3_hit,

            mean_regret,

            p95_regret,

            mean_utilization,

            near_limit,
        ],
    }
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True,
)


os.makedirs(
    "results/tables",
    exist_ok=True,
)


component_result_df.to_csv(
    COMPONENT_OUTPUT,
    index=False,
)


assembly_result_df.to_csv(
    ASSEMBLY_OUTPUT,
    index=False,
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


stage_df.to_csv(
    STAGE_OUTPUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

runtime = (
    time.time()
    -
    start_time
)


print(
    "\n"
    "============================================================"
)

print(
    "EQUAL-BUDGET ABLATION RESULTS"
)

print(
    "============================================================"
)


print(
    f"\nZero mean final quality          : "
    f"{zero_mean:.4f}"
)


print(
    f"Structured-20 mean final quality : "
    f"{structured20_mean:.4f}"
)


print(
    f"Warm-start BO mean final quality : "
    f"{warmstart_mean:.4f}"
)


print(
    f"\nStructured-20 beats ZERO         : "
    f"{structured20_win_zero:.2f}%"
)


print(
    f"Structured-20 beats RANDOM       : "
    f"{structured20_win_random:.2f}%"
)


print(
    f"Structured-20 beats OLD BO       : "
    f"{structured20_win_oldbo:.2f}%"
)


print(
    f"\nWarm-start BO beats Structured-20: "
    f"{warmstart_win_structured:.2f}%"
)


print(
    f"Structured-20 beats Warm-start BO: "
    f"{structured_win_warmstart:.2f}%"
)


print(
    f"\nMean (Warm-start - Structured20) : "
    f"{mean_difference:.6f}"
)


print(
    f"Median difference                : "
    f"{median_difference:.6f}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "STRUCTURED-20 ONLINE QUALITY"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Prediction MAE                    : "
    f"{mean_prediction_error:.4f}"
)


print(
    f"Exact-best-of-20                 : "
    f"{exact_hit:.2f}%"
)


print(
    f"Top-3-of-20                      : "
    f"{top3_hit:.2f}%"
)


print(
    f"Mean regret                      : "
    f"{mean_regret:.4f}"
)


print(
    f"P95 regret                       : "
    f"{p95_regret:.4f}"
)


print(
    f"Mean utilization                 : "
    f"{mean_utilization:.3f}"
)


print(
    f">=90% capability                 : "
    f"{near_limit:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "STAGE-WISE"
)

print(
    "------------------------------------------------------------\n"
)


print(
    stage_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "BAYESIAN OPTIMIZATION CONTRIBUTION TEST"
)

print(
    "============================================================"
)


relative_difference_percent = (
    (
        structured20_mean
        -
        warmstart_mean
    )
    /
    max(
        abs(
            structured20_mean
        ),
        1e-9,
    )
    *
    100.0
)


print(
    f"\nRelative mean quality benefit of warm-start BO "
    f"vs Structured-20:"
)


print(
    f"{relative_difference_percent:.3f}%"
)


if (
    relative_difference_percent
    >
    2.0
    and
    warmstart_win_structured
    >
    55.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "Bayesian refinement provides a measurable additional "
        "benefit beyond equal-budget structured RF search."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "The final architecture can defensibly retain "
        "structured initialization plus Bayesian refinement."
    )


elif (
    abs(
        relative_difference_percent
    )
    <=
    2.0
):


    print(
        "\nRESULT B:"
    )


    print(
        "Warm-start BO and equal-budget structured RF search "
        "perform essentially equivalently at the system level."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "Structured initialization is responsible for most "
        "of the performance recovery."
    )


    print(
        "Bayesian refinement currently provides only limited "
        "incremental benefit."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Investigate where BO refinement is useful rather than "
        "forcing it at every assembly decision."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "Equal-budget structured RF search outperforms the "
        "current warm-start Bayesian refinement."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "The current BO-refinement stage does not yet justify "
        "its computational cost."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Use BO selectively or redesign its refinement trigger "
        "instead of applying BO to every component."
    )


print(
    "\nIMPORTANT:"
)


print(
    "Do not interpret a non-superior BO result as a failure."
)


print(
    "This ablation identifies which part of the adaptive "
    "architecture actually produces the observed performance."
)


print(
    "\nSaved:"
)


print(
    COMPONENT_OUTPUT
)

print(
    ASSEMBLY_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print(
    STAGE_OUTPUT
)


print(
    f"\nRuntime: "
    f"{runtime / 60.0:.2f} min"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 EQUAL-BUDGET SEARCH ABLATION COMPLETED"
)

print(
    "============================================================"
)