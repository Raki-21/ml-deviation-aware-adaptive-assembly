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
# V3.2
# STRUCTURED RF CONTROLLER DIAGNOSIS
# ============================================================
#
# RESEARCH QUESTION
#
# Previous evidence:
#
#   - simulator-direct greedy search performed extremely well
#   - surrogate candidate ranking on unseen states is strong
#   - conventional BO controller remained weak
#
# Therefore the next hypothesis is:
#
#   The remaining problem may be the SEARCH STRATEGY used
#   between the surrogate and the correction space rather than
#   the surrogate itself.
#
#
# This experiment does NOT use Bayesian Optimization.
#
# Instead, at each sequential decision:
#
#   1. generate the same type of structured correction
#      candidates used during closed-loop training
#
#   2. predict all candidates with the profile-aware RF
#
#   3. select the candidate with lowest predicted quality
#
#   4. verify ALL candidates with the actual simulator
#
#   5. determine whether RF selected the actual best / top-3
#
#   6. advance the real sequential state using the RF-selected
#      correction
#
#
# This isolates:
#
#       surrogate quality
#
# from
#
#       Bayesian optimizer search quality.
#
#
# IMPORTANT
#
# This is a diagnostic controller, NOT the intended final
# optimization method.
#
# If it succeeds, Bayesian Optimization will be redesigned
# using informed / structured initialization rather than
# abandoned.
# ============================================================


# ============================================================
# PATHS
# ============================================================

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

ORIGINAL_PILOT_FILE = (
    "data/processed/"
    "v3_2_ml_bo_pilot_assembly_results.csv"
)

MODEL_FILE = (
    "models/"
    "v3_2_profile_aware_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_profile_aware_surrogate_features.txt"
)


COMPONENT_OUTPUT = (
    "data/processed/"
    "v3_2_structured_rf_controller_component_results.csv"
)

ASSEMBLY_OUTPUT = (
    "data/processed/"
    "v3_2_structured_rf_controller_assembly_results.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_structured_rf_controller_summary.csv"
)

STAGE_OUTPUT = (
    "results/tables/"
    "v3_2_structured_rf_controller_by_stage.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    ORIGINAL_PILOT_FILE,
    MODEL_FILE,
    FEATURE_FILE,
]


missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]


if missing_files:

    print(
        "\nERROR - Missing files:"
    )

    for path in missing_files:

        print(path)

    raise SystemExit(
        "\nRequired V3.2 files are missing."
    )


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 20260825

N_COMPONENTS = 5


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
# LOAD DATA + MODEL
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 STRUCTURED RF CONTROLLER DIAGNOSIS"
)

print(
    "============================================================"
)


component_df = pd.read_csv(
    COMPONENT_FILE
)


original_pilot_df = pd.read_csv(
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


# ============================================================
# SAME 100 ASSEMBLIES AS ORIGINAL PILOT
# ============================================================

selected_assemblies = sorted(
    original_pilot_df[
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
    "Candidates / decision       : 21"
)

print(
    "Bayesian Optimization used  : NO"
)

print(
    "Actual candidate verification: YES"
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
                **
                2
            )
            /
            (
                2.0
                *
                fixture_sigma
                **
                2
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
# SCALAR STATE FEATURES
# ============================================================

def calculate_scalar_state_features(
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


    angle_deg = float(
        np.rad2deg(
            np.arctan(
                signed_end_difference
                /
                PROFILE_LENGTH_MM
            )
        )
    )


    return {

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
            angle_deg,
    }


# ============================================================
# PROFILE STATE FEATURES
# ============================================================

def calculate_profile_state_features(
    state
):

    result = {}


    for (
        name,
        index,
    ) in zip(
        PROFILE_SAMPLE_NAMES,
        PROFILE_SAMPLE_INDICES,
    ):

        result[
            name
        ] = float(
            state[
                index
            ]
        )


    return result


# ============================================================
# CAPABILITY
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
# CLIP CORRECTION
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
# INFORMED CORRECTION CENTRE
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
# STRUCTURED CANDIDATE SET
# ============================================================

def generate_candidate_set(
    state,
    row,
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    candidates = []


    # --------------------------------------------------------
    # ZERO
    # --------------------------------------------------------

    candidates.append(
        {
            "z_adj":
                0.0,

            "theta_adj":
                0.0,

            "locator_offset":
                0.0,

            "candidate_source":
                "zero",
        }
    )


    # --------------------------------------------------------
    # INFORMED
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

            "candidate_source":
                "informed",
        }
    )


    # --------------------------------------------------------
    # LOCAL INFORMED - 8
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

                "candidate_source":
                    "local_informed",
            }
        )


    # --------------------------------------------------------
    # MODERATE RANDOM - 6
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

                "candidate_source":
                    "moderate_random",
            }
        )


    # --------------------------------------------------------
    # BROAD RANDOM - 4
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

                "candidate_source":
                    "broad_random",
            }
        )


    # --------------------------------------------------------
    # NEAR BOUNDARY - 1
    # --------------------------------------------------------

    signs = rng.choice(
        [
            -1.0,
            1.0,
        ],
        size=3,
    )


    candidates.append(
        {
            "z_adj":
                float(
                    signs[
                        0
                    ]
                    *
                    0.90
                    *
                    Z_LIMIT
                ),

            "theta_adj":
                float(
                    signs[
                        1
                    ]
                    *
                    0.90
                    *
                    THETA_LIMIT
                ),

            "locator_offset":
                float(
                    signs[
                        2
                    ]
                    *
                    0.90
                    *
                    LOCATOR_LIMIT
                ),

            "candidate_source":
                "near_boundary",
        }
    )


    return candidates


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_input(
    row,
    state,
    candidates,
):


    scalar_features = (
        calculate_scalar_state_features(
            state
        )
    )


    profile_features = (
        calculate_profile_state_features(
            state
        )
    )


    records = []


    for candidate in candidates:


        utilization = calculate_utilization(

            candidate[
                "z_adj"
            ],

            candidate[
                "theta_adj"
            ],

            candidate[
                "locator_offset"
            ],
        )


        record = {

            **scalar_features,

            **profile_features,


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
                candidate[
                    "z_adj"
                ],

            "theta_adj":
                candidate[
                    "theta_adj"
                ],

            "locator_offset":
                candidate[
                    "locator_offset"
                ],

            "correction_utilization":
                utilization,


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


    missing_features = [
        feature
        for feature
        in FEATURE_COLUMNS
        if feature
        not in X.columns
    ]


    if missing_features:

        raise ValueError(
            "\nMissing model features:\n"
            +
            "\n".join(
                missing_features
            )
        )


    return X[
        FEATURE_COLUMNS
    ]


# ============================================================
# ACTUAL CANDIDATE EVALUATION
# ============================================================

def evaluate_actual_candidate(
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


    candidate_state = update_assembly_state(

        previous_state=state,

        component_deviation=(
            component_profile
        ),

        fixture_drift=(
            batch_disturbance
        ),

        correction=(
            correction
        ),
    )


    metrics = calculate_quality_metrics(
        candidate_state
    )


    return (
        candidate_state,
        metrics,
    )


# ============================================================
# STORAGE
# ============================================================

component_records = []

assembly_records = []


# ============================================================
# RUN
# ============================================================

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


    controller_state = create_initial_state()


    original_row = original_pilot_df[
        original_pilot_df[
            "assembly_id"
        ]
        ==
        assembly_id
    ].iloc[
        0
    ]


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


        candidate_seed = (
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


        candidates = generate_candidate_set(

            state=controller_state,

            row=row,

            seed=candidate_seed,
        )


        # ----------------------------------------------------
        # RF prediction for all 21 candidates in one batch
        # ----------------------------------------------------

        X = build_model_input(

            row=row,

            state=controller_state,

            candidates=candidates,
        )


        predictions = model.predict(
            X
        )


        # ----------------------------------------------------
        # ACTUAL simulator verification for all candidates
        # ----------------------------------------------------

        actual_qualities = []

        actual_states = []

        actual_metrics_list = []


        for candidate in candidates:


            (
                candidate_state,
                candidate_metrics,
            ) = evaluate_actual_candidate(

                state=controller_state,

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


            actual_metrics_list.append(
                candidate_metrics
            )


            actual_qualities.append(
                candidate_metrics[
                    "quality_score"
                ]
            )


        actual_qualities = np.asarray(
            actual_qualities,
            dtype=float,
        )


        # ----------------------------------------------------
        # RF SELECTED
        # ----------------------------------------------------

        predicted_best_index = int(
            np.argmin(
                predictions
            )
        )


        selected_candidate = candidates[
            predicted_best_index
        ]


        selected_actual_quality = float(
            actual_qualities[
                predicted_best_index
            ]
        )


        selected_metrics = (
            actual_metrics_list[
                predicted_best_index
            ]
        )


        selected_state = (
            actual_states[
                predicted_best_index
            ]
        )


        # ----------------------------------------------------
        # TRUE BEST WITHIN SAME 21 CANDIDATES
        # ----------------------------------------------------

        true_best_index = int(
            np.argmin(
                actual_qualities
            )
        )


        true_best_quality = float(
            actual_qualities[
                true_best_index
            ]
        )


        true_order = np.argsort(
            actual_qualities
        )


        selected_true_rank = int(
            np.where(
                true_order
                ==
                predicted_best_index
            )[0][0]
        ) + 1


        exact_best_hit = (
            predicted_best_index
            ==
            true_best_index
        )


        top3_hit = (
            selected_true_rank
            <=
            3
        )


        regret = (
            selected_actual_quality
            -
            true_best_quality
        )


        # ----------------------------------------------------
        # ZERO CANDIDATE
        # ----------------------------------------------------

        zero_index = next(

            index

            for index, candidate
            in enumerate(
                candidates
            )

            if candidate[
                "candidate_source"
            ]
            ==
            "zero"
        )


        zero_candidate_quality = float(
            actual_qualities[
                zero_index
            ]
        )


        selected_beats_zero = (
            selected_actual_quality
            <
            zero_candidate_quality
        )


        true_best_beats_zero = (
            true_best_quality
            <
            zero_candidate_quality
        )


        utilization = calculate_utilization(

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
                            predicted_best_index
                        ]
                    ),

                "selected_actual_quality":
                    selected_actual_quality,

                "true_best_candidate_quality":
                    true_best_quality,

                "zero_candidate_quality":
                    zero_candidate_quality,

                "exact_best_hit":
                    exact_best_hit,

                "top3_hit":
                    top3_hit,

                "selected_true_rank":
                    selected_true_rank,

                "absolute_regret":
                    regret,

                "selected_beats_zero":
                    selected_beats_zero,

                "true_best_beats_zero":
                    true_best_beats_zero,

                "selected_candidate_source":
                    selected_candidate[
                        "candidate_source"
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
                    utilization,

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
        # ADVANCE USING RF-SELECTED CORRECTION
        # ----------------------------------------------------

        controller_state = selected_state


    # ========================================================
    # FINAL ASSEMBLY
    # ========================================================

    final_metrics = calculate_quality_metrics(
        controller_state
    )


    structured_final_quality = float(
        final_metrics[
            "quality_score"
        ]
    )


    zero_final_quality = float(
        original_row[
            "zero_final_quality"
        ]
    )


    random_final_quality = float(
        original_row[
            "random_final_quality"
        ]
    )


    old_bo_final_quality = float(
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
                zero_final_quality,

            "random_final_quality":
                random_final_quality,

            "old_bo_final_quality":
                old_bo_final_quality,

            "structured_rf_final_quality":
                structured_final_quality,

            "structured_beats_zero":
                structured_final_quality
                <
                zero_final_quality,

            "structured_beats_random":
                structured_final_quality
                <
                random_final_quality,

            "structured_beats_old_bo":
                structured_final_quality
                <
                old_bo_final_quality,
        }
    )


    if (
        assembly_counter
        %
        10
        ==
        0
    ):


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


        # Checkpoint

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
# METRICS
# ============================================================

exact_hit_rate = (
    component_result_df[
        "exact_best_hit"
    ]
    .mean()
    *
    100.0
)


top3_hit_rate = (
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


selected_beats_zero_rate = (
    component_result_df[
        "selected_beats_zero"
    ]
    .mean()
    *
    100.0
)


true_best_beats_zero_rate = (
    component_result_df[
        "true_best_beats_zero"
    ]
    .mean()
    *
    100.0
)


mean_utilization = (
    component_result_df[
        "correction_utilization"
    ]
    .mean()
)


near_limit_rate = (
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


structured_mean_final = (
    assembly_result_df[
        "structured_rf_final_quality"
    ]
    .mean()
)


zero_mean_final = (
    assembly_result_df[
        "zero_final_quality"
    ]
    .mean()
)


random_mean_final = (
    assembly_result_df[
        "random_final_quality"
    ]
    .mean()
)


old_bo_mean_final = (
    assembly_result_df[
        "old_bo_final_quality"
    ]
    .mean()
)


win_vs_zero = (
    assembly_result_df[
        "structured_beats_zero"
    ]
    .mean()
    *
    100.0
)


win_vs_random = (
    assembly_result_df[
        "structured_beats_random"
    ]
    .mean()
    *
    100.0
)


win_vs_old_bo = (
    assembly_result_df[
        "structured_beats_old_bo"
    ]
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
                    "selected_actual_quality"
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

            "random_mean_final_quality",

            "old_bo_mean_final_quality",

            "structured_rf_mean_final_quality",

            "structured_rf_win_vs_zero_percent",

            "structured_rf_win_vs_random_percent",

            "structured_rf_win_vs_old_bo_percent",

            "online_exact_best_hit_rate_percent",

            "online_top3_hit_rate_percent",

            "online_mean_regret",

            "online_p95_regret",

            "online_selected_beats_zero_rate_percent",

            "online_true_best_beats_zero_rate_percent",

            "mean_correction_utilization",

            "near_limit_rate_percent",
        ],

        "value": [

            zero_mean_final,

            random_mean_final,

            old_bo_mean_final,

            structured_mean_final,

            win_vs_zero,

            win_vs_random,

            win_vs_old_bo,

            exact_hit_rate,

            top3_hit_rate,

            mean_regret,

            p95_regret,

            selected_beats_zero_rate,

            true_best_beats_zero_rate,

            mean_utilization,

            near_limit_rate,
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
    "STRUCTURED RF CONTROLLER RESULTS"
)

print(
    "============================================================"
)


print(
    f"\nZero mean final quality       : "
    f"{zero_mean_final:.4f}"
)


print(
    f"Random mean final quality     : "
    f"{random_mean_final:.4f}"
)


print(
    f"Old BO mean final quality     : "
    f"{old_bo_mean_final:.4f}"
)


print(
    f"Structured RF mean final Q    : "
    f"{structured_mean_final:.4f}"
)


print(
    f"\nStructured RF beats ZERO      : "
    f"{win_vs_zero:.2f}%"
)


print(
    f"Structured RF beats RANDOM    : "
    f"{win_vs_random:.2f}%"
)


print(
    f"Structured RF beats OLD BO    : "
    f"{win_vs_old_bo:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "ONLINE RANKING"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Exact-best hit rate           : "
    f"{exact_hit_rate:.2f}%"
)


print(
    f"Top-3 hit rate                : "
    f"{top3_hit_rate:.2f}%"
)


print(
    f"Mean regret                   : "
    f"{mean_regret:.4f}"
)


print(
    f"P95 regret                    : "
    f"{p95_regret:.4f}"
)


print(
    f"RF selected beats zero        : "
    f"{selected_beats_zero_rate:.2f}%"
)


print(
    f"True candidate best beats zero: "
    f"{true_best_beats_zero_rate:.2f}%"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "CAPABILITY"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Mean utilization              : "
    f"{mean_utilization:.3f}"
)


print(
    f">=90% capability              : "
    f"{near_limit_rate:.2f}%"
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
# SCIENTIFIC DECISION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "OPTIMIZER-SEARCH DIAGNOSIS"
)

print(
    "============================================================"
)


if (
    structured_mean_final
    <
    zero_mean_final
    and
    win_vs_zero
    >=
    70.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "The structured RF controller substantially "
        "outperforms zero correction."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "The surrogate is capable of useful sequential "
        "decision-making when evaluated in a physically "
        "structured candidate region."
    )


    print(
        "The remaining weakness is strongly associated with "
        "the previous continuous Bayesian-optimization search "
        "strategy / initialization."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Redesign Bayesian Optimization using structured "
        "informed initialization and a locally focused search "
        "around physically meaningful correction regions."
    )


elif (
    structured_mean_final
    <
    zero_mean_final
    and
    win_vs_zero
    >=
    55.0
):


    print(
        "\nRESULT B:"
    )


    print(
        "Structured RF improves mean final quality, but "
        "performance is still inconsistent."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "Both optimizer search and remaining closed-loop "
        "surrogate error contribute."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Strengthen structured training coverage before "
        "reintroducing Bayesian Optimization."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "Structured RF does not reliably beat zero correction."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "The problem remains deeper than BO search alone."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Investigate adaptive-state representation and "
        "controller training distribution further."
    )


print(
    "\nIMPORTANT:"
)

print(
    "This is a diagnostic controller, not the final V3.2 "
    "optimization architecture."
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
    "V3.2 STRUCTURED RF CONTROLLER DIAGNOSIS COMPLETED"
)

print(
    "============================================================"
)