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
# PRE-DECISION SELECTIVE BO TRIGGER CALIBRATION
# ============================================================
#
# PURPOSE
#
# Equal-budget ablation established that:
#
#   Structured-20 RF search
#       and
#   Structured warm-start BO
#
# have nearly identical average system-level performance.
#
# BO refinement was useful only in a small minority of cases.
#
#
# Therefore the next research question is:
#
#   Can difficult / ambiguous decisions be identified BEFORE
#   a correction is physically applied?
#
#
# ONLINE-AVAILABLE SIGNALS
#
# 1. Predicted best-vs-second-best quality margin
#
#       margin =
#       Qhat_second - Qhat_best
#
#    Small margin -> ambiguous candidate ranking.
#
#
# 2. Random-Forest tree disagreement
#
#       tree_std =
#       std(prediction from each RF tree)
#
#    Large disagreement -> lower model confidence.
#
#
# 3. Selected correction utilization
#
#       U = max normalized correction magnitude
#
#
# IMPORTANT SCIENTIFIC RULE
#
# Actual simulator regret and exact-best information are used
# ONLY to evaluate/calibrate the trigger.
#
# They are POST-HOC labels.
#
# They are NOT allowed to become online trigger inputs.
#
#
# THRESHOLDS
#
# Thresholds are NOT arbitrarily chosen.
#
# Candidate thresholds are derived from empirical quantiles of
# the 500 unseen sequential decisions.
#
# We then search for trigger rules that:
#
#   - activate on a limited fraction of decisions
#   - capture high-regret decisions
#   - capture exact-best misses
#
#
# This is calibration, not final validation.
# ============================================================


# ============================================================
# PATHS
# ============================================================

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_full_component_level_dataset.csv"
)

STRUCTURED_COMPONENT_FILE = (
    "data/processed/"
    "v3_2_equal_budget_structured20_component_results.csv"
)

STRUCTURED_ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_equal_budget_structured20_assembly_results.csv"
)

MODEL_FILE = (
    "models/"
    "v3_2_profile_aware_quality_surrogate.joblib"
)

FEATURE_FILE = (
    "models/"
    "v3_2_profile_aware_surrogate_features.txt"
)


SIGNAL_OUTPUT = (
    "data/processed/"
    "v3_2_predecision_ambiguity_signals.csv"
)

GRID_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_trigger_grid.csv"
)

PARETO_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_trigger_pareto.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_selective_bo_trigger_calibration_summary.csv"
)

STAGE_OUTPUT = (
    "results/tables/"
    "v3_2_predecision_signal_by_stage.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    STRUCTURED_COMPONENT_FILE,
    STRUCTURED_ASSEMBLY_FILE,
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
        "\nERROR - Missing required files:"
    )

    for path in missing_files:
        print(path)

    raise SystemExit(
        "\nComplete the equal-budget ablation first."
    )


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 20260825

N_COMPONENTS = 5

TOTAL_CANDIDATES = 20


Z_LIMIT = 2.5

THETA_LIMIT = 1.2

LOCATOR_LIMIT = 1.0


# ============================================================
# TRIGGER BUDGET
# ============================================================
#
# The purpose of selective BO is computational selectivity.
#
# Therefore we initially require the trigger to activate on no
# more than 20% of component decisions.
#
# This is NOT claimed as an industrial optimum.
#
# It is a screening constraint used to identify useful
# low-frequency trigger rules.
# ============================================================

MAX_TRIGGER_RATE_PERCENT = 20.0


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
# LOAD
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 PRE-DECISION SELECTIVE BO TRIGGER CALIBRATION"
)

print(
    "============================================================"
)


component_df = pd.read_csv(
    COMPONENT_FILE
)


structured_component_df = pd.read_csv(
    STRUCTURED_COMPONENT_FILE
)


structured_assembly_df = pd.read_csv(
    STRUCTURED_ASSEMBLY_FILE
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
    structured_assembly_df[
        "assembly_id"
    ]
    .unique()
    .tolist()
)


print(
    f"\nAssemblies          : "
    f"{len(selected_assemblies)}"
)

print(
    f"Expected decisions  : "
    f"{len(selected_assemblies) * N_COMPONENTS}"
)

print(
    f"RF trees            : "
    f"{len(model.estimators_)}"
)


# ============================================================
# COMPONENT PROFILE RECONSTRUCTION
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
# UTILIZATION
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
# RECREATE EXACT STRUCTURED-20 CANDIDATE SET
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
    # ZERO
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

            "source":
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

                "source":
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

                "source":
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

                "source":
                    "broad_random",
            }
        )


    if len(
        candidates
    ) != TOTAL_CANDIDATES:

        raise RuntimeError(
            f"Expected {TOTAL_CANDIDATES} candidates, "
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


        records.append(
            {

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
        )


    X = pd.DataFrame(
        records
    )


    return X[
        FEATURE_COLUMNS
    ]


# ============================================================
# APPLY SAVED STRUCTURED CORRECTION
# ============================================================

def advance_state_with_saved_correction(
    state,
    row,
    saved_decision,
):

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


    correction = correction_profile(

        z_adj_mm=(
            saved_decision[
                "z_adj"
            ]
        ),

        theta_adj_deg=(
            saved_decision[
                "theta_adj"
            ]
        ),

        locator_offset_mm=(
            saved_decision[
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

        correction=(
            correction
        ),
    )


    return new_state


# ============================================================
# COLLECT PRE-DECISION SIGNALS
# ============================================================

signal_records = []


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


    for _, row in assembly_rows.iterrows():


        component_index = int(
            row[
                "component_index"
            ]
        )


        saved_rows = structured_component_df[
            (
                structured_component_df[
                    "assembly_id"
                ]
                ==
                assembly_id
            )
            &
            (
                structured_component_df[
                    "component_index"
                ]
                ==
                component_index
            )
        ]


        if len(
            saved_rows
        ) != 1:

            raise ValueError(
                f"\nExpected one saved structured decision "
                f"for assembly {assembly_id}, "
                f"component {component_index}; "
                f"found {len(saved_rows)}."
            )


        saved_decision = saved_rows.iloc[
            0
        ]


        decision_seed = (
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
                seed=decision_seed,
            )
        )


        X = build_model_input(
            row=row,
            state=state,
            candidates=candidates,
        )


        # ----------------------------------------------------
        # FOREST MEAN PREDICTION
        # ----------------------------------------------------

        forest_predictions = model.predict(
            X
        )


        prediction_order = np.argsort(
            forest_predictions
        )


        best_index = int(
            prediction_order[
                0
            ]
        )


        second_index = int(
            prediction_order[
                1
            ]
        )


        best_prediction = float(
            forest_predictions[
                best_index
            ]
        )


        second_prediction = float(
            forest_predictions[
                second_index
            ]
        )


        prediction_margin = float(
            second_prediction
            -
            best_prediction
        )


        normalized_margin = float(
            prediction_margin
            /
            max(
                abs(
                    best_prediction
                ),
                1e-6,
            )
        )


        # ----------------------------------------------------
        # TREE-LEVEL UNCERTAINTY
        # ----------------------------------------------------
        #
        # Evaluate the selected candidate with every tree.
        # ----------------------------------------------------

        selected_X = (
            X.iloc[
                [
                    best_index
                ]
            ]
            .to_numpy()
        )


        tree_predictions = np.asarray(
            [
                float(
                    tree.predict(
                        selected_X
                    )[0]
                )

                for tree
                in model.estimators_
            ]
        )


        selected_tree_std = float(
            np.std(
                tree_predictions
            )
        )


        selected_tree_iqr = float(
            np.quantile(
                tree_predictions,
                0.75,
            )
            -
            np.quantile(
                tree_predictions,
                0.25,
            )
        )


        selected_tree_range = float(
            np.max(
                tree_predictions
            )
            -
            np.min(
                tree_predictions
            )
        )


        selected_candidate = candidates[
            best_index
        ]


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


        # ----------------------------------------------------
        # POST-HOC LABELS
        #
        # These are allowed for calibration/analysis ONLY.
        # ----------------------------------------------------

        actual_regret = float(
            saved_decision[
                "absolute_regret"
            ]
        )


        exact_best_hit = bool(
            saved_decision[
                "exact_best_hit"
            ]
        )


        top3_hit = bool(
            saved_decision[
                "top3_hit"
            ]
        )


        signal_records.append(
            {

                "assembly_id":
                    assembly_id,

                "batch_condition":
                    saved_decision[
                        "batch_condition"
                    ],

                "component_index":
                    component_index,

                "severity":
                    saved_decision[
                        "severity"
                    ],

                "scenario_type":
                    saved_decision[
                        "scenario_type"
                    ],


                # ============================================
                # PRE-DECISION SIGNALS
                # ============================================

                "best_predicted_quality":
                    best_prediction,

                "second_best_predicted_quality":
                    second_prediction,

                "predicted_quality_margin":
                    prediction_margin,

                "normalized_quality_margin":
                    normalized_margin,

                "selected_tree_std":
                    selected_tree_std,

                "selected_tree_iqr":
                    selected_tree_iqr,

                "selected_tree_range":
                    selected_tree_range,

                "selected_utilization":
                    selected_utilization,

                "selected_candidate_source":
                    selected_candidate[
                        "source"
                    ],


                # ============================================
                # POST-HOC CALIBRATION LABELS
                # ============================================

                "actual_regret":
                    actual_regret,

                "exact_best_hit":
                    exact_best_hit,

                "exact_best_miss":
                    not exact_best_hit,

                "top3_hit":
                    top3_hit,
            }
        )


        # ----------------------------------------------------
        # Reconstruct EXACT structured trajectory using the
        # previously saved selected correction.
        # ----------------------------------------------------

        state = advance_state_with_saved_correction(

            state=state,

            row=row,

            saved_decision=(
                saved_decision
            ),
        )


    if assembly_counter % 20 == 0:

        elapsed = (
            time.time()
            -
            start_time
        )


        print(
            f"Processed "
            f"{assembly_counter}/"
            f"{len(selected_assemblies)} assemblies"
            f" | elapsed = "
            f"{elapsed:.1f} sec"
        )


# ============================================================
# SIGNAL DATAFRAME
# ============================================================

signal_df = pd.DataFrame(
    signal_records
)


if len(
    signal_df
) != (
    len(
        selected_assemblies
    )
    *
    N_COMPONENTS
):

    raise ValueError(
        "\nUnexpected number of decision records."
    )


# ============================================================
# HIGH-REGRET LABEL
# ============================================================
#
# Instead of arbitrarily declaring a fixed regret value
# "difficult", we define the upper 10% of observed regret as
# high-regret decisions for this calibration.
#
# This threshold is a DATA-DRIVEN diagnostic label.
#
# It is NOT an online input.
# ============================================================

high_regret_threshold = float(
    signal_df[
        "actual_regret"
    ]
    .quantile(
        0.90
    )
)


signal_df[
    "high_regret"
] = (
    signal_df[
        "actual_regret"
    ]
    >=
    high_regret_threshold
)


n_high_regret = int(
    signal_df[
        "high_regret"
    ]
    .sum()
)


n_exact_misses = int(
    signal_df[
        "exact_best_miss"
    ]
    .sum()
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "CALIBRATION LABELS"
)

print(
    "------------------------------------------------------------"
)


print(
    f"Decisions                 : "
    f"{len(signal_df)}"
)


print(
    f"High-regret threshold P90 : "
    f"{high_regret_threshold:.6f}"
)


print(
    f"High-regret decisions     : "
    f"{n_high_regret}"
)


print(
    f"Exact-best misses         : "
    f"{n_exact_misses}"
)


# ============================================================
# SIGNAL QUANTILES
# ============================================================

margin_quantiles = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]


uncertainty_quantiles = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


utilization_quantiles = [
    0.80,
    0.85,
    0.90,
    0.95,
]


# ============================================================
# TRIGGER EVALUATION
# ============================================================

def evaluate_trigger(
    trigger_mask,
    rule_name,
    margin_threshold=np.nan,
    uncertainty_threshold=np.nan,
    utilization_threshold=np.nan,
):


    trigger_mask = pd.Series(
        trigger_mask,
        index=signal_df.index,
    )


    n_triggered = int(
        trigger_mask.sum()
    )


    trigger_rate = (
        n_triggered
        /
        len(
            signal_df
        )
        *
        100.0
    )


    high_regret_mask = signal_df[
        "high_regret"
    ]


    exact_miss_mask = signal_df[
        "exact_best_miss"
    ]


    high_regret_capture = (
        (
            trigger_mask
            &
            high_regret_mask
        )
        .sum()
        /
        max(
            high_regret_mask.sum(),
            1,
        )
        *
        100.0
    )


    exact_miss_capture = (
        (
            trigger_mask
            &
            exact_miss_mask
        )
        .sum()
        /
        max(
            exact_miss_mask.sum(),
            1,
        )
        *
        100.0
    )


    # --------------------------------------------------------
    # Precision-like diagnostic:
    #
    # among triggered decisions, how many are actually
    # high-regret / exact-best misses?
    # --------------------------------------------------------

    high_regret_precision = (
        (
            trigger_mask
            &
            high_regret_mask
        )
        .sum()
        /
        max(
            n_triggered,
            1,
        )
        *
        100.0
    )


    exact_miss_precision = (
        (
            trigger_mask
            &
            exact_miss_mask
        )
        .sum()
        /
        max(
            n_triggered,
            1,
        )
        *
        100.0
    )


    triggered_regret_mean = (
        signal_df.loc[
            trigger_mask,
            "actual_regret"
        ]
        .mean()
        if n_triggered
        >
        0
        else np.nan
    )


    non_triggered_regret_mean = (
        signal_df.loc[
            ~trigger_mask,
            "actual_regret"
        ]
        .mean()
        if (
            ~trigger_mask
        ).sum()
        >
        0
        else np.nan
    )


    return {

        "rule_name":
            rule_name,

        "margin_threshold":
            margin_threshold,

        "uncertainty_threshold":
            uncertainty_threshold,

        "utilization_threshold":
            utilization_threshold,

        "n_triggered":
            n_triggered,

        "trigger_rate_percent":
            trigger_rate,

        "high_regret_capture_percent":
            high_regret_capture,

        "exact_miss_capture_percent":
            exact_miss_capture,

        "high_regret_precision_percent":
            high_regret_precision,

        "exact_miss_precision_percent":
            exact_miss_precision,

        "triggered_mean_regret":
            triggered_regret_mean,

        "nontriggered_mean_regret":
            non_triggered_regret_mean,
    }


# ============================================================
# GRID SEARCH
# ============================================================

grid_records = []


# ============================================================
# 1. MARGIN ONLY
# ============================================================

for q_margin in margin_quantiles:


    margin_threshold = float(
        signal_df[
            "predicted_quality_margin"
        ]
        .quantile(
            q_margin
        )
    )


    trigger = (
        signal_df[
            "predicted_quality_margin"
        ]
        <=
        margin_threshold
    )


    record = evaluate_trigger(

        trigger_mask=trigger,

        rule_name=(
            f"margin_only_q{q_margin:.2f}"
        ),

        margin_threshold=(
            margin_threshold
        ),
    )


    grid_records.append(
        record
    )


# ============================================================
# 2. TREE UNCERTAINTY ONLY
# ============================================================

for q_std in uncertainty_quantiles:


    std_threshold = float(
        signal_df[
            "selected_tree_std"
        ]
        .quantile(
            q_std
        )
    )


    trigger = (
        signal_df[
            "selected_tree_std"
        ]
        >=
        std_threshold
    )


    record = evaluate_trigger(

        trigger_mask=trigger,

        rule_name=(
            f"tree_std_only_q{q_std:.2f}"
        ),

        uncertainty_threshold=(
            std_threshold
        ),
    )


    grid_records.append(
        record
    )


# ============================================================
# 3. MARGIN OR TREE UNCERTAINTY
# ============================================================

for q_margin in margin_quantiles:


    margin_threshold = float(
        signal_df[
            "predicted_quality_margin"
        ]
        .quantile(
            q_margin
        )
    )


    for q_std in uncertainty_quantiles:


        std_threshold = float(
            signal_df[
                "selected_tree_std"
            ]
            .quantile(
                q_std
            )
        )


        trigger = (

            (
                signal_df[
                    "predicted_quality_margin"
                ]
                <=
                margin_threshold
            )

            |

            (
                signal_df[
                    "selected_tree_std"
                ]
                >=
                std_threshold
            )
        )


        record = evaluate_trigger(

            trigger_mask=trigger,

            rule_name=(
                "margin_or_tree_std"
            ),

            margin_threshold=(
                margin_threshold
            ),

            uncertainty_threshold=(
                std_threshold
            ),
        )


        grid_records.append(
            record
        )


# ============================================================
# 4. MARGIN OR TREE UNCERTAINTY OR HIGH UTILIZATION
# ============================================================

for q_margin in [
    0.10,
    0.15,
    0.20,
]:


    margin_threshold = float(
        signal_df[
            "predicted_quality_margin"
        ]
        .quantile(
            q_margin
        )
    )


    for q_std in [
        0.80,
        0.85,
        0.90,
    ]:


        std_threshold = float(
            signal_df[
                "selected_tree_std"
            ]
            .quantile(
                q_std
            )
        )


        for q_util in utilization_quantiles:


            utilization_threshold = float(
                signal_df[
                    "selected_utilization"
                ]
                .quantile(
                    q_util
                )
            )


            trigger = (

                (
                    signal_df[
                        "predicted_quality_margin"
                    ]
                    <=
                    margin_threshold
                )

                |

                (
                    signal_df[
                        "selected_tree_std"
                    ]
                    >=
                    std_threshold
                )

                |

                (
                    signal_df[
                        "selected_utilization"
                    ]
                    >=
                    utilization_threshold
                )
            )


            record = evaluate_trigger(

                trigger_mask=trigger,

                rule_name=(
                    "margin_or_tree_std_or_utilization"
                ),

                margin_threshold=(
                    margin_threshold
                ),

                uncertainty_threshold=(
                    std_threshold
                ),

                utilization_threshold=(
                    utilization_threshold
                ),
            )


            grid_records.append(
                record
            )


# ============================================================
# GRID DATAFRAME
# ============================================================

grid_df = pd.DataFrame(
    grid_records
)


# ============================================================
# FEASIBLE LOW-FREQUENCY RULES
# ============================================================

feasible_df = grid_df[
    grid_df[
        "trigger_rate_percent"
    ]
    <=
    MAX_TRIGGER_RATE_PERCENT
].copy()


# ============================================================
# PARETO FILTER
# ============================================================
#
# A rule is dominated if another feasible rule:
#
#   - has equal/lower trigger rate
#   - has equal/higher high-regret capture
#   - has equal/higher exact-miss capture
#
# and is strictly better in at least one dimension.
# ============================================================

pareto_indices = []


for idx, row in feasible_df.iterrows():


    dominated = False


    for other_idx, other in feasible_df.iterrows():


        if idx == other_idx:
            continue


        no_more_cost = (
            other[
                "trigger_rate_percent"
            ]
            <=
            row[
                "trigger_rate_percent"
            ]
        )


        no_worse_regret = (
            other[
                "high_regret_capture_percent"
            ]
            >=
            row[
                "high_regret_capture_percent"
            ]
        )


        no_worse_miss = (
            other[
                "exact_miss_capture_percent"
            ]
            >=
            row[
                "exact_miss_capture_percent"
            ]
        )


        strictly_better = (

            other[
                "trigger_rate_percent"
            ]
            <
            row[
                "trigger_rate_percent"
            ]

            or

            other[
                "high_regret_capture_percent"
            ]
            >
            row[
                "high_regret_capture_percent"
            ]

            or

            other[
                "exact_miss_capture_percent"
            ]
            >
            row[
                "exact_miss_capture_percent"
            ]
        )


        if (
            no_more_cost
            and
            no_worse_regret
            and
            no_worse_miss
            and
            strictly_better
        ):

            dominated = True

            break


    if not dominated:

        pareto_indices.append(
            idx
        )


pareto_df = (
    feasible_df.loc[
        pareto_indices
    ]
    .copy()
)


# ============================================================
# PROVISIONAL RULE SELECTION
# ============================================================
#
# We do NOT build an arbitrary weighted score.
#
# Among rules using <=20% of decisions:
#
# Primary:
#     maximize capture of high-regret decisions
#
# Secondary:
#     maximize exact-best-miss capture
#
# Tertiary:
#     minimize trigger rate
#
# This creates a transparent lexicographic selection rule.
# ============================================================

if len(
    feasible_df
) == 0:

    raise RuntimeError(
        "\nNo trigger configuration satisfied the "
        "maximum trigger-rate constraint."
    )


ranked_df = feasible_df.sort_values(

    by=[
        "high_regret_capture_percent",
        "exact_miss_capture_percent",
        "trigger_rate_percent",
    ],

    ascending=[
        False,
        False,
        True,
    ],
)


provisional_best = ranked_df.iloc[
    0
]


# ============================================================
# STAGE ANALYSIS
# ============================================================

stage_rows = []


for component_index in range(
    1,
    6,
):


    subset = signal_df[
        signal_df[
            "component_index"
        ]
        ==
        component_index
    ]


    stage_rows.append(
        {

            "component_index":
                component_index,

            "n_decisions":
                len(
                    subset
                ),

            "mean_margin":
                subset[
                    "predicted_quality_margin"
                ]
                .mean(),

            "median_margin":
                subset[
                    "predicted_quality_margin"
                ]
                .median(),

            "mean_tree_std":
                subset[
                    "selected_tree_std"
                ]
                .mean(),

            "median_tree_std":
                subset[
                    "selected_tree_std"
                ]
                .median(),

            "mean_utilization":
                subset[
                    "selected_utilization"
                ]
                .mean(),

            "exact_best_miss_rate_percent":
                subset[
                    "exact_best_miss"
                ]
                .mean()
                *
                100.0,

            "high_regret_rate_percent":
                subset[
                    "high_regret"
                ]
                .mean()
                *
                100.0,

            "mean_actual_regret":
                subset[
                    "actual_regret"
                ]
                .mean(),
        }
    )


stage_df = pd.DataFrame(
    stage_rows
)


# ============================================================
# SIMPLE SIGNAL CORRELATION
# ============================================================
#
# Correlations are descriptive only.
# ============================================================

correlation_df = signal_df[
    [
        "predicted_quality_margin",
        "normalized_quality_margin",
        "selected_tree_std",
        "selected_tree_iqr",
        "selected_tree_range",
        "selected_utilization",
        "actual_regret",
        "exact_best_miss",
    ]
].copy()


correlation_df[
    "exact_best_miss"
] = correlation_df[
    "exact_best_miss"
].astype(
    int
)


correlation_matrix = (
    correlation_df
    .corr()
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    {

        "metric": [

            "n_decisions",

            "high_regret_threshold_p90",

            "n_high_regret_decisions",

            "n_exact_best_misses",

            "max_trigger_rate_screen_percent",

            "provisional_trigger_rate_percent",

            "provisional_high_regret_capture_percent",

            "provisional_exact_miss_capture_percent",

            "provisional_high_regret_precision_percent",

            "provisional_exact_miss_precision_percent",

            "provisional_margin_threshold",

            "provisional_tree_std_threshold",

            "provisional_utilization_threshold",
        ],

        "value": [

            len(
                signal_df
            ),

            high_regret_threshold,

            n_high_regret,

            n_exact_misses,

            MAX_TRIGGER_RATE_PERCENT,

            provisional_best[
                "trigger_rate_percent"
            ],

            provisional_best[
                "high_regret_capture_percent"
            ],

            provisional_best[
                "exact_miss_capture_percent"
            ],

            provisional_best[
                "high_regret_precision_percent"
            ],

            provisional_best[
                "exact_miss_precision_percent"
            ],

            provisional_best[
                "margin_threshold"
            ],

            provisional_best[
                "uncertainty_threshold"
            ],

            provisional_best[
                "utilization_threshold"
            ],
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


signal_df.to_csv(
    SIGNAL_OUTPUT,
    index=False,
)


grid_df.to_csv(
    GRID_OUTPUT,
    index=False,
)


pareto_df.to_csv(
    PARETO_OUTPUT,
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


correlation_matrix.to_csv(
    "results/tables/"
    "v3_2_predecision_signal_correlations.csv"
)


# ============================================================
# PRINT RESULTS
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
    "PRE-DECISION SIGNAL SUMMARY"
)

print(
    "============================================================"
)


signal_summary_columns = [
    "predicted_quality_margin",
    "selected_tree_std",
    "selected_utilization",
    "actual_regret",
]


print(
    "\n"
    +
    signal_df[
        signal_summary_columns
    ]
    .describe(
        percentiles=[
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
    .round(
        6
    )
    .to_string()
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "PROVISIONAL SELECTIVE-BO TRIGGER"
)

print(
    "------------------------------------------------------------"
)


print(
    f"\nRule type:"
)

print(
    provisional_best[
        "rule_name"
    ]
)


print(
    f"\nTrigger rate                : "
    f"{provisional_best['trigger_rate_percent']:.2f}%"
)


print(
    f"High-regret capture         : "
    f"{provisional_best['high_regret_capture_percent']:.2f}%"
)


print(
    f"Exact-best-miss capture     : "
    f"{provisional_best['exact_miss_capture_percent']:.2f}%"
)


print(
    f"High-regret precision       : "
    f"{provisional_best['high_regret_precision_percent']:.2f}%"
)


print(
    f"Exact-miss precision        : "
    f"{provisional_best['exact_miss_precision_percent']:.2f}%"
)


print(
    f"\nMargin threshold            : "
    f"{provisional_best['margin_threshold']}"
)


print(
    f"Tree-std threshold          : "
    f"{provisional_best['uncertainty_threshold']}"
)


print(
    f"Utilization threshold       : "
    f"{provisional_best['utilization_threshold']}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "TOP PARETO TRIGGER RULES"
)

print(
    "------------------------------------------------------------\n"
)


pareto_display = (
    pareto_df[
        [
            "rule_name",
            "trigger_rate_percent",
            "high_regret_capture_percent",
            "exact_miss_capture_percent",
            "high_regret_precision_percent",
            "exact_miss_precision_percent",
            "margin_threshold",
            "uncertainty_threshold",
            "utilization_threshold",
        ]
    ]
    .sort_values(
        [
            "high_regret_capture_percent",
            "exact_miss_capture_percent",
            "trigger_rate_percent",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )
    .head(
        15
    )
)


print(
    pareto_display
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "STAGE-WISE PRE-DECISION SIGNALS"
)

print(
    "------------------------------------------------------------\n"
)


print(
    stage_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ============================================================
# SCIENTIFIC DECISION
# ============================================================

trigger_rate = float(
    provisional_best[
        "trigger_rate_percent"
    ]
)


high_regret_capture = float(
    provisional_best[
        "high_regret_capture_percent"
    ]
)


exact_miss_capture = float(
    provisional_best[
        "exact_miss_capture_percent"
    ]
)


print(
    "\n"
    "============================================================"
)

print(
    "SELECTIVE-BO TRIGGER CALIBRATION VERDICT"
)

print(
    "============================================================"
)


if (
    trigger_rate
    <=
    20.0
    and
    high_regret_capture
    >=
    50.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "A low-frequency pre-decision ambiguity trigger "
        "captures a meaningful share of difficult decisions."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "Selective Bayesian refinement is technically "
        "worth testing as an online controller."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Run a complete sequential selective-BO controller "
        "using ONLY these pre-decision signals."
    )


elif (
    trigger_rate
    <=
    20.0
    and
    (
        high_regret_capture
        >=
        30.0
        or
        exact_miss_capture
        >=
        30.0
    )
):


    print(
        "\nRESULT B:"
    )


    print(
        "The pre-decision signals contain some useful "
        "information, but discrimination is moderate."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Test the best trigger cautiously and compare against "
        "always-structured and always-BO baselines."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "The current pre-decision ambiguity signals do not "
        "reliably isolate difficult decisions."
    )


    print(
        "\nINTERPRETATION:"
    )


    print(
        "Do NOT force a selective-BO architecture merely "
        "because it sounds attractive."
    )


    print(
        "\nNEXT STEP:"
    )


    print(
        "Retain structured search as the operational controller "
        "and treat BO as an ablation / optional refinement."
    )


print(
    "\nIMPORTANT:"
)


print(
    "This calibration used post-hoc labels only for evaluating "
    "the trigger."
)


print(
    "The proposed online signals themselves require no "
    "simulator truth."
)


print(
    "\nSaved:"
)


print(
    SIGNAL_OUTPUT
)

print(
    GRID_OUTPUT
)

print(
    PARETO_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print(
    STAGE_OUTPUT
)


print(
    f"\nRuntime: "
    f"{runtime:.2f} sec"
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 SELECTIVE-BO TRIGGER CALIBRATION COMPLETED"
)

print(
    "============================================================"
)