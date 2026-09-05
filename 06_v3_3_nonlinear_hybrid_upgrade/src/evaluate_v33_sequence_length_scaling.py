"""
V3.3 sequence-length scaling audit.

Purpose
-------
Evaluate whether the already-trained and already-locked hybrid controller

    LSQ_PLUS_ML_ALL

retains its advantage over deterministic LSQ when sequential assembly
length changes from:

    K = 5
to
    K = 10

No retraining, controller tuning, architecture selection, or
hyperparameter modification is performed.

Scientific interpretation
-------------------------
Absolute final quality at K=5 and K=10 should not be compared as if
lower K were intrinsically a better method.

The primary scaling evidence is:

1. hybrid improvement relative to LSQ,
2. hybrid-vs-LSQ assembly win rate,
3. consistency of the first five stages,
4. behavior in stages 6-10.

The same generated 10-component assemblies are used for both sequence
lengths. K=5 is the exact first-five-component prefix of K=10.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear

from nonlinear_assembly_model import (
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
    correction_profile_nonlinear,
    calculate_quality_metrics,
)

from v33_config import (
    PROFILE_LENGTH_MM,
    N_PROFILE_POINTS,
    Z_ADJ_LIMIT_MM,
    THETA_ADJ_LIMIT_DEG,
    LOCATOR_OFFSET_LIMIT_MM,
)


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

MODEL_FILE = (
    V33_ROOT
    / "models"
    / "residual_learning_full"
    / "v33_residual_rf_full.joblib"
)

FEATURE_FILE = (
    V33_ROOT
    / "data"
    / "residual_learning_full"
    / "v33_residual_learning_full_observable_features.txt"
)

RESULTS_DIR = (
    V33_ROOT
    / "results"
    / "sequence_length_scaling"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ASSEMBLY_FILE = (
    RESULTS_DIR
    / "v33_sequence_length_scaling_assembly_results.csv"
)

STAGE_FILE = (
    RESULTS_DIR
    / "v33_sequence_length_scaling_stage_results.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "v33_sequence_length_scaling_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTHS = [
    5,
    10,
]

MAX_SEQUENCE_LENGTH = 10

N_ASSEMBLIES = 200

BASE_SEED = 20260915

NONLINEARITY_STRENGTH = 1.0

FIXTURE_DRIFT_STD_MM = 0.015


# ============================================================
# METHODS
# ============================================================

METHOD_ZERO = "ZERO"

METHOD_LSQ = "LSQ"

METHOD_HYBRID = "LOCKED_HYBRID"


METHODS = [
    METHOD_ZERO,
    METHOD_LSQ,
    METHOD_HYBRID,
]


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_FILE}"
    )


if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"\nFeature file not found:\n{FEATURE_FILE}"
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


if getattr(
    model,
    "n_features_in_",
    None,
) != 26:

    raise RuntimeError(
        "Expected residual model with 26 input features."
    )


if getattr(
    model,
    "n_outputs_",
    None,
) != 3:

    raise RuntimeError(
        "Expected residual model with 3 outputs."
    )


# ============================================================
# PROFILE AXIS
# ============================================================

s = np.linspace(
    0.0,
    1.0,
    N_PROFILE_POINTS,
)


# ============================================================
# EXACT LSQ BASIS
# ============================================================

Z_BASIS = np.ones_like(
    s,
    dtype=float,
)


THETA_BASIS = (
    PROFILE_LENGTH_MM
    *
    (
        s
        -
        0.5
    )
)


LOCATOR_SIGMA = 0.12


LOCATOR_BASIS = np.exp(
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
        LOCATOR_SIGMA
        ** 2
    )
)


CORRECTION_BASIS = np.column_stack(
    [
        Z_BASIS,
        THETA_BASIS,
        LOCATOR_BASIS,
    ]
)


THETA_TAN_LIMIT = float(
    np.tan(
        np.deg2rad(
            THETA_ADJ_LIMIT_DEG
        )
    )
)


LOWER_BOUNDS = np.array(
    [
        -Z_ADJ_LIMIT_MM,
        -THETA_TAN_LIMIT,
        -LOCATOR_OFFSET_LIMIT_MM,
    ],
    dtype=float,
)


UPPER_BOUNDS = np.array(
    [
        Z_ADJ_LIMIT_MM,
        THETA_TAN_LIMIT,
        LOCATOR_OFFSET_LIMIT_MM,
    ],
    dtype=float,
)


# ============================================================
# PROFILE FEATURE LOCATIONS
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


# ============================================================
# COMPONENT GENERATION
# ============================================================

def generate_component_parameters(
    rng,
):

    return {
        "offset_mm":
            rng.uniform(
                -0.60,
                0.60,
            ),

        "tilt_deg":
            rng.uniform(
                -0.10,
                0.10,
            ),

        "bend_mm":
            rng.uniform(
                -0.45,
                0.45,
            ),

        "waviness_mm":
            rng.uniform(
                -0.30,
                0.30,
            ),

        "twist_mm":
            rng.uniform(
                -0.25,
                0.25,
            ),

        "local_bump_mm":
            rng.uniform(
                -0.30,
                0.30,
            ),
    }


def build_component_profile(
    parameters,
):

    return (
        deviation_offset(
            parameters[
                "offset_mm"
            ]
        )
        +
        deviation_tilt(
            parameters[
                "tilt_deg"
            ]
        )
        +
        deviation_bend(
            parameters[
                "bend_mm"
            ]
        )
        +
        deviation_waviness(
            parameters[
                "waviness_mm"
            ],
            waves=3,
        )
        +
        deviation_twist(
            parameters[
                "twist_mm"
            ]
        )
        +
        deviation_local_bump(
            parameters[
                "local_bump_mm"
            ]
        )
    )


# ============================================================
# GENERATE COMMON 10-COMPONENT CASES
# ============================================================

def generate_cases():
    """
    Generate every assembly to K=10.

    K=5 uses the exact first five components of the same assembly.
    """

    rng = np.random.default_rng(
        BASE_SEED
    )


    cases = []


    for assembly_index in range(
        1,
        N_ASSEMBLIES
        +
        1
    ):

        components = []


        for component_index in range(
            1,
            MAX_SEQUENCE_LENGTH
            +
            1
        ):

            parameters = (
                generate_component_parameters(
                    rng
                )
            )


            component_profile = (
                build_component_profile(
                    parameters
                )
            )


            fixture_drift = rng.normal(
                loc=0.0,
                scale=FIXTURE_DRIFT_STD_MM,
                size=N_PROFILE_POINTS,
            )


            components.append(
                {
                    "component_index":
                        component_index,

                    "component_profile":
                        component_profile,

                    "fixture_drift":
                        fixture_drift,
                }
            )


        cases.append(
            {
                "assembly_index":
                    assembly_index,

                "components":
                    components,
            }
        )


    return cases


# ============================================================
# LSQ
# ============================================================

def deterministic_lsq_correction(
    state,
):

    result = lsq_linear(
        CORRECTION_BASIS,
        -state,
        bounds=(
            LOWER_BOUNDS,
            UPPER_BOUNDS,
        ),
        method="trf",
        lsmr_tol="auto",
    )


    z_adj = float(
        result.x[0]
    )


    tan_theta = float(
        result.x[1]
    )


    theta_adj = float(
        np.rad2deg(
            np.arctan(
                tan_theta
            )
        )
    )


    theta_adj = float(
        np.clip(
            theta_adj,
            -THETA_ADJ_LIMIT_DEG,
            THETA_ADJ_LIMIT_DEG,
        )
    )


    locator = float(
        result.x[2]
    )


    return np.array(
        [
            z_adj,
            theta_adj,
            locator,
        ],
        dtype=float,
    )


# ============================================================
# UTILIZATION
# ============================================================

def calculate_utilization(
    point,
):

    return float(
        max(
            abs(
                point[0]
            )
            /
            Z_ADJ_LIMIT_MM,

            abs(
                point[1]
            )
            /
            THETA_ADJ_LIMIT_DEG,

            abs(
                point[2]
            )
            /
            LOCATOR_OFFSET_LIMIT_MM,
        )
    )


# ============================================================
# OBSERVABLE FEATURES
# ============================================================

def calculate_features(
    pre_state,
    component_profile,
    component_index,
    lsq_point,
):

    metrics = calculate_quality_metrics(
        pre_state
    )


    signed_mean = float(
        np.mean(
            pre_state
        )
    )


    signed_end_difference = float(
        pre_state[-1]
        -
        pre_state[0]
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


    component_rms = float(
        np.sqrt(
            np.mean(
                component_profile
                ** 2
            )
        )
    )


    component_parallelism = float(
        abs(
            component_profile[-1]
            -
            component_profile[0]
        )
    )


    record = {
        "state_mean_gap":
            float(
                metrics[
                    "mean_gap"
                ]
            ),

        "state_max_gap":
            float(
                metrics[
                    "max_gap"
                ]
            ),

        "state_parallelism":
            float(
                metrics[
                    "parallelism_error"
                ]
            ),

        "state_rms":
            float(
                metrics[
                    "rms_deviation"
                ]
            ),

        "state_quality":
            float(
                metrics[
                    "quality_score"
                ]
            ),

        "state_signed_mean":
            signed_mean,

        "state_signed_end_difference":
            signed_end_difference,

        "state_estimated_angle_deg":
            estimated_angle_deg,

        "component_profile_rms":
            component_rms,

        "component_parallelism":
            component_parallelism,

        "lsq_z_adj_mm":
            float(
                lsq_point[0]
            ),

        "lsq_theta_adj_deg":
            float(
                lsq_point[1]
            ),

        "lsq_locator_offset_mm":
            float(
                lsq_point[2]
            ),

        "lsq_utilization":
            calculate_utilization(
                lsq_point
            ),

        "component_index":
            int(
                component_index
            ),
    }


    for index in PROFILE_SAMPLE_INDICES:

        record[
            f"state_profile_p{index:03d}"
        ] = float(
            pre_state[
                index
            ]
        )


    return record


# ============================================================
# ML PREDICTION
# ============================================================

def predict_residual(
    pre_state,
    component_profile,
    component_index,
    lsq_point,
):

    feature_record = calculate_features(
        pre_state=pre_state,
        component_profile=component_profile,
        component_index=component_index,
        lsq_point=lsq_point,
    )


    X = np.array(
        [
            [
                feature_record[
                    column
                ]
                for column in FEATURE_COLUMNS
            ]
        ],
        dtype=float,
    )


    return np.asarray(
        model.predict(
            X
        )[0],
        dtype=float,
    )


# ============================================================
# CORRECTION CLIPPING
# ============================================================

def clip_point(
    point,
):

    point = np.asarray(
        point,
        dtype=float,
    ).copy()


    point[0] = np.clip(
        point[0],
        -Z_ADJ_LIMIT_MM,
        Z_ADJ_LIMIT_MM,
    )


    point[1] = np.clip(
        point[1],
        -THETA_ADJ_LIMIT_DEG,
        THETA_ADJ_LIMIT_DEG,
    )


    point[2] = np.clip(
        point[2],
        -LOCATOR_OFFSET_LIMIT_MM,
        LOCATOR_OFFSET_LIMIT_MM,
    )


    return point


# ============================================================
# TRUE NONLINEAR RESPONSE
# ============================================================

def apply_true_correction(
    pre_state,
    point,
):

    correction = correction_profile_nonlinear(
        previous_state=pre_state,
        z_adj_mm=float(
            point[0]
        ),
        theta_adj_deg=float(
            point[1]
        ),
        locator_offset_mm=float(
            point[2]
        ),
        nonlinearity_strength=NONLINEARITY_STRENGTH,
    )


    return (
        pre_state
        +
        correction
    )


# ============================================================
# RUN ONE SEQUENCE LENGTH
# ============================================================

def run_sequence_length(
    cases,
    sequence_length,
):

    assembly_rows = []

    stage_rows = []


    for assembly_case in cases:

        zero_state = create_initial_state()

        lsq_state = create_initial_state()

        hybrid_state = create_initial_state()


        for component_record in (
            assembly_case[
                "components"
            ][
                :sequence_length
            ]
        ):

            component_index = (
                component_record[
                    "component_index"
                ]
            )


            component_profile = (
                component_record[
                    "component_profile"
                ]
            )


            fixture_drift = (
                component_record[
                    "fixture_drift"
                ]
            )


            # =================================================
            # ZERO
            # =================================================

            zero_pre_state = (
                zero_state
                +
                component_profile
                +
                fixture_drift
            )


            zero_state = zero_pre_state


            # =================================================
            # LSQ
            # =================================================

            lsq_pre_state = (
                lsq_state
                +
                component_profile
                +
                fixture_drift
            )


            lsq_point = (
                deterministic_lsq_correction(
                    lsq_pre_state
                )
            )


            lsq_state = (
                apply_true_correction(
                    lsq_pre_state,
                    lsq_point,
                )
            )


            # =================================================
            # LOCKED HYBRID
            # =================================================

            hybrid_pre_state = (
                hybrid_state
                +
                component_profile
                +
                fixture_drift
            )


            hybrid_lsq_point = (
                deterministic_lsq_correction(
                    hybrid_pre_state
                )
            )


            residual = predict_residual(
                pre_state=hybrid_pre_state,
                component_profile=component_profile,
                component_index=component_index,
                lsq_point=hybrid_lsq_point,
            )


            hybrid_point = clip_point(
                hybrid_lsq_point
                +
                residual
            )


            hybrid_state = (
                apply_true_correction(
                    hybrid_pre_state,
                    hybrid_point,
                )
            )


            # =================================================
            # STAGE RESULTS
            # =================================================

            lsq_stage_quality = float(
                calculate_quality_metrics(
                    lsq_state
                )[
                    "quality_score"
                ]
            )


            hybrid_stage_quality = float(
                calculate_quality_metrics(
                    hybrid_state
                )[
                    "quality_score"
                ]
            )


            stage_rows.append(
                {
                    "sequence_length":
                        sequence_length,

                    "assembly_index":
                        assembly_case[
                            "assembly_index"
                        ],

                    "component_index":
                        component_index,

                    "lsq_quality":
                        lsq_stage_quality,

                    "hybrid_quality":
                        hybrid_stage_quality,

                    "hybrid_minus_lsq":
                        hybrid_stage_quality
                        -
                        lsq_stage_quality,
                }
            )


        # =====================================================
        # FINAL RESULTS
        # =====================================================

        zero_final = float(
            calculate_quality_metrics(
                zero_state
            )[
                "quality_score"
            ]
        )


        lsq_final = float(
            calculate_quality_metrics(
                lsq_state
            )[
                "quality_score"
            ]
        )


        hybrid_final = float(
            calculate_quality_metrics(
                hybrid_state
            )[
                "quality_score"
            ]
        )


        assembly_rows.append(
            {
                "sequence_length":
                    sequence_length,

                "assembly_index":
                    assembly_case[
                        "assembly_index"
                    ],

                "zero_final_quality":
                    zero_final,

                "lsq_final_quality":
                    lsq_final,

                "hybrid_final_quality":
                    hybrid_final,

                "hybrid_minus_lsq":
                    hybrid_final
                    -
                    lsq_final,

                "hybrid_beats_lsq":
                    bool(
                        hybrid_final
                        <
                        lsq_final
                        -
                        1e-12
                    ),
            }
        )


    return (
        assembly_rows,
        stage_rows,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 SEQUENCE-LENGTH SCALING AUDIT"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies per sequence length : {N_ASSEMBLIES}"
)

print(
    f"Sequence lengths               : {SEQUENCE_LENGTHS}"
)

print(
    f"Nonlinearity                   : "
    f"{NONLINEARITY_STRENGTH}"
)


cases = generate_cases()


all_assembly_rows = []

all_stage_rows = []


for sequence_length in SEQUENCE_LENGTHS:

    print(
        f"\nRunning K={sequence_length}"
    )


    assembly_rows, stage_rows = (
        run_sequence_length(
            cases=cases,
            sequence_length=sequence_length,
        )
    )


    all_assembly_rows.extend(
        assembly_rows
    )


    all_stage_rows.extend(
        stage_rows
    )


    print(
        f"Completed K={sequence_length}"
    )


assembly_df = pd.DataFrame(
    all_assembly_rows
)


stage_df = pd.DataFrame(
    all_stage_rows
)


assembly_df.to_csv(
    ASSEMBLY_FILE,
    index=False,
)


stage_df.to_csv(
    STAGE_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for sequence_length in SEQUENCE_LENGTHS:

    current = (
        assembly_df[
            assembly_df[
                "sequence_length"
            ]
            ==
            sequence_length
        ]
    )


    mean_zero = float(
        current[
            "zero_final_quality"
        ]
        .mean()
    )


    mean_lsq = float(
        current[
            "lsq_final_quality"
        ]
        .mean()
    )


    mean_hybrid = float(
        current[
            "hybrid_final_quality"
        ]
        .mean()
    )


    median_lsq = float(
        current[
            "lsq_final_quality"
        ]
        .median()
    )


    median_hybrid = float(
        current[
            "hybrid_final_quality"
        ]
        .median()
    )


    p95_lsq = float(
        current[
            "lsq_final_quality"
        ]
        .quantile(
            0.95
        )
    )


    p95_hybrid = float(
        current[
            "hybrid_final_quality"
        ]
        .quantile(
            0.95
        )
    )


    relative_improvement = float(
        100.0
        *
        (
            mean_lsq
            -
            mean_hybrid
        )
        /
        max(
            mean_lsq,
            1e-12,
        )
    )


    hybrid_win_percent = float(
        100.0
        *
        current[
            "hybrid_beats_lsq"
        ]
        .mean()
    )


    summary_rows.append(
        {
            "sequence_length":
                sequence_length,

            "assemblies":
                len(
                    current
                ),

            "mean_zero_quality":
                mean_zero,

            "mean_lsq_quality":
                mean_lsq,

            "mean_hybrid_quality":
                mean_hybrid,

            "median_lsq_quality":
                median_lsq,

            "median_hybrid_quality":
                median_hybrid,

            "p95_lsq_quality":
                p95_lsq,

            "p95_hybrid_quality":
                p95_hybrid,

            "mean_hybrid_minus_lsq":
                float(
                    current[
                        "hybrid_minus_lsq"
                    ]
                    .mean()
                ),

            "relative_hybrid_improvement_percent":
                relative_improvement,

            "hybrid_beats_lsq_percent":
                hybrid_win_percent,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# PREFIX CONSISTENCY
# ============================================================

stage5_k5 = (
    stage_df[
        (
            stage_df[
                "sequence_length"
            ]
            ==
            5
        )
        &
        (
            stage_df[
                "component_index"
            ]
            ==
            5
        )
    ]
    .sort_values(
        "assembly_index"
    )
)


stage5_k10 = (
    stage_df[
        (
            stage_df[
                "sequence_length"
            ]
            ==
            10
        )
        &
        (
            stage_df[
                "component_index"
            ]
            ==
            5
        )
    ]
    .sort_values(
        "assembly_index"
    )
)


prefix_lsq_difference = float(
    np.max(
        np.abs(
            stage5_k5[
                "lsq_quality"
            ]
            .to_numpy()
            -
            stage5_k10[
                "lsq_quality"
            ]
            .to_numpy()
        )
    )
)


prefix_hybrid_difference = float(
    np.max(
        np.abs(
            stage5_k5[
                "hybrid_quality"
            ]
            .to_numpy()
            -
            stage5_k10[
                "hybrid_quality"
            ]
            .to_numpy()
        )
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "SEQUENCE-LENGTH RESULTS"
)

print(
    "============================================================"
)


for _, row in summary_df.iterrows():

    print(
        f"\nK={int(row['sequence_length'])}"
    )


    print(
        f"  Zero mean quality        : "
        f"{row['mean_zero_quality']:.6f}"
    )


    print(
        f"  LSQ mean quality         : "
        f"{row['mean_lsq_quality']:.6f}"
    )


    print(
        f"  Hybrid mean quality      : "
        f"{row['mean_hybrid_quality']:.6f}"
    )


    print(
        f"  Hybrid - LSQ             : "
        f"{row['mean_hybrid_minus_lsq']:.6f}"
    )


    print(
        f"  Relative improvement     : "
        f"{row['relative_hybrid_improvement_percent']:.2f}%"
    )


    print(
        f"  Hybrid beats LSQ         : "
        f"{row['hybrid_beats_lsq_percent']:.2f}%"
    )


    print(
        f"  LSQ median               : "
        f"{row['median_lsq_quality']:.6f}"
    )


    print(
        f"  Hybrid median            : "
        f"{row['median_hybrid_quality']:.6f}"
    )


    print(
        f"  LSQ P95                  : "
        f"{row['p95_lsq_quality']:.6f}"
    )


    print(
        f"  Hybrid P95               : "
        f"{row['p95_hybrid_quality']:.6f}"
    )


print(
    "\n"
    "============================================================"
)

print(
    "PREFIX CONSISTENCY CHECK"
)

print(
    "============================================================"
)


print(
    f"\nMaximum LSQ stage-5 difference "
    f"between K=5 and K=10 runs:"
)


print(
    f"{prefix_lsq_difference:.16e}"
)


print(
    "\nMaximum hybrid stage-5 difference "
    "between K=5 and K=10 runs:"
)


print(
    f"{prefix_hybrid_difference:.16e}"
)


if (
    prefix_lsq_difference
    <=
    1e-12
    and
    prefix_hybrid_difference
    <=
    1e-12
):

    print(
        "\nPASS: K=5 is an exact prefix of the K=10 experiment."
    )


else:

    print(
        "\nFAIL: Prefix inconsistency detected."
    )


# ============================================================
# SCALING INTERPRETATION
# ============================================================

k5_row = (
    summary_df[
        summary_df[
            "sequence_length"
        ]
        ==
        5
    ]
    .iloc[0]
)


k10_row = (
    summary_df[
        summary_df[
            "sequence_length"
        ]
        ==
        10
    ]
    .iloc[0]
)


print(
    "\n"
    "============================================================"
)

print(
    "SEQUENCE-LENGTH SCALING INTERPRETATION"
)

print(
    "============================================================"
)


print(
    f"\nK=5 hybrid improvement  : "
    f"{k5_row['relative_hybrid_improvement_percent']:.2f}%"
)


print(
    f"K=10 hybrid improvement : "
    f"{k10_row['relative_hybrid_improvement_percent']:.2f}%"
)


print(
    f"\nK=5 hybrid win rate     : "
    f"{k5_row['hybrid_beats_lsq_percent']:.2f}%"
)


print(
    f"K=10 hybrid win rate    : "
    f"{k10_row['hybrid_beats_lsq_percent']:.2f}%"
)


if (
    k5_row[
        "relative_hybrid_improvement_percent"
    ]
    >
    0.0
    and
    k10_row[
        "relative_hybrid_improvement_percent"
    ]
    >
    0.0
):

    print(
        "\nPASS:"
    )


    print(
        "The locked hybrid retains a positive mean advantage "
        "over LSQ at both K=5 and K=10."
    )


else:

    print(
        "\nSCALING LIMITATION:"
    )


    print(
        "The hybrid advantage is not retained at both tested "
        "sequence lengths."
    )


print(
    "\nImportant:"
)


print(
    "Increasing sequence length is not interpreted as making "
    "the hybrid intrinsically better."
)


print(
    "The scaling evidence concerns persistence of the hybrid "
    "advantage under a longer sequential assembly."
)


print(
    "\nSaved assembly results:"
)

print(
    ASSEMBLY_FILE
)


print(
    "\nSaved stage results:"
)

print(
    STAGE_FILE
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
    "V3.3 SEQUENCE-LENGTH SCALING AUDIT COMPLETED"
)

print(
    "============================================================"
)