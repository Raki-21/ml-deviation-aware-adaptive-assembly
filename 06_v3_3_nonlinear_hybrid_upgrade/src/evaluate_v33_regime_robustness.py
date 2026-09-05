"""
V3.3 locked-hybrid regime and measurement-noise robustness audit.

Purpose
-------
Evaluate the already-trained and already-locked V3.3 controller:

    LSQ_PLUS_ML_ALL

across controlled changes in:

1. physical nonlinearity strength,
2. measurement/profile noise.

No retraining, architecture selection, or tuning is performed.

Methods
-------
ZERO
LSQ
LOCKED HYBRID = LSQ + predicted [delta_z, delta_theta, delta_locator]

Important
---------
Both LSQ and the hybrid receive the same noisy measurement of the
pre-correction state.

The true nonlinear simulator itself remains noise-free. Measurement
noise affects the information available to the controller, not the
underlying physical state.

This is a robustness/regime experiment, not another development set.
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
    N_COMPONENTS,
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
    / "regime_robustness"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ASSEMBLY_FILE = (
    RESULTS_DIR
    / "v33_regime_robustness_assembly_results.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "v33_regime_robustness_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = N_COMPONENTS

N_ASSEMBLIES = 150

BASE_SEED = 20260914

FIXTURE_DRIFT_STD_MM = 0.015


NONLINEARITY_LEVELS = [
    0.0,
    0.5,
    1.0,
]


MEASUREMENT_NOISE_LEVELS_MM = [
    0.0,
    0.01,
    0.03,
]


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
        "Residual RF does not expect 26 features."
    )


if getattr(
    model,
    "n_outputs_",
    None,
) != 3:

    raise RuntimeError(
        "Residual RF does not have 3 outputs."
    )


# ============================================================
# PROFILE AXIS + EXACT LSQ BASIS
# ============================================================

s = np.linspace(
    0.0,
    1.0,
    N_PROFILE_POINTS,
)


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
# PROFILE FEATURE SAMPLES
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
# GENERATE IDENTICAL TRUE ASSEMBLIES
# ============================================================

def generate_cases():
    """
    Generate true geometry once.

    Every regime configuration replays these exact same assemblies.
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
            SEQUENCE_LENGTH
            +
            1
        ):

            parameters = (
                generate_component_parameters(
                    rng
                )
            )


            profile = (
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
                        profile,

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
# EXACT LSQ
# ============================================================

def deterministic_lsq_correction(
    measured_state,
):

    result = lsq_linear(
        CORRECTION_BASIS,
        -measured_state,
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
    measured_pre_state,
    measured_component_profile,
    component_index,
    lsq_point,
):

    metrics = calculate_quality_metrics(
        measured_pre_state
    )


    signed_mean = float(
        np.mean(
            measured_pre_state
        )
    )


    signed_end_difference = float(
        measured_pre_state[-1]
        -
        measured_pre_state[0]
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
                measured_component_profile
                ** 2
            )
        )
    )


    component_parallelism = float(
        abs(
            measured_component_profile[-1]
            -
            measured_component_profile[0]
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
            measured_pre_state[
                index
            ]
        )


    return record


# ============================================================
# ML RESIDUAL
# ============================================================

def predict_residual(
    measured_pre_state,
    measured_component_profile,
    component_index,
    lsq_point,
):

    feature_record = calculate_features(
        measured_pre_state=measured_pre_state,
        measured_component_profile=measured_component_profile,
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
# TRUE PHYSICAL RESPONSE
# ============================================================

def apply_true_correction(
    true_pre_state,
    point,
    nonlinearity_strength,
):

    correction = (
        correction_profile_nonlinear(
            previous_state=true_pre_state,
            z_adj_mm=float(
                point[0]
            ),
            theta_adj_deg=float(
                point[1]
            ),
            locator_offset_mm=float(
                point[2]
            ),
            nonlinearity_strength=nonlinearity_strength,
        )
    )


    return (
        true_pre_state
        +
        correction
    )


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_configuration(
    cases,
    nonlinearity_strength,
    measurement_noise_mm,
):
    """
    Evaluate all three methods on one regime.

    Measurement noise is generated deterministically from the
    configuration and assembly/component identifiers.

    LSQ and hybrid receive the same measurement realization.
    """

    method_final_rows = []


    for assembly_case in cases:

        states = {
            METHOD_ZERO:
                create_initial_state(),

            METHOD_LSQ:
                create_initial_state(),

            METHOD_HYBRID:
                create_initial_state(),
        }


        for component_record in (
            assembly_case[
                "components"
            ]
        ):

            component_index = (
                component_record[
                    "component_index"
                ]
            )


            true_component = (
                component_record[
                    "component_profile"
                ]
            )


            fixture_drift = (
                component_record[
                    "fixture_drift"
                ]
            )


            noise_seed = (
                BASE_SEED
                +
                int(
                    round(
                        nonlinearity_strength
                        *
                        1000
                    )
                )
                *
                100000
                +
                int(
                    round(
                        measurement_noise_mm
                        *
                        100000
                    )
                )
                *
                1000
                +
                assembly_case[
                    "assembly_index"
                ]
                *
                10
                +
                component_index
            )


            noise_rng = np.random.default_rng(
                noise_seed
            )


            state_noise = noise_rng.normal(
                loc=0.0,
                scale=measurement_noise_mm,
                size=N_PROFILE_POINTS,
            )


            component_noise = noise_rng.normal(
                loc=0.0,
                scale=measurement_noise_mm,
                size=N_PROFILE_POINTS,
            )


            # ------------------------------------------------
            # ZERO
            # ------------------------------------------------

            zero_pre_state = (
                states[
                    METHOD_ZERO
                ]
                +
                true_component
                +
                fixture_drift
            )


            states[
                METHOD_ZERO
            ] = zero_pre_state


            # ------------------------------------------------
            # LSQ
            # ------------------------------------------------

            lsq_true_pre_state = (
                states[
                    METHOD_LSQ
                ]
                +
                true_component
                +
                fixture_drift
            )


            lsq_measured_state = (
                lsq_true_pre_state
                +
                state_noise
            )


            lsq_point = (
                deterministic_lsq_correction(
                    lsq_measured_state
                )
            )


            states[
                METHOD_LSQ
            ] = apply_true_correction(
                true_pre_state=lsq_true_pre_state,
                point=lsq_point,
                nonlinearity_strength=nonlinearity_strength,
            )


            # ------------------------------------------------
            # LOCKED HYBRID
            # ------------------------------------------------

            hybrid_true_pre_state = (
                states[
                    METHOD_HYBRID
                ]
                +
                true_component
                +
                fixture_drift
            )


            hybrid_measured_state = (
                hybrid_true_pre_state
                +
                state_noise
            )


            measured_component = (
                true_component
                +
                component_noise
            )


            hybrid_lsq_point = (
                deterministic_lsq_correction(
                    hybrid_measured_state
                )
            )


            predicted_residual = (
                predict_residual(
                    measured_pre_state=hybrid_measured_state,
                    measured_component_profile=measured_component,
                    component_index=component_index,
                    lsq_point=hybrid_lsq_point,
                )
            )


            hybrid_point = clip_point(
                hybrid_lsq_point
                +
                predicted_residual
            )


            states[
                METHOD_HYBRID
            ] = apply_true_correction(
                true_pre_state=hybrid_true_pre_state,
                point=hybrid_point,
                nonlinearity_strength=nonlinearity_strength,
            )


        # ----------------------------------------------------
        # FINAL ASSEMBLY
        # ----------------------------------------------------

        for method in METHODS:

            final_quality = float(
                calculate_quality_metrics(
                    states[
                        method
                    ]
                )[
                    "quality_score"
                ]
            )


            method_final_rows.append(
                {
                    "assembly_index":
                        assembly_case[
                            "assembly_index"
                        ],

                    "nonlinearity_strength":
                        nonlinearity_strength,

                    "measurement_noise_mm":
                        measurement_noise_mm,

                    "method":
                        method,

                    "final_quality":
                        final_quality,
                }
            )


    return method_final_rows


# ============================================================
# GENERATE CASES
# ============================================================

cases = generate_cases()


# ============================================================
# EXPERIMENT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 LOCKED-HYBRID REGIME ROBUSTNESS"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies/configuration : {N_ASSEMBLIES}"
)

print(
    f"Sequence length          : {SEQUENCE_LENGTH}"
)

print(
    f"Nonlinearity levels      : {NONLINEARITY_LEVELS}"
)

print(
    f"Measurement noise levels : "
    f"{MEASUREMENT_NOISE_LEVELS_MM}"
)


all_rows = []


for alpha in NONLINEARITY_LEVELS:

    for noise in MEASUREMENT_NOISE_LEVELS_MM:

        print(
            f"\nRunning alpha={alpha:.2f}, "
            f"noise={noise:.3f} mm"
        )


        configuration_rows = (
            run_configuration(
                cases=cases,
                nonlinearity_strength=alpha,
                measurement_noise_mm=noise,
            )
        )


        all_rows.extend(
            configuration_rows
        )


        print(
            "Completed."
        )


assembly_df = pd.DataFrame(
    all_rows
)


assembly_df.to_csv(
    ASSEMBLY_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for alpha in NONLINEARITY_LEVELS:

    for noise in MEASUREMENT_NOISE_LEVELS_MM:

        config_df = (
            assembly_df[
                (
                    assembly_df[
                        "nonlinearity_strength"
                    ]
                    ==
                    alpha
                )
                &
                (
                    assembly_df[
                        "measurement_noise_mm"
                    ]
                    ==
                    noise
                )
            ]
        )


        pivot = config_df.pivot(
            index="assembly_index",
            columns="method",
            values="final_quality",
        )


        zero = (
            pivot[
                METHOD_ZERO
            ]
            .to_numpy(
                dtype=float
            )
        )


        lsq = (
            pivot[
                METHOD_LSQ
            ]
            .to_numpy(
                dtype=float
            )
        )


        hybrid = (
            pivot[
                METHOD_HYBRID
            ]
            .to_numpy(
                dtype=float
            )
        )


        difference = (
            hybrid
            -
            lsq
        )


        mean_zero = float(
            np.mean(
                zero
            )
        )


        mean_lsq = float(
            np.mean(
                lsq
            )
        )


        mean_hybrid = float(
            np.mean(
                hybrid
            )
        )


        relative_hybrid_improvement = float(
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
            np.mean(
                difference
                <
                -1e-12
            )
        )


        summary_rows.append(
            {
                "nonlinearity_strength":
                    alpha,

                "measurement_noise_mm":
                    noise,

                "assemblies":
                    len(
                        pivot
                    ),

                "mean_zero_quality":
                    mean_zero,

                "mean_lsq_quality":
                    mean_lsq,

                "mean_locked_hybrid_quality":
                    mean_hybrid,

                "mean_hybrid_minus_lsq":
                    float(
                        np.mean(
                            difference
                        )
                    ),

                "relative_hybrid_improvement_percent":
                    relative_hybrid_improvement,

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
# PRINT REGIME MAP
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 REGIME MAP"
)

print(
    "============================================================"
)


for _, row in summary_df.iterrows():

    print(
        "\n"
        f"alpha={row['nonlinearity_strength']:.2f}, "
        f"noise={row['measurement_noise_mm']:.3f} mm"
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
        f"{row['mean_locked_hybrid_quality']:.6f}"
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


# ============================================================
# REGIME INTERPRETATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "REGIME ROBUSTNESS INTERPRETATION"
)

print(
    "============================================================"
)


for alpha in NONLINEARITY_LEVELS:

    alpha_df = (
        summary_df[
            summary_df[
                "nonlinearity_strength"
            ]
            ==
            alpha
        ]
    )


    clean_row = (
        alpha_df[
            alpha_df[
                "measurement_noise_mm"
            ]
            ==
            0.0
        ]
        .iloc[0]
    )


    print(
        f"\nalpha={alpha:.2f} clean-measurement result:"
    )


    print(
        f"  Hybrid improvement vs LSQ: "
        f"{clean_row['relative_hybrid_improvement_percent']:.2f}%"
    )


    print(
        f"  Hybrid win rate          : "
        f"{clean_row['hybrid_beats_lsq_percent']:.2f}%"
    )


print(
    "\nImportant:"
)

print(
    "The residual RF was trained at nonlinearity=1.0 using "
    "clean simulated measurements."
)

print(
    "Performance at other nonlinearities and noise levels is "
    "therefore a robustness/generalization result, not training-fit evidence."
)


print(
    "\nSaved assembly results:"
)

print(
    ASSEMBLY_FILE
)


print(
    "\nSaved regime summary:"
)

print(
    SUMMARY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 REGIME ROBUSTNESS AUDIT COMPLETED"
)

print(
    "============================================================"
)