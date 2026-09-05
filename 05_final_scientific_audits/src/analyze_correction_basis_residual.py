"""
Correction-basis residual audit.

Purpose
-------
Explain why the deterministic least-squares baseline performs strongly
in the current sequential assembly model.

The script replays the same independent 300-assembly validation
population used by the deterministic engineering baseline.

No ML model is retrained.
No controller is tuned.
No correction bounds are changed.
The frozen V3.2 implementation is not modified.

At every sequential decision the script measures:

1. How much of the current uncorrected profile can be represented by
   the available correction basis.
2. How much additional residual is caused by correction bounds.
3. Whether Structured-20 becomes relatively more competitive when the
   geometry is poorly represented by the analytical correction basis.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import lsq_linear


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AUDIT_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_ROOT = os.path.dirname(
    AUDIT_DIR
)

V32_DIR = os.path.join(
    PROJECT_ROOT,
    "03_v3_2_sequential_multicomponent",
)

V32_SRC_DIR = os.path.join(
    V32_DIR,
    "src",
)

if V32_SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        V32_SRC_DIR,
    )


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
# INPUT FILES
# ============================================================

COMPONENT_FILE = os.path.join(
    V32_DIR,
    "04_correctability_extension",
    "results",
    "prediction",
    "tables",
    "correctability_prediction_dataset.csv",
)

CORRECTABILITY_FILE = os.path.join(
    V32_DIR,
    "04_correctability_extension",
    "results",
    "tables",
    "component_correctability_analysis.csv",
)

DETERMINISTIC_REFERENCE_FILE = os.path.join(
    AUDIT_DIR,
    "results",
    "deterministic_baseline",
    "deterministic_assembly_results.csv",
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    AUDIT_DIR,
    "results",
    "correction_basis_residual",
)

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)


COMPONENT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "basis_residual_component_results.csv",
)

ASSEMBLY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "basis_residual_assembly_results.csv",
)

QUARTILE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "basis_residual_by_quartile.csv",
)

STAGE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "basis_residual_by_stage.csv",
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "basis_residual_summary.csv",
)


# ============================================================
# CORRECTION CAPABILITY
# ============================================================

Z_LIMIT = 2.5
THETA_LIMIT = 1.2
LOCATOR_LIMIT = 1.0

EPS = 1e-12


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    CORRECTABILITY_FILE,
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMissing required file:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

component_df = pd.read_csv(
    COMPONENT_FILE
)

correctability_df = pd.read_csv(
    CORRECTABILITY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "CORRECTION-BASIS RESIDUAL AUDIT"
)

print(
    "============================================================"
)

print(
    f"\nComponent decisions : {len(component_df)}"
)

print(
    f"Assemblies           : "
    f"{component_df['assembly_id'].nunique()}"
)


# ============================================================
# BASIC DATA CHECKS
# ============================================================

required_component_columns = [
    "assembly_id",
    "component_index",
    "offset_mm",
    "tilt_deg",
    "bend_mm",
    "waviness_mm",
    "twist_mm",
    "local_bump_mm",
    "batch_offset_bias_mm",
    "batch_angular_bias_deg",
    "fixture_drift_mm",
]

missing_columns = [
    column
    for column in required_component_columns
    if column not in component_df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )


if component_df.duplicated(
    subset=[
        "assembly_id",
        "component_index",
    ]
).any():

    raise ValueError(
        "\nDuplicate assembly/component decisions detected."
    )


# ============================================================
# RECONSTRUCT COMPONENT PROFILE
# ============================================================

def reconstruct_component_profile(row):

    profile = np.zeros_like(
        s,
        dtype=float,
    )

    offset_mm = float(
        row["offset_mm"]
    )

    tilt_deg = float(
        row["tilt_deg"]
    )

    bend_mm = float(
        row["bend_mm"]
    )

    waviness_mm = float(
        row["waviness_mm"]
    )

    twist_mm = float(
        row["twist_mm"]
    )

    local_bump_mm = float(
        row["local_bump_mm"]
    )


    if offset_mm != 0.0:

        profile += deviation_offset(
            offset_mm=offset_mm
        )


    if tilt_deg != 0.0:

        profile += deviation_tilt(
            angle_deg=tilt_deg
        )


    if bend_mm != 0.0:

        profile += deviation_bend(
            amplitude_mm=bend_mm
        )


    if waviness_mm != 0.0:

        profile += deviation_waviness(
            amplitude_mm=waviness_mm,
            waves=3,
        )


    if twist_mm != 0.0:

        profile += deviation_twist(
            amplitude_mm=twist_mm
        )


    if local_bump_mm != 0.0:

        profile += deviation_local_bump(
            amplitude_mm=local_bump_mm,
            sigma=0.12,
        )


    return profile


# ============================================================
# RECONSTRUCT BATCH / PROCESS DISTURBANCE
# ============================================================

def reconstruct_batch_disturbance(row):

    offset_profile = np.full_like(
        s,
        float(
            row["batch_offset_bias_mm"]
        ),
        dtype=float,
    )


    angular_profile = (
        np.tan(
            np.deg2rad(
                float(
                    row["batch_angular_bias_deg"]
                )
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
        float(
            row["fixture_drift_mm"]
        )
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
# CAPABILITY UTILIZATION
# ============================================================

def calculate_utilization(
    z_adj,
    theta_adj,
    locator_offset,
):

    return float(
        max(
            abs(z_adj) / Z_LIMIT,
            abs(theta_adj) / THETA_LIMIT,
            abs(locator_offset) / LOCATOR_LIMIT,
        )
    )


# ============================================================
# CORRECTION BASIS
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
        LOCATOR_SIGMA ** 2
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
            THETA_LIMIT
        )
    )
)


LOWER_BOUNDS = np.array(
    [
        -Z_LIMIT,
        -THETA_TAN_LIMIT,
        -LOCATOR_LIMIT,
    ],
    dtype=float,
)


UPPER_BOUNDS = np.array(
    [
        Z_LIMIT,
        THETA_TAN_LIMIT,
        LOCATOR_LIMIT,
    ],
    dtype=float,
)


# ============================================================
# BASIS FIT
# ============================================================

def basis_fit_diagnostics(
    uncorrected_state,
):

    state = np.asarray(
        uncorrected_state,
        dtype=float,
    )


    state_norm = float(
        np.linalg.norm(
            state
        )
    )


    # --------------------------------------------------------
    # BOUNDED FIT
    # Same optimization used by the deterministic baseline.
    # --------------------------------------------------------

    bounded_result = lsq_linear(
        CORRECTION_BASIS,
        -state,
        bounds=(
            LOWER_BOUNDS,
            UPPER_BOUNDS,
        ),
        method="trf",
        lsmr_tol="auto",
    )


    bounded_parameters = np.asarray(
        bounded_result.x,
        dtype=float,
    )


    bounded_correction = (
        CORRECTION_BASIS
        @
        bounded_parameters
    )


    bounded_residual = (
        state
        +
        bounded_correction
    )


    bounded_residual_norm = float(
        np.linalg.norm(
            bounded_residual
        )
    )


    if state_norm > EPS:

        bounded_residual_ratio = float(
            bounded_residual_norm
            /
            state_norm
        )

        bounded_explained_energy = float(
            1.0
            -
            (
                bounded_residual_norm ** 2
                /
                state_norm ** 2
            )
        )

    else:

        bounded_residual_ratio = 0.0
        bounded_explained_energy = 1.0


    bounded_explained_energy = float(
        np.clip(
            bounded_explained_energy,
            0.0,
            1.0,
        )
    )


    # --------------------------------------------------------
    # UNBOUNDED FIT
    #
    # This isolates geometric basis representability from
    # correction capability limits.
    # --------------------------------------------------------

    unbounded_parameters, _, _, _ = (
        np.linalg.lstsq(
            CORRECTION_BASIS,
            -state,
            rcond=None,
        )
    )


    unbounded_correction = (
        CORRECTION_BASIS
        @
        unbounded_parameters
    )


    unbounded_residual = (
        state
        +
        unbounded_correction
    )


    unbounded_residual_norm = float(
        np.linalg.norm(
            unbounded_residual
        )
    )


    if state_norm > EPS:

        unbounded_residual_ratio = float(
            unbounded_residual_norm
            /
            state_norm
        )

        unbounded_explained_energy = float(
            1.0
            -
            (
                unbounded_residual_norm ** 2
                /
                state_norm ** 2
            )
        )

    else:

        unbounded_residual_ratio = 0.0
        unbounded_explained_energy = 1.0


    unbounded_explained_energy = float(
        np.clip(
            unbounded_explained_energy,
            0.0,
            1.0,
        )
    )


    # --------------------------------------------------------
    # CAPABILITY LOSS
    #
    # Difference between bounded and unbounded residuals.
    # --------------------------------------------------------

    capability_residual_penalty = float(
        bounded_residual_ratio
        -
        unbounded_residual_ratio
    )


    # --------------------------------------------------------
    # CONVERT BOUNDED PARAMETERS TO ACTUAL CONTROLLER VALUES
    # --------------------------------------------------------

    z_adj = float(
        bounded_parameters[0]
    )


    tan_theta = float(
        bounded_parameters[1]
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
            -THETA_LIMIT,
            THETA_LIMIT,
        )
    )


    locator_offset = float(
        bounded_parameters[2]
    )


    return {
        "z_adj":
            z_adj,

        "theta_adj":
            theta_adj,

        "locator_offset":
            locator_offset,

        "solve_success":
            bool(
                bounded_result.success
            ),

        "state_l2_norm":
            state_norm,

        "bounded_residual_l2":
            bounded_residual_norm,

        "bounded_residual_ratio":
            bounded_residual_ratio,

        "bounded_explained_energy":
            bounded_explained_energy,

        "unbounded_residual_l2":
            unbounded_residual_norm,

        "unbounded_residual_ratio":
            unbounded_residual_ratio,

        "unbounded_explained_energy":
            unbounded_explained_energy,

        "capability_residual_penalty":
            capability_residual_penalty,
    }


# ============================================================
# STRUCTURED-20 REFERENCE
# ============================================================

structured_quality_column_candidates = [
    "structured_actual_quality",
    "structured_quality",
    "corrected_quality",
]


STRUCTURED_QUALITY_COLUMN = None


for candidate in (
    structured_quality_column_candidates
):

    if candidate in correctability_df.columns:

        STRUCTURED_QUALITY_COLUMN = candidate
        break


if STRUCTURED_QUALITY_COLUMN is None:

    raise ValueError(
        "\nCould not identify the Structured-20 "
        "actual-quality column."
    )


print(
    f"\nStructured reference column : "
    f"{STRUCTURED_QUALITY_COLUMN}"
)


structured_reference_df = (
    correctability_df[
        [
            "assembly_id",
            "component_index",
            STRUCTURED_QUALITY_COLUMN,
        ]
    ]
    .copy()
    .rename(
        columns={
            STRUCTURED_QUALITY_COLUMN:
                "structured_actual_quality"
        }
    )
)


# ============================================================
# MAIN SEQUENTIAL REPLAY
# ============================================================

component_records = []
assembly_records = []


assembly_ids = sorted(
    component_df[
        "assembly_id"
    ]
    .unique()
)


for assembly_counter, assembly_id in enumerate(
    assembly_ids,
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


    zero_state = (
        create_initial_state()
    )


    deterministic_state = (
        create_initial_state()
    )


    assembly_basis_ratios = []
    assembly_unbounded_ratios = []


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


        # ----------------------------------------------------
        # ZERO-CORRECTION TRAJECTORY
        # ----------------------------------------------------

        zero_correction = (
            correction_profile(
                z_adj_mm=0.0,
                theta_adj_deg=0.0,
                locator_offset_mm=0.0,
            )
        )


        zero_state = (
            update_assembly_state(
                previous_state=zero_state,
                component_deviation=component_profile,
                fixture_drift=batch_disturbance,
                correction=zero_correction,
            )
        )


        zero_metrics = (
            calculate_quality_metrics(
                zero_state
            )
        )


        # ----------------------------------------------------
        # DETERMINISTIC STATE BEFORE CURRENT CORRECTION
        # ----------------------------------------------------

        deterministic_uncorrected_state = (
            update_assembly_state(
                previous_state=deterministic_state,
                component_deviation=component_profile,
                fixture_drift=batch_disturbance,
                correction=zero_correction,
            )
        )


        uncorrected_metrics = (
            calculate_quality_metrics(
                deterministic_uncorrected_state
            )
        )


        # ----------------------------------------------------
        # BASIS REPRESENTABILITY DIAGNOSTICS
        # ----------------------------------------------------

        fit = basis_fit_diagnostics(
            deterministic_uncorrected_state
        )


        z_adj = fit[
            "z_adj"
        ]

        theta_adj = fit[
            "theta_adj"
        ]

        locator_offset = fit[
            "locator_offset"
        ]


        deterministic_correction_profile = (
            correction_profile(
                z_adj_mm=z_adj,
                theta_adj_deg=theta_adj,
                locator_offset_mm=locator_offset,
            )
        )


        deterministic_state = (
            update_assembly_state(
                previous_state=deterministic_state,
                component_deviation=component_profile,
                fixture_drift=batch_disturbance,
                correction=deterministic_correction_profile,
            )
        )


        deterministic_metrics = (
            calculate_quality_metrics(
                deterministic_state
            )
        )


        utilization = (
            calculate_utilization(
                z_adj,
                theta_adj,
                locator_offset,
            )
        )


        # ----------------------------------------------------
        # STRUCTURED-20 RESULT
        # ----------------------------------------------------

        structured_row = (
            structured_reference_df[
                (
                    structured_reference_df[
                        "assembly_id"
                    ]
                    ==
                    assembly_id
                )
                &
                (
                    structured_reference_df[
                        "component_index"
                    ]
                    ==
                    component_index
                )
            ]
        )


        if len(
            structured_row
        ) != 1:

            raise ValueError(
                f"\nCould not uniquely match Structured-20 "
                f"reference for assembly {assembly_id}, "
                f"component {component_index}."
            )


        structured_quality = float(
            structured_row[
                "structured_actual_quality"
            ]
            .iloc[0]
        )


        structured_advantage = float(
            deterministic_metrics[
                "quality_score"
            ]
            -
            structured_quality
        )


        structured_wins = bool(
            structured_quality
            <
            deterministic_metrics[
                "quality_score"
            ]
        )


        assembly_basis_ratios.append(
            fit[
                "bounded_residual_ratio"
            ]
        )


        assembly_unbounded_ratios.append(
            fit[
                "unbounded_residual_ratio"
            ]
        )


        component_records.append(
            {
                "assembly_id":
                    int(
                        assembly_id
                    ),

                "component_index":
                    component_index,

                "zero_quality":
                    float(
                        zero_metrics[
                            "quality_score"
                        ]
                    ),

                "deterministic_uncorrected_quality":
                    float(
                        uncorrected_metrics[
                            "quality_score"
                        ]
                    ),

                "deterministic_quality":
                    float(
                        deterministic_metrics[
                            "quality_score"
                        ]
                    ),

                "structured_quality":
                    structured_quality,

                "structured_advantage":
                    structured_advantage,

                "structured_wins":
                    structured_wins,

                "state_l2_norm":
                    fit[
                        "state_l2_norm"
                    ],

                "bounded_residual_l2":
                    fit[
                        "bounded_residual_l2"
                    ],

                "bounded_residual_ratio":
                    fit[
                        "bounded_residual_ratio"
                    ],

                "bounded_explained_energy":
                    fit[
                        "bounded_explained_energy"
                    ],

                "unbounded_residual_l2":
                    fit[
                        "unbounded_residual_l2"
                    ],

                "unbounded_residual_ratio":
                    fit[
                        "unbounded_residual_ratio"
                    ],

                "unbounded_explained_energy":
                    fit[
                        "unbounded_explained_energy"
                    ],

                "capability_residual_penalty":
                    fit[
                        "capability_residual_penalty"
                    ],

                "deterministic_z_adj":
                    z_adj,

                "deterministic_theta_adj":
                    theta_adj,

                "deterministic_locator_offset":
                    locator_offset,

                "deterministic_utilization":
                    utilization,

                "least_squares_success":
                    fit[
                        "solve_success"
                    ],
            }
        )


    # --------------------------------------------------------
    # FINAL ASSEMBLY RESULTS
    # --------------------------------------------------------

    final_zero_quality = float(
        calculate_quality_metrics(
            zero_state
        )[
            "quality_score"
        ]
    )


    final_deterministic_quality = float(
        calculate_quality_metrics(
            deterministic_state
        )[
            "quality_score"
        ]
    )


    final_structured_quality = float(
        structured_reference_df[
            structured_reference_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
        .sort_values(
            "component_index"
        )[
            "structured_actual_quality"
        ]
        .iloc[-1]
    )


    assembly_records.append(
        {
            "assembly_id":
                int(
                    assembly_id
                ),

            "zero_final_quality":
                final_zero_quality,

            "deterministic_final_quality":
                final_deterministic_quality,

            "structured_final_quality":
                final_structured_quality,

            "structured_final_advantage":
                (
                    final_deterministic_quality
                    -
                    final_structured_quality
                ),

            "mean_bounded_residual_ratio":
                float(
                    np.mean(
                        assembly_basis_ratios
                    )
                ),

            "max_bounded_residual_ratio":
                float(
                    np.max(
                        assembly_basis_ratios
                    )
                ),

            "mean_unbounded_residual_ratio":
                float(
                    np.mean(
                        assembly_unbounded_ratios
                    )
                ),
        }
    )


    if (
        assembly_counter
        %
        25
        ==
        0
    ):

        print(
            f"Completed "
            f"{assembly_counter}/"
            f"{len(assembly_ids)} assemblies"
        )


# ============================================================
# RESULT DATAFRAMES
# ============================================================

component_result_df = pd.DataFrame(
    component_records
)


assembly_result_df = pd.DataFrame(
    assembly_records
)


# ============================================================
# SANITY CHECK AGAINST FROZEN DETERMINISTIC RESULT
# ============================================================

reference_max_difference = np.nan


if os.path.exists(
    DETERMINISTIC_REFERENCE_FILE
):

    deterministic_reference_df = pd.read_csv(
        DETERMINISTIC_REFERENCE_FILE
    )


    comparison_df = (
        assembly_result_df[
            [
                "assembly_id",
                "deterministic_final_quality",
            ]
        ]
        .merge(
            deterministic_reference_df[
                [
                    "assembly_id",
                    "deterministic_final_quality",
                ]
            ],
            on="assembly_id",
            suffixes=(
                "_replayed",
                "_reference",
            ),
            how="inner",
        )
    )


    reference_max_difference = float(
        np.max(
            np.abs(
                comparison_df[
                    "deterministic_final_quality_replayed"
                ]
                -
                comparison_df[
                    "deterministic_final_quality_reference"
                ]
            )
        )
    )


# ============================================================
# QUARTILE ANALYSIS
# ============================================================

component_result_df[
    "basis_residual_quartile"
] = pd.qcut(
    component_result_df[
        "bounded_residual_ratio"
    ],
    q=4,
    labels=[
        "Q1_best_basis_fit",
        "Q2",
        "Q3",
        "Q4_worst_basis_fit",
    ],
    duplicates="drop",
)


quartile_df = (
    component_result_df
    .groupby(
        "basis_residual_quartile",
        observed=False,
    )
    .agg(
        decisions=(
            "assembly_id",
            "count",
        ),

        mean_bounded_residual_ratio=(
            "bounded_residual_ratio",
            "mean",
        ),

        mean_unbounded_residual_ratio=(
            "unbounded_residual_ratio",
            "mean",
        ),

        mean_bounded_explained_energy=(
            "bounded_explained_energy",
            "mean",
        ),

        mean_capability_penalty=(
            "capability_residual_penalty",
            "mean",
        ),

        mean_deterministic_quality=(
            "deterministic_quality",
            "mean",
        ),

        mean_structured_quality=(
            "structured_quality",
            "mean",
        ),

        structured_win_percent=(
            "structured_wins",
            lambda x:
                100.0
                *
                float(
                    np.mean(x)
                ),
        ),

        mean_structured_advantage=(
            "structured_advantage",
            "mean",
        ),

        mean_deterministic_utilization=(
            "deterministic_utilization",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# STAGE ANALYSIS
# ============================================================

stage_df = (
    component_result_df
    .groupby(
        "component_index"
    )
    .agg(
        decisions=(
            "assembly_id",
            "count",
        ),

        mean_bounded_residual_ratio=(
            "bounded_residual_ratio",
            "mean",
        ),

        mean_unbounded_residual_ratio=(
            "unbounded_residual_ratio",
            "mean",
        ),

        mean_bounded_explained_energy=(
            "bounded_explained_energy",
            "mean",
        ),

        mean_capability_penalty=(
            "capability_residual_penalty",
            "mean",
        ),

        structured_win_percent=(
            "structured_wins",
            lambda x:
                100.0
                *
                float(
                    np.mean(x)
                ),
        ),

        mean_structured_advantage=(
            "structured_advantage",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# CORRELATIONS
# ============================================================

corr_bounded_vs_structured = float(
    component_result_df[
        [
            "bounded_residual_ratio",
            "structured_advantage",
        ]
    ]
    .corr()
    .iloc[
        0,
        1,
    ]
)


corr_unbounded_vs_structured = float(
    component_result_df[
        [
            "unbounded_residual_ratio",
            "structured_advantage",
        ]
    ]
    .corr()
    .iloc[
        0,
        1,
    ]
)


corr_capability_vs_structured = float(
    component_result_df[
        [
            "capability_residual_penalty",
            "structured_advantage",
        ]
    ]
    .corr()
    .iloc[
        0,
        1,
    ]
)


corr_assembly_basis_vs_final_advantage = float(
    assembly_result_df[
        [
            "mean_bounded_residual_ratio",
            "structured_final_advantage",
        ]
    ]
    .corr()
    .iloc[
        0,
        1,
    ]
)


# ============================================================
# GLOBAL METRICS
# ============================================================

mean_bounded_residual_ratio = float(
    component_result_df[
        "bounded_residual_ratio"
    ]
    .mean()
)


median_bounded_residual_ratio = float(
    component_result_df[
        "bounded_residual_ratio"
    ]
    .median()
)


p95_bounded_residual_ratio = float(
    component_result_df[
        "bounded_residual_ratio"
    ]
    .quantile(
        0.95
    )
)


mean_unbounded_residual_ratio = float(
    component_result_df[
        "unbounded_residual_ratio"
    ]
    .mean()
)


mean_bounded_explained_energy = float(
    component_result_df[
        "bounded_explained_energy"
    ]
    .mean()
)


mean_unbounded_explained_energy = float(
    component_result_df[
        "unbounded_explained_energy"
    ]
    .mean()
)


mean_capability_penalty = float(
    component_result_df[
        "capability_residual_penalty"
    ]
    .mean()
)


structured_component_win_percent = float(
    100.0
    *
    component_result_df[
        "structured_wins"
    ]
    .mean()
)


zero_mean = float(
    assembly_result_df[
        "zero_final_quality"
    ]
    .mean()
)


deterministic_mean = float(
    assembly_result_df[
        "deterministic_final_quality"
    ]
    .mean()
)


structured_mean = float(
    assembly_result_df[
        "structured_final_quality"
    ]
    .mean()
)


solve_success_rate = float(
    100.0
    *
    component_result_df[
        "least_squares_success"
    ]
    .mean()
)


# ============================================================
# SAVE TABLES
# ============================================================

component_result_df.to_csv(
    COMPONENT_OUTPUT,
    index=False,
)


assembly_result_df.to_csv(
    ASSEMBLY_OUTPUT,
    index=False,
)


quartile_df.to_csv(
    QUARTILE_OUTPUT,
    index=False,
)


stage_df.to_csv(
    STAGE_OUTPUT,
    index=False,
)


summary_df = pd.DataFrame(
    {
        "metric": [
            "assemblies",
            "component_decisions",
            "zero_mean_final_quality",
            "deterministic_mean_final_quality",
            "structured_mean_final_quality",
            "mean_bounded_residual_ratio",
            "median_bounded_residual_ratio",
            "p95_bounded_residual_ratio",
            "mean_unbounded_residual_ratio",
            "mean_bounded_explained_energy",
            "mean_unbounded_explained_energy",
            "mean_capability_residual_penalty",
            "structured_component_win_percent",
            "correlation_bounded_residual_vs_structured_advantage",
            "correlation_unbounded_residual_vs_structured_advantage",
            "correlation_capability_penalty_vs_structured_advantage",
            "correlation_assembly_basis_vs_final_structured_advantage",
            "least_squares_success_percent",
            "max_abs_replay_difference_vs_frozen_baseline",
        ],

        "value": [
            len(
                assembly_result_df
            ),

            len(
                component_result_df
            ),

            zero_mean,

            deterministic_mean,

            structured_mean,

            mean_bounded_residual_ratio,

            median_bounded_residual_ratio,

            p95_bounded_residual_ratio,

            mean_unbounded_residual_ratio,

            mean_bounded_explained_energy,

            mean_unbounded_explained_energy,

            mean_capability_penalty,

            structured_component_win_percent,

            corr_bounded_vs_structured,

            corr_unbounded_vs_structured,

            corr_capability_vs_structured,

            corr_assembly_basis_vs_final_advantage,

            solve_success_rate,

            reference_max_difference,
        ],
    }
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# FIGURES
# ============================================================

plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.hist(
    component_result_df[
        "bounded_residual_ratio"
    ],
    bins=30,
)


plt.xlabel(
    "Normalized bounded basis residual"
)

plt.ylabel(
    "Decision count"
)

plt.title(
    "Correction-Basis Residual Across Sequential Decisions"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "basis_residual_distribution.png",
    ),
    dpi=250,
)

plt.close()


plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.scatter(
    component_result_df[
        "bounded_residual_ratio"
    ],
    component_result_df[
        "structured_advantage"
    ],
    alpha=0.45,
)


plt.axhline(
    0.0,
    linewidth=1.0,
)


plt.xlabel(
    "Normalized bounded basis residual"
)

plt.ylabel(
    "Structured-20 advantage "
    "(Q_LSQ - Q_Structured)"
)

plt.title(
    "ML Relative Value vs Correction-Basis Residual"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "basis_residual_vs_structured_advantage.png",
    ),
    dpi=250,
)

plt.close()


plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.bar(
    quartile_df[
        "basis_residual_quartile"
    ]
    .astype(str),
    quartile_df[
        "structured_win_percent"
    ],
)


plt.xlabel(
    "Correction-basis residual region"
)

plt.ylabel(
    "Structured-20 win rate (%)"
)

plt.title(
    "Structured-20 Win Rate by Basis-Residual Region"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "structured_win_by_basis_residual.png",
    ),
    dpi=250,
)

plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "BASIS REPRESENTABILITY RESULTS"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies                    : "
    f"{len(assembly_result_df)}"
)

print(
    f"Component decisions           : "
    f"{len(component_result_df)}"
)


print(
    "\nFinal mean quality:"
)

print(
    f"Zero                          : "
    f"{zero_mean:.4f}"
)

print(
    f"Deterministic LSQ             : "
    f"{deterministic_mean:.4f}"
)

print(
    f"Structured-20                 : "
    f"{structured_mean:.4f}"
)


print(
    "\nCorrection-basis fit:"
)

print(
    f"Mean bounded residual ratio   : "
    f"{mean_bounded_residual_ratio:.4f}"
)

print(
    f"Median bounded residual ratio : "
    f"{median_bounded_residual_ratio:.4f}"
)

print(
    f"P95 bounded residual ratio    : "
    f"{p95_bounded_residual_ratio:.4f}"
)

print(
    f"Mean unbounded residual ratio : "
    f"{mean_unbounded_residual_ratio:.4f}"
)

print(
    f"Mean bounded explained energy : "
    f"{100.0 * mean_bounded_explained_energy:.2f}%"
)

print(
    f"Mean unbounded explained energy: "
    f"{100.0 * mean_unbounded_explained_energy:.2f}%"
)

print(
    f"Mean capability penalty       : "
    f"{mean_capability_penalty:.4f}"
)


print(
    "\nML relative-value relationships:"
)

print(
    f"Residual vs Structured advantage      : "
    f"{corr_bounded_vs_structured:.4f}"
)

print(
    f"Unbounded residual vs Structured adv. : "
    f"{corr_unbounded_vs_structured:.4f}"
)

print(
    f"Capability penalty vs Structured adv. : "
    f"{corr_capability_vs_structured:.4f}"
)

print(
    f"Assembly residual vs final Structured adv.: "
    f"{corr_assembly_basis_vs_final_advantage:.4f}"
)


print(
    f"\nStructured component win rate : "
    f"{structured_component_win_percent:.2f}%"
)

print(
    f"LSQ solve success             : "
    f"{solve_success_rate:.2f}%"
)


if not np.isnan(
    reference_max_difference
):

    print(
        f"Max replay difference vs frozen LSQ : "
        f"{reference_max_difference:.12f}"
    )


print(
    "\nBasis-residual quartiles:"
)

print(
    quartile_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nStage-wise basis residual:"
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
# CONSERVATIVE SCIENTIFIC VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "SCIENTIFIC VERDICT"
)

print(
    "============================================================"
)


if (
    mean_unbounded_explained_energy
    >=
    0.80
):

    print(
        "\nRESULT A:"
    )

    print(
        "The available correction basis explains a large "
        "fraction of the current geometric state."
    )

    print(
        "This provides a mechanistic explanation for the "
        "strong performance of deterministic LSQ under the "
        "present additive assembly model."
    )


elif (
    mean_unbounded_explained_energy
    >=
    0.50
):

    print(
        "\nRESULT B:"
    )

    print(
        "The analytical correction basis explains a "
        "substantial but incomplete fraction of the current "
        "geometric state."
    )

    print(
        "Deterministic LSQ benefits from significant basis "
        "alignment, although non-basis geometric structure "
        "remains."
    )


else:

    print(
        "\nRESULT C:"
    )

    print(
        "The available analytical correction basis explains "
        "less than half of the geometric-state energy."
    )

    print(
        "The strong deterministic performance therefore "
        "cannot be explained primarily by basis alignment "
        "alone."
    )


print(
    "\nML VALUE CHECK:"
)


if (
    corr_bounded_vs_structured
    >
    0.20
):

    print(
        "Structured-20 becomes relatively more competitive "
        "as the bounded correction-basis residual increases."
    )

    print(
        "This provides evidence for an ML-value region in "
        "geometrically less basis-aligned states."
    )


elif (
    corr_bounded_vs_structured
    <
    -0.20
):

    print(
        "Structured-20 does not become more competitive as "
        "basis mismatch increases; the observed relationship "
        "instead favors deterministic LSQ."
    )

    print(
        "No claim should therefore be made that the current "
        "ML controller specifically solves non-basis geometry "
        "better than LSQ."
    )


else:

    print(
        "No strong relationship is observed between basis "
        "residual and Structured-20 relative advantage."
    )

    print(
        "The data do not support a strong claim that the "
        "current ML controller is specifically superior in "
        "poorly basis-aligned geometric states."
    )


print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "This audit separates geometric basis representability "
    "from correction-capability limitations."
)

print(
    "It does not alter the validated controllers and does "
    "not claim that least-squares profile fitting is a global "
    "optimum of the weighted thesis quality objective."
)


print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)


print(
    "\n"
    "============================================================"
)

print(
    "CORRECTION-BASIS RESIDUAL AUDIT COMPLETED"
)

print(
    "============================================================"
)