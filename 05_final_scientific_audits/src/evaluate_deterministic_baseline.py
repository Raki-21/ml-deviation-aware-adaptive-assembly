import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import lsq_linear


# ============================================================
# DETERMINISTIC ENGINEERING BASELINE AUDIT
# ============================================================
#
# PURPOSE
#
# Evaluate whether a transparent non-ML geometric correction
# can explain most of the performance obtained by the final
# Structured-20 recommendation controller.
#
# At each sequential assembly decision:
#
#   1. Reconstruct the incoming component profile.
#   2. Reconstruct the repeated batch/process disturbance.
#   3. Compute the current uncorrected next-state profile.
#   4. Fit the available correction basis directly to the
#      negative geometric error using bounded least squares.
#   5. Apply that correction to the actual sequential simulator.
#
# No ML surrogate is used.
# No Bayesian Optimization is used.
# No random search is used.
#
# This is intentionally a strong engineering baseline.
#
# The frozen V3.2 implementation is NOT modified.
# ============================================================


# ============================================================
# PROJECT PATHS
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


# Allow import from frozen working V3.2 source directory.
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
#
# The correctability prediction dataset was reconstructed from
# the independent 300-assembly validation population.
#
# It provides the incoming component and process information
# necessary to replay those assemblies.
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


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    AUDIT_DIR,
    "results",
    "deterministic_baseline",
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
    "deterministic_component_results.csv",
)


ASSEMBLY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "deterministic_assembly_results.csv",
)


SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "deterministic_baseline_summary.csv",
)


STAGE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "deterministic_by_stage.csv",
)


# ============================================================
# CORRECTION CAPABILITY
# ============================================================

Z_LIMIT = 2.5

THETA_LIMIT = 1.2

LOCATOR_LIMIT = 1.0


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    COMPONENT_FILE,
    CORRECTABILITY_FILE,
]


for path in required_files:

    if not os.path.exists(
        path
    ):

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
    "DETERMINISTIC ENGINEERING BASELINE AUDIT"
)

print(
    "============================================================"
)


print(
    f"\nComponent decisions : "
    f"{len(component_df)}"
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
        "\nMissing required component columns:\n"
        +
        "\n".join(
            missing_columns
        )
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

def reconstruct_component_profile(
    row
):

    profile = np.zeros_like(
        s,
        dtype=float,
    )


    if float(
        row[
            "offset_mm"
        ]
    ) != 0.0:

        profile += deviation_offset(
            offset_mm=float(
                row[
                    "offset_mm"
                ]
            )
        )


    if float(
        row[
            "tilt_deg"
        ]
    ) != 0.0:

        profile += deviation_tilt(
            angle_deg=float(
                row[
                    "tilt_deg"
                ]
            )
        )


    if float(
        row[
            "bend_mm"
        ]
    ) != 0.0:

        profile += deviation_bend(
            amplitude_mm=float(
                row[
                    "bend_mm"
                ]
            )
        )


    if float(
        row[
            "waviness_mm"
        ]
    ) != 0.0:

        profile += deviation_waviness(

            amplitude_mm=float(
                row[
                    "waviness_mm"
                ]
            ),

            waves=3,
        )


    if float(
        row[
            "twist_mm"
        ]
    ) != 0.0:

        profile += deviation_twist(
            amplitude_mm=float(
                row[
                    "twist_mm"
                ]
            )
        )


    if float(
        row[
            "local_bump_mm"
        ]
    ) != 0.0:

        profile += deviation_local_bump(

            amplitude_mm=float(
                row[
                    "local_bump_mm"
                ]
            ),

            sigma=0.12,
        )


    return profile


# ============================================================
# RECONSTRUCT BATCH / PROCESS DISTURBANCE
# ============================================================

def reconstruct_batch_disturbance(
    row
):

    offset_profile = np.full_like(

        s,

        float(
            row[
                "batch_offset_bias_mm"
            ]
        ),

        dtype=float,
    )


    angular_profile = (

        np.tan(
            np.deg2rad(
                float(
                    row[
                        "batch_angular_bias_deg"
                    ]
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
            row[
                "fixture_drift_mm"
            ]
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
# DETERMINISTIC CORRECTION BASIS
# ============================================================
#
# Correction profile:
#
# C(s) =
# z_adj
# +
# tan(theta_adj) * L * (s - 0.5)
# +
# locator_offset * Gaussian(s)
#
#
# We solve initially in the transformed parameter:
#
# t = tan(theta_adj)
#
# because the profile is linear in t.
#
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
# BOUNDED LEAST-SQUARES CORRECTION
# ============================================================

def deterministic_correction(
    uncorrected_state
):

    result = lsq_linear(

        CORRECTION_BASIS,

        -uncorrected_state,

        bounds=(
            LOWER_BOUNDS,
            UPPER_BOUNDS,
        ),

        method="trf",

        lsmr_tol="auto",
    )


    z_adj = float(
        result.x[
            0
        ]
    )


    tan_theta = float(
        result.x[
            1
        ]
    )


    theta_adj = float(
        np.rad2deg(
            np.arctan(
                tan_theta
            )
        )
    )


    locator_offset = float(
        result.x[
            2
        ]
    )


    theta_adj = float(
        np.clip(
            theta_adj,
            -THETA_LIMIT,
            THETA_LIMIT,
        )
    )


    return (

        z_adj,

        theta_adj,

        locator_offset,

        bool(
            result.success
        ),
    )


# ============================================================
# STRUCTURED-20 REFERENCE EXTRACTION
# ============================================================
#
# We use the already-existing independent-validation outcome.
# We do not rerun Structured-20.
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

    print(
        "\nAvailable correctability columns:"
    )

    for column in correctability_df.columns:

        print(
            column
        )

    raise ValueError(
        "\nCould not identify the existing Structured-20 "
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
)


structured_reference_df = (
    structured_reference_df.rename(

        columns={

            STRUCTURED_QUALITY_COLUMN:
                "structured_actual_quality"
        }
    )
)


# ============================================================
# MAIN SEQUENTIAL AUDIT
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


    zero_state = create_initial_state()


    deterministic_state = (
        create_initial_state()
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


        # ----------------------------------------------------
        # ZERO-CORRECTION TRAJECTORY
        # ----------------------------------------------------

        zero_correction = correction_profile(

            z_adj_mm=0.0,

            theta_adj_deg=0.0,

            locator_offset_mm=0.0,
        )


        zero_state = update_assembly_state(

            previous_state=zero_state,

            component_deviation=(
                component_profile
            ),

            fixture_drift=(
                batch_disturbance
            ),

            correction=(
                zero_correction
            ),
        )


        zero_metrics = (
            calculate_quality_metrics(
                zero_state
            )
        )


        # ----------------------------------------------------
        # DETERM. STATE BEFORE CURRENT CORRECTION
        # ----------------------------------------------------

        deterministic_uncorrected_state = (
            update_assembly_state(

                previous_state=(
                    deterministic_state
                ),

                component_deviation=(
                    component_profile
                ),

                fixture_drift=(
                    batch_disturbance
                ),

                correction=(
                    zero_correction
                ),
            )
        )


        # ----------------------------------------------------
        # GEOMETRIC LEAST-SQUARES RECOMMENDATION
        # ----------------------------------------------------

        (
            z_adj,
            theta_adj,
            locator_offset,
            solve_success,
        ) = deterministic_correction(
            deterministic_uncorrected_state
        )


        deterministic_correction_profile = (
            correction_profile(

                z_adj_mm=z_adj,

                theta_adj_deg=theta_adj,

                locator_offset_mm=(
                    locator_offset
                ),
            )
        )


        deterministic_state = (
            update_assembly_state(

                previous_state=(
                    deterministic_state
                ),

                component_deviation=(
                    component_profile
                ),

                fixture_drift=(
                    batch_disturbance
                ),

                correction=(
                    deterministic_correction_profile
                ),
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
            .iloc[
                0
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

                "deterministic_quality":
                    float(
                        deterministic_metrics[
                            "quality_score"
                        ]
                    ),

                "structured_quality":
                    structured_quality,

                "deterministic_z_adj":
                    z_adj,

                "deterministic_theta_adj":
                    theta_adj,

                "deterministic_locator_offset":
                    locator_offset,

                "deterministic_utilization":
                    utilization,

                "least_squares_success":
                    solve_success,
            }
        )


    # --------------------------------------------------------
    # FINAL ASSEMBLY VALUES
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

        .iloc[
            -1
        ]
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
            f"\nCompleted "
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
# ASSEMBLY-LEVEL COMPARISON
# ============================================================

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


deterministic_improvement = float(

    (
        zero_mean
        -
        deterministic_mean
    )

    /

    zero_mean

    *

    100.0
)


structured_improvement = float(

    (
        zero_mean
        -
        structured_mean
    )

    /

    zero_mean

    *

    100.0
)


deterministic_win_zero = float(

    (
        assembly_result_df[
            "deterministic_final_quality"
        ]
        <
        assembly_result_df[
            "zero_final_quality"
        ]
    )

    .mean()

    *

    100.0
)


structured_win_deterministic = float(

    (
        assembly_result_df[
            "structured_final_quality"
        ]
        <
        assembly_result_df[
            "deterministic_final_quality"
        ]
    )

    .mean()

    *

    100.0
)


deterministic_win_structured = float(

    (
        assembly_result_df[
            "deterministic_final_quality"
        ]
        <
        assembly_result_df[
            "structured_final_quality"
        ]
    )

    .mean()

    *

    100.0
)


quality_gap = float(
    deterministic_mean
    -
    structured_mean
)


relative_structured_advantage = float(

    (
        deterministic_mean
        -
        structured_mean
    )

    /

    deterministic_mean

    *

    100.0
)


# ============================================================
# COMPONENT / CAPABILITY RESULTS
# ============================================================

mean_deterministic_utilization = float(

    component_result_df[
        "deterministic_utilization"
    ]
    .mean()
)


near_limit_rate = float(

    (
        component_result_df[
            "deterministic_utilization"
        ]
        >=
        0.90
    )

    .mean()

    *

    100.0
)


solve_success_rate = float(

    component_result_df[
        "least_squares_success"
    ]
    .mean()

    *

    100.0
)


# ============================================================
# STAGE SUMMARY
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

        zero_quality=(
            "zero_quality",
            "mean",
        ),

        deterministic_quality=(
            "deterministic_quality",
            "mean",
        ),

        structured_quality=(
            "structured_quality",
            "mean",
        ),

        deterministic_utilization=(
            "deterministic_utilization",
            "mean",
        ),
    )

    .reset_index()
)


# ============================================================
# SAVE RESULTS
# ============================================================

component_result_df.to_csv(
    COMPONENT_OUTPUT,
    index=False,
)


assembly_result_df.to_csv(
    ASSEMBLY_OUTPUT,
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

            "deterministic_improvement_percent",

            "structured_improvement_percent",

            "deterministic_win_vs_zero_percent",

            "structured_win_vs_deterministic_percent",

            "deterministic_win_vs_structured_percent",

            "absolute_quality_gap_det_minus_structured",

            "relative_structured_advantage_percent",

            "mean_deterministic_utilization",

            "deterministic_near_90pct_capability_percent",

            "least_squares_success_percent",
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

            deterministic_improvement,

            structured_improvement,

            deterministic_win_zero,

            structured_win_deterministic,

            deterministic_win_structured,

            quality_gap,

            relative_structured_advantage,

            mean_deterministic_utilization,

            near_limit_rate,

            solve_success_rate,
        ],
    }
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# FIGURE
# ============================================================

plot_df = pd.DataFrame(
    {

        "Strategy": [

            "Zero",

            "Deterministic LSQ",

            "Structured-20",
        ],

        "Mean final quality": [

            zero_mean,

            deterministic_mean,

            structured_mean,
        ],
    }
)


plt.figure(
    figsize=(
        8.0,
        5.0,
    )
)


plt.bar(

    plot_df[
        "Strategy"
    ],

    plot_df[
        "Mean final quality"
    ],
)


plt.ylabel(
    "Mean final quality score"
)


plt.title(
    "Deterministic Engineering Baseline Audit"
)


plt.grid(
    axis="y",
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        FIGURE_DIR,
        "deterministic_vs_structured.png",
    ),

    dpi=300,

    bbox_inches="tight",
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
    "DETERMINISTIC BASELINE SUMMARY"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies                 : "
    f"{len(assembly_result_df)}"
)


print(
    f"Component decisions        : "
    f"{len(component_result_df)}"
)


print(
    "\nMean final quality:"
)


print(
    f"\nZero                       : "
    f"{zero_mean:.4f}"
)


print(
    f"Deterministic LSQ          : "
    f"{deterministic_mean:.4f}"
)


print(
    f"Structured-20              : "
    f"{structured_mean:.4f}"
)


print(
    "\nImprovement relative to zero:"
)


print(
    f"\nDeterministic LSQ          : "
    f"{deterministic_improvement:.2f}%"
)


print(
    f"Structured-20              : "
    f"{structured_improvement:.2f}%"
)


print(
    "\nStrategy comparison:"
)


print(
    f"\nDeterministic beats zero   : "
    f"{deterministic_win_zero:.2f}%"
)


print(
    f"Structured beats deterministic: "
    f"{structured_win_deterministic:.2f}%"
)


print(
    f"Deterministic beats Structured: "
    f"{deterministic_win_structured:.2f}%"
)


print(
    f"\nMean quality gap "
    f"(det - structured)        : "
    f"{quality_gap:.4f}"
)


print(
    f"Relative Structured advantage: "
    f"{relative_structured_advantage:.2f}%"
)


print(
    "\nCapability:"
)


print(
    f"\nMean deterministic U       : "
    f"{mean_deterministic_utilization:.3f}"
)


print(
    f">=90% deterministic U      : "
    f"{near_limit_rate:.2f}%"
)


print(
    f"LSQ solve success          : "
    f"{solve_success_rate:.2f}%"
)


print(
    "\nStage-wise:"
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
# SCIENTIFIC VERDICT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "AUDIT 2 VERDICT"
)

print(
    "============================================================"
)


if (
    relative_structured_advantage
    >=
    20.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "Structured-20 substantially outperforms the "
        "transparent deterministic geometric baseline."
    )


    print(
        "\nThe ML-supported candidate ranking therefore "
        "provides clear additional value beyond direct "
        "least-squares compensation of the available "
        "correction modes."
    )


elif (
    relative_structured_advantage
    >=
    5.0
):


    print(
        "\nRESULT B:"
    )


    print(
        "Structured-20 provides a measurable but moderate "
        "advantage over deterministic geometric correction."
    )


    print(
        "\nThe ML contribution is useful, but a substantial "
        "fraction of the achievable correction can already "
        "be explained by the physical correction basis."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "The deterministic geometric baseline performs close "
        "to Structured-20."
    )


    print(
        "\nThis indicates that the simplified current assembly "
        "model is highly correctable using direct geometric "
        "projection alone."
    )


    print(
        "\nThe role of ML should therefore be framed primarily "
        "as an extensible surrogate/recommendation mechanism "
        "for more complex or expensive response models rather "
        "than as necessary for solving the present simplified "
        "geometry."
    )


print(
    "\nIMPORTANT:"
)


print(
    "The deterministic baseline minimizes profile L2 error "
    "rather than directly optimizing the weighted thesis "
    "quality score."
)


print(
    "Therefore it is a transparent engineering baseline, "
    "not a global optimum for the final quality objective."
)


print(
    "\nSaved results:"
)


print(
    OUTPUT_DIR
)


print(
    "\n"
    "============================================================"
)

print(
    "DETERMINISTIC BASELINE AUDIT COMPLETED"
)

print(
    "============================================================"
)