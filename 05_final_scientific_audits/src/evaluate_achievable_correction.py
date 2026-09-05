import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution


# ============================================================
# ACHIEVABLE-CORRECTION AUDIT
# ============================================================
#
# Purpose
# -------
#
# Estimate a strong simulator-direct reference for the
# sequential correction problem.
#
# Unlike:
#
#   Structured-20
#       surrogate-supported candidate recommendation
#
#   Deterministic LSQ
#       direct L2 geometric projection
#
# this audit directly minimizes the actual weighted thesis
# quality score using the assembly simulator itself.
#
# The result is used as an APPROXIMATE achievable-performance
# reference under the present simulator and correction bounds.
#
# It must NOT be called a mathematical proof of the global
# optimum.
#
# The frozen V3.2 controller is not modified.
# ============================================================


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


DETERMINISTIC_ASSEMBLY_FILE = os.path.join(
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
    "achievable_correction",
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
    "direct_simulator_component_results.csv",
)


ASSEMBLY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "direct_simulator_assembly_results.csv",
)


SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "achievable_correction_summary.csv",
)


STAGE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "achievable_correction_by_stage.csv",
)


# ============================================================
# SETTINGS
# ============================================================

Z_LIMIT = 2.5

THETA_LIMIT = 1.2

LOCATOR_LIMIT = 1.0


BOUNDS = [

    (
        -Z_LIMIT,
        Z_LIMIT,
    ),

    (
        -THETA_LIMIT,
        THETA_LIMIT,
    ),

    (
        -LOCATOR_LIMIT,
        LOCATOR_LIMIT,
    ),
]


# ------------------------------------------------------------
# Direct optimizer settings
# ------------------------------------------------------------
#
# The optimization problem contains only three variables.
#
# These settings intentionally provide a fairly strong
# reference while keeping the 1500-decision experiment
# computationally manageable.
# ------------------------------------------------------------

DE_MAXITER = 18

DE_POPSIZE = 7

DE_TOL = 1e-6

DE_POLISH = True

RANDOM_SEED = 20260827


# ============================================================
# INPUT CHECK
# ============================================================

required_files = [

    COMPONENT_FILE,

    CORRECTABILITY_FILE,

    DETERMINISTIC_ASSEMBLY_FILE,
]


for path in required_files:

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"\nMissing required file:\n{path}"
        )


# ============================================================
# LOAD
# ============================================================

component_df = pd.read_csv(
    COMPONENT_FILE
)


correctability_df = pd.read_csv(
    CORRECTABILITY_FILE
)


deterministic_assembly_df = pd.read_csv(
    DETERMINISTIC_ASSEMBLY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "ACHIEVABLE-CORRECTION AUDIT"
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
# CHECK DECISION KEYS
# ============================================================

KEY_COLUMNS = [

    "assembly_id",

    "component_index",
]


if component_df.duplicated(
    subset=KEY_COLUMNS
).any():

    raise ValueError(
        "\nDuplicate component decision keys detected."
    )


# ============================================================
# COMPONENT PROFILE RECONSTRUCTION
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
# BATCH DISTURBANCE RECONSTRUCTION
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
# SIMULATE ONE CORRECTION
# ============================================================

def simulate_candidate(
    state,
    component_profile,
    batch_disturbance,
    point,
):

    (
        z_adj,

        theta_adj,

        locator_offset,
    ) = point


    correction = correction_profile(

        z_adj_mm=float(
            z_adj
        ),

        theta_adj_deg=float(
            theta_adj
        ),

        locator_offset_mm=float(
            locator_offset
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


    metrics = calculate_quality_metrics(
        new_state
    )


    return (

        new_state,

        metrics,
    )


# ============================================================
# DIRECT SIMULATOR OPTIMIZATION
# ============================================================

def optimize_current_decision(
    state,
    component_profile,
    batch_disturbance,
    seed,
):


    evaluation_counter = {
        "count": 0
    }


    def objective(
        point
    ):

        evaluation_counter[
            "count"
        ] += 1


        (
            _,
            metrics,
        ) = simulate_candidate(

            state=state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            point=point,
        )


        return float(
            metrics[
                "quality_score"
            ]
        )


    result = differential_evolution(

        func=objective,

        bounds=BOUNDS,

        maxiter=DE_MAXITER,

        popsize=DE_POPSIZE,

        tol=DE_TOL,

        polish=DE_POLISH,

        seed=seed,

        workers=1,

        updating="immediate",
    )


    best_point = (

        float(
            result.x[
                0
            ]
        ),

        float(
            result.x[
                1
            ]
        ),

        float(
            result.x[
                2
            ]
        ),
    )


    (
        best_state,

        best_metrics,
    ) = simulate_candidate(

        state=state,

        component_profile=(
            component_profile
        ),

        batch_disturbance=(
            batch_disturbance
        ),

        point=best_point,
    )


    return {

        "z_adj":
            best_point[
                0
            ],

        "theta_adj":
            best_point[
                1
            ],

        "locator_offset":
            best_point[
                2
            ],

        "state":
            best_state,

        "metrics":
            best_metrics,

        "optimizer_success":
            bool(
                result.success
            ),

        "evaluations":
            int(
                evaluation_counter[
                    "count"
                ]
            ),

        "objective":
            float(
                result.fun
            ),
    }


# ============================================================
# STRUCTURED-20 QUALITY REFERENCE
# ============================================================

structured_quality_candidates = [

    "structured_actual_quality",

    "structured_quality",

    "corrected_quality",
]


STRUCTURED_QUALITY_COLUMN = None


for candidate in (
    structured_quality_candidates
):

    if candidate in (
        correctability_df.columns
    ):

        STRUCTURED_QUALITY_COLUMN = candidate

        break


if STRUCTURED_QUALITY_COLUMN is None:

    raise ValueError(
        "\nCould not find the Structured-20 actual-quality "
        "column in the correctability table."
    )


structured_df = (

    correctability_df[
        [
            "assembly_id",
            "component_index",
            STRUCTURED_QUALITY_COLUMN,
        ]
    ]

    .rename(
        columns={
            STRUCTURED_QUALITY_COLUMN:
                "structured_quality"
        }
    )

    .copy()
)


# ============================================================
# MAIN SEQUENTIAL EXPERIMENT
# ============================================================

component_records = []

assembly_records = []


assembly_ids = sorted(
    component_df[
        "assembly_id"
    ]
    .unique()
)


experiment_start = time.time()


for (
    assembly_counter,
    assembly_id,
) in enumerate(

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


    direct_state = create_initial_state()


    for _, row in (
        assembly_rows.iterrows()
    ):


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


        # ====================================================
        # ZERO TRAJECTORY
        # ====================================================

        (
            zero_state,

            zero_metrics,
        ) = simulate_candidate(

            state=zero_state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            point=(
                0.0,
                0.0,
                0.0,
            ),
        )


        # ====================================================
        # QUALITY BEFORE DIRECT CORRECTION
        # ====================================================

        (
            _,

            direct_uncorrected_metrics,
        ) = simulate_candidate(

            state=direct_state,

            component_profile=(
                component_profile
            ),

            batch_disturbance=(
                batch_disturbance
            ),

            point=(
                0.0,
                0.0,
                0.0,
            ),
        )


        uncorrected_direct_quality = float(

            direct_uncorrected_metrics[
                "quality_score"
            ]
        )


        # ====================================================
        # DIRECT SIMULATOR OPTIMUM
        # ====================================================

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


        direct_result = (
            optimize_current_decision(

                state=direct_state,

                component_profile=(
                    component_profile
                ),

                batch_disturbance=(
                    batch_disturbance
                ),

                seed=decision_seed,
            )
        )


        direct_state = (
            direct_result[
                "state"
            ]
        )


        direct_metrics = (
            direct_result[
                "metrics"
            ]
        )


        direct_quality = float(

            direct_metrics[
                "quality_score"
            ]
        )


        improvement_potential = (

            uncorrected_direct_quality

            -

            direct_quality
        )


        improvement_percent = (

            improvement_potential

            /

            max(
                uncorrected_direct_quality,
                1e-12,
            )

            *

            100.0
        )


        utilization = (
            calculate_utilization(

                direct_result[
                    "z_adj"
                ],

                direct_result[
                    "theta_adj"
                ],

                direct_result[
                    "locator_offset"
                ],
            )
        )


        structured_match = (

            structured_df[

                (
                    structured_df[
                        "assembly_id"
                    ]
                    ==
                    assembly_id
                )

                &

                (
                    structured_df[
                        "component_index"
                    ]
                    ==
                    component_index
                )
            ]
        )


        if len(
            structured_match
        ) != 1:

            raise ValueError(
                f"\nStructured reference match failed for "
                f"assembly {assembly_id}, component "
                f"{component_index}."
            )


        structured_quality = float(

            structured_match[
                "structured_quality"
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

                "direct_uncorrected_quality":
                    uncorrected_direct_quality,

                "direct_optimized_quality":
                    direct_quality,

                "structured_quality":
                    structured_quality,

                "direct_z_adj":
                    direct_result[
                        "z_adj"
                    ],

                "direct_theta_adj":
                    direct_result[
                        "theta_adj"
                    ],

                "direct_locator_offset":
                    direct_result[
                        "locator_offset"
                    ],

                "direct_utilization":
                    utilization,

                "direct_improvement_potential":
                    improvement_potential,

                "direct_improvement_percent":
                    improvement_percent,

                "optimizer_success":
                    direct_result[
                        "optimizer_success"
                    ],

                "optimizer_evaluations":
                    direct_result[
                        "evaluations"
                    ],
            }
        )


    # ========================================================
    # FINAL ASSEMBLY QUALITY
    # ========================================================

    zero_final_quality = float(

        calculate_quality_metrics(
            zero_state
        )[
            "quality_score"
        ]
    )


    direct_final_quality = float(

        calculate_quality_metrics(
            direct_state
        )[
            "quality_score"
        ]
    )


    final_structured_quality = float(

        structured_df[

            structured_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]

        .sort_values(
            "component_index"
        )[
            "structured_quality"
        ]

        .iloc[
            -1
        ]
    )


    deterministic_match = (

        deterministic_assembly_df[

            deterministic_assembly_df[
                "assembly_id"
            ]
            ==
            assembly_id
        ]
    )


    if len(
        deterministic_match
    ) != 1:

        raise ValueError(
            f"\nDeterministic assembly reference failed "
            f"for assembly {assembly_id}."
        )


    deterministic_final_quality = float(

        deterministic_match[
            "deterministic_final_quality"
        ]

        .iloc[
            0
        ]
    )


    assembly_records.append(
        {

            "assembly_id":
                int(
                    assembly_id
                ),

            "zero_final_quality":
                zero_final_quality,

            "direct_final_quality":
                direct_final_quality,

            "deterministic_final_quality":
                deterministic_final_quality,

            "structured_final_quality":
                final_structured_quality,
        }
    )


    if (
        assembly_counter
        %
        10
        ==
        0
    ):

        elapsed_minutes = (

            time.time()
            -
            experiment_start

        ) / 60.0


        print(
            f"\nCompleted "
            f"{assembly_counter}/"
            f"{len(assembly_ids)} assemblies"
            f" | elapsed "
            f"{elapsed_minutes:.2f} min"
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
# MEAN QUALITY
# ============================================================

zero_mean = float(

    assembly_result_df[
        "zero_final_quality"
    ]
    .mean()
)


direct_mean = float(

    assembly_result_df[
        "direct_final_quality"
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


# ============================================================
# TOTAL ACHIEVABLE IMPROVEMENT
# ============================================================

achievable_improvement = (

    zero_mean

    -

    direct_mean
)


structured_improvement = (

    zero_mean

    -

    structured_mean
)


deterministic_improvement = (

    zero_mean

    -

    deterministic_mean
)


structured_capture = (

    structured_improvement

    /

    max(
        achievable_improvement,
        1e-12,
    )

    *

    100.0
)


deterministic_capture = (

    deterministic_improvement

    /

    max(
        achievable_improvement,
        1e-12,
    )

    *

    100.0
)


# ============================================================
# REGRET RELATIVE TO DIRECT REFERENCE
# ============================================================

structured_regret = (

    structured_mean

    -

    direct_mean
)


deterministic_regret = (

    deterministic_mean

    -

    direct_mean
)


structured_relative_regret = (

    structured_regret

    /

    max(
        direct_mean,
        1e-12,
    )

    *

    100.0
)


deterministic_relative_regret = (

    deterministic_regret

    /

    max(
        direct_mean,
        1e-12,
    )

    *

    100.0
)


# ============================================================
# ASSEMBLY WIN RATES
# ============================================================

direct_beats_structured = float(

    (
        assembly_result_df[
            "direct_final_quality"
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


direct_beats_deterministic = float(

    (
        assembly_result_df[
            "direct_final_quality"
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


structured_near_direct_5pct = float(

    (
        assembly_result_df[
            "structured_final_quality"
        ]

        <=

        assembly_result_df[
            "direct_final_quality"
        ]

        *
        1.05
    )

    .mean()

    *

    100.0
)


deterministic_near_direct_5pct = float(

    (
        assembly_result_df[
            "deterministic_final_quality"
        ]

        <=

        assembly_result_df[
            "direct_final_quality"
        ]

        *
        1.05
    )

    .mean()

    *

    100.0
)


# ============================================================
# DIRECT CAPABILITY
# ============================================================

mean_direct_utilization = float(

    component_result_df[
        "direct_utilization"
    ]
    .mean()
)


direct_near_limit = float(

    (
        component_result_df[
            "direct_utilization"
        ]
        >=
        0.90
    )

    .mean()

    *

    100.0
)


optimizer_success_rate = float(

    component_result_df[
        "optimizer_success"
    ]
    .mean()

    *

    100.0
)


mean_optimizer_evaluations = float(

    component_result_df[
        "optimizer_evaluations"
    ]
    .mean()
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

        direct_quality=(
            "direct_optimized_quality",
            "mean",
        ),

        structured_quality=(
            "structured_quality",
            "mean",
        ),

        direct_utilization=(
            "direct_utilization",
            "mean",
        ),

        direct_improvement_percent=(
            "direct_improvement_percent",
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

            "direct_mean_final_quality",

            "deterministic_mean_final_quality",

            "structured_mean_final_quality",

            "achievable_improvement_absolute",

            "structured_improvement_capture_percent",

            "deterministic_improvement_capture_percent",

            "structured_absolute_regret_vs_direct",

            "deterministic_absolute_regret_vs_direct",

            "structured_relative_regret_percent",

            "deterministic_relative_regret_percent",

            "direct_beats_structured_percent",

            "direct_beats_deterministic_percent",

            "structured_within_5pct_direct_percent",

            "deterministic_within_5pct_direct_percent",

            "mean_direct_utilization",

            "direct_at_or_above_90pct_capability_percent",

            "optimizer_success_percent",

            "mean_optimizer_evaluations",
        ],

        "value": [

            len(
                assembly_result_df
            ),

            len(
                component_result_df
            ),

            zero_mean,

            direct_mean,

            deterministic_mean,

            structured_mean,

            achievable_improvement,

            structured_capture,

            deterministic_capture,

            structured_regret,

            deterministic_regret,

            structured_relative_regret,

            deterministic_relative_regret,

            direct_beats_structured,

            direct_beats_deterministic,

            structured_near_direct_5pct,

            deterministic_near_direct_5pct,

            mean_direct_utilization,

            direct_near_limit,

            optimizer_success_rate,

            mean_optimizer_evaluations,
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

            "Structured-20",

            "Deterministic LSQ",

            "Direct simulator",
        ],

        "Mean final quality": [

            zero_mean,

            structured_mean,

            deterministic_mean,

            direct_mean,
        ],
    }
)


plt.figure(
    figsize=(
        9.0,
        5.5,
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
    "Achievable-Correction Reference"
)


plt.grid(
    axis="y",
    alpha=0.25,
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        FIGURE_DIR,
        "achievable_correction_comparison.png",
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
    "ACHIEVABLE-CORRECTION SUMMARY"
)

print(
    "============================================================"
)


print(
    "\nMean final quality:"
)


print(
    f"\nZero                    : "
    f"{zero_mean:.4f}"
)


print(
    f"Direct simulator        : "
    f"{direct_mean:.4f}"
)


print(
    f"Deterministic LSQ       : "
    f"{deterministic_mean:.4f}"
)


print(
    f"Structured-20           : "
    f"{structured_mean:.4f}"
)


print(
    "\nFraction of direct-simulator improvement captured:"
)


print(
    f"\nStructured-20           : "
    f"{structured_capture:.2f}%"
)


print(
    f"Deterministic LSQ       : "
    f"{deterministic_capture:.2f}%"
)


print(
    "\nRegret relative to direct simulator:"
)


print(
    f"\nStructured absolute     : "
    f"{structured_regret:.4f}"
)


print(
    f"Structured relative     : "
    f"{structured_relative_regret:.2f}%"
)


print(
    f"\nDeterministic absolute  : "
    f"{deterministic_regret:.4f}"
)


print(
    f"Deterministic relative  : "
    f"{deterministic_relative_regret:.2f}%"
)


print(
    "\nAssembly comparisons:"
)


print(
    f"\nDirect beats Structured : "
    f"{direct_beats_structured:.2f}%"
)


print(
    f"Direct beats LSQ        : "
    f"{direct_beats_deterministic:.2f}%"
)


print(
    f"Structured within 5% "
    f"of direct               : "
    f"{structured_near_direct_5pct:.2f}%"
)


print(
    f"LSQ within 5% of direct : "
    f"{deterministic_near_direct_5pct:.2f}%"
)


print(
    "\nDirect-simulator capability:"
)


print(
    f"\nMean utilization        : "
    f"{mean_direct_utilization:.3f}"
)


print(
    f">=90% utilization       : "
    f"{direct_near_limit:.2f}%"
)


print(
    f"Optimizer success       : "
    f"{optimizer_success_rate:.2f}%"
)


print(
    f"Mean simulator evals    : "
    f"{mean_optimizer_evaluations:.1f}"
)


print(
    "\nStage-wise results:"
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
# INTERPRETATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "AUDIT 3 VERDICT"
)

print(
    "============================================================"
)


if (
    structured_capture
    >=
    95.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "Structured-20 captures at least 95% of the "
        "improvement achieved by the simulator-direct "
        "reference."
    )


    print(
        "\nAlthough direct optimization can improve the "
        "simplified simulator further, Structured-20 is "
        "already close to the achievable greedy performance."
    )


elif (
    structured_capture
    >=
    85.0
):


    print(
        "\nRESULT B:"
    )


    print(
        "Structured-20 captures most, but not all, of the "
        "improvement available under direct simulator "
        "optimization."
    )


    print(
        "\nThere is measurable controller regret, while the "
        "overall adaptive recommendation remains effective."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "A substantial gap remains between Structured-20 "
        "and simulator-direct optimization."
    )


    print(
        "\nThe present simplified model therefore contains "
        "considerable correction potential that is not "
        "captured by the ML-supported controller."
    )


print(
    "\nIMPORTANT:"
)


print(
    "The direct simulator result is an approximate numerical "
    "reference, not a mathematical proof of a global optimum."
)


print(
    "Its purpose is to separate available correction potential "
    "from controller performance under the present model."
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
    "ACHIEVABLE-CORRECTION AUDIT COMPLETED"
)

print(
    "============================================================"
)