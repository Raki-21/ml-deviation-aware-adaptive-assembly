"""
V3.3 direct nonlinear reference stability audit.

Purpose
-------
Check whether the finite-budget Differential Evolution reference used
in V3.3 gives reasonably stable sequential assembly-quality estimates
when the optimizer budget is increased.

This is NOT a controller-selection experiment.

The already-locked hybrid controller is not changed.

Three direct numerical-reference budgets are compared:

    LOW     : maxiter=8,  popsize=5
    MEDIUM  : maxiter=18, popsize=7
    HIGH    : maxiter=30, popsize=10

The same geometric assemblies are replayed for every budget.

If higher budgets provide only small additional quality gains, the
finite-budget direct reference is sufficiently stable for contextual
comparison.

If substantial differences remain, direct-reference claims must be
reported more cautiously.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution

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

RESULTS_DIR = (
    V33_ROOT
    / "results"
    / "direct_reference_stability"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ASSEMBLY_OUTPUT = (
    RESULTS_DIR
    / "v33_direct_reference_stability_assembly_results.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "v33_direct_reference_stability_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = N_COMPONENTS

NONLINEARITY_STRENGTH = 1.0

N_ASSEMBLIES = 30

VALIDATION_SEED = 20260912

FIXTURE_DRIFT_STD_MM = 0.015


REFERENCE_BOUNDS = [
    (
        -Z_ADJ_LIMIT_MM,
        Z_ADJ_LIMIT_MM,
    ),
    (
        -THETA_ADJ_LIMIT_DEG,
        THETA_ADJ_LIMIT_DEG,
    ),
    (
        -LOCATOR_OFFSET_LIMIT_MM,
        LOCATOR_OFFSET_LIMIT_MM,
    ),
]


BUDGETS = [
    {
        "name":
            "LOW_8x5",

        "maxiter":
            8,

        "popsize":
            5,
    },

    {
        "name":
            "MEDIUM_18x7",

        "maxiter":
            18,

        "popsize":
            7,
    },

    {
        "name":
            "HIGH_30x10",

        "maxiter":
            30,

        "popsize":
            10,
    },
]


# ============================================================
# GEOMETRY GENERATION
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


def generate_cases():
    """
    Reproduce the first 30 assemblies from the final independent
    validation population.
    """

    rng = np.random.default_rng(
        VALIDATION_SEED
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
# APPLY TRUE NONLINEAR CORRECTION
# ============================================================

def apply_true_correction(
    pre_correction_state,
    point,
):

    correction = (
        correction_profile_nonlinear(
            previous_state=pre_correction_state,
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
    )


    return (
        pre_correction_state
        +
        correction
    )


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_reference(
    pre_correction_state,
    seed,
    maxiter,
    popsize,
):

    evaluation_counter = {
        "count":
            0
    }


    def objective(
        point,
    ):

        evaluation_counter[
            "count"
        ] += 1


        corrected_state = (
            apply_true_correction(
                pre_correction_state,
                point,
            )
        )


        return float(
            calculate_quality_metrics(
                corrected_state
            )[
                "quality_score"
            ]
        )


    result = differential_evolution(
        objective,
        bounds=REFERENCE_BOUNDS,
        seed=int(
            seed
        ),
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        polish=True,
        updating="immediate",
        workers=1,
    )


    point = np.asarray(
        result.x,
        dtype=float,
    )


    corrected_state = (
        apply_true_correction(
            pre_correction_state,
            point,
        )
    )


    return {
        "point":
            point,

        "state":
            corrected_state,

        "quality":
            float(
                calculate_quality_metrics(
                    corrected_state
                )[
                    "quality_score"
                ]
            ),

        "success":
            bool(
                result.success
            ),

        "evaluations":
            int(
                evaluation_counter[
                    "count"
                ]
            ),
    }


# ============================================================
# RUN ONE BUDGET
# ============================================================

def run_budget(
    cases,
    budget,
):

    assembly_rows = []

    total_success = 0

    total_decisions = 0

    total_evaluations = 0


    start_time = time.time()


    for assembly_case in cases:

        state = (
            create_initial_state()
        )


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


            pre_correction_state = (
                state
                +
                component_record[
                    "component_profile"
                ]
                +
                component_record[
                    "fixture_drift"
                ]
            )


            seed = (
                VALIDATION_SEED
                +
                assembly_case[
                    "assembly_index"
                ]
                *
                1000
                +
                component_index
            )


            result = optimize_reference(
                pre_correction_state=pre_correction_state,
                seed=seed,
                maxiter=budget[
                    "maxiter"
                ],
                popsize=budget[
                    "popsize"
                ],
            )


            state = (
                result[
                    "state"
                ]
            )


            total_success += int(
                result[
                    "success"
                ]
            )


            total_evaluations += int(
                result[
                    "evaluations"
                ]
            )


            total_decisions += 1


        final_quality = float(
            calculate_quality_metrics(
                state
            )[
                "quality_score"
            ]
        )


        assembly_rows.append(
            {
                "assembly_index":
                    assembly_case[
                        "assembly_index"
                    ],

                "budget":
                    budget[
                        "name"
                    ],

                "final_quality":
                    final_quality,
            }
        )


        if (
            assembly_case[
                "assembly_index"
            ]
            %
            5
            ==
            0
        ):

            print(
                f"{budget['name']}: "
                f"assembly "
                f"{assembly_case['assembly_index']}/"
                f"{N_ASSEMBLIES}"
            )


    runtime = float(
        time.time()
        -
        start_time
    )


    return (
        assembly_rows,
        {
            "budget":
                budget[
                    "name"
                ],

            "maxiter":
                budget[
                    "maxiter"
                ],

            "popsize":
                budget[
                    "popsize"
                ],

            "decisions":
                total_decisions,

            "mean_evaluations_per_decision":
                float(
                    total_evaluations
                    /
                    total_decisions
                ),

            "formal_success_percent":
                float(
                    100.0
                    *
                    total_success
                    /
                    total_decisions
                ),

            "runtime_seconds":
                runtime,
        }
    )


# ============================================================
# EXPERIMENT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 DIRECT REFERENCE STABILITY AUDIT"
)

print(
    "============================================================"
)


print(
    f"\nAssemblies      : {N_ASSEMBLIES}"
)

print(
    f"Sequence length : {SEQUENCE_LENGTH}"
)

print(
    f"Budgets tested  : {len(BUDGETS)}"
)


cases = (
    generate_cases()
)


all_assembly_rows = []

budget_metadata = []


for budget in BUDGETS:

    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        f"Running {budget['name']}"
    )

    print(
        f"maxiter={budget['maxiter']}, "
        f"popsize={budget['popsize']}"
    )

    print(
        "------------------------------------------------------------"
    )


    rows, metadata = run_budget(
        cases,
        budget,
    )


    all_assembly_rows.extend(
        rows
    )


    budget_metadata.append(
        metadata
    )


assembly_df = pd.DataFrame(
    all_assembly_rows
)


assembly_df.to_csv(
    ASSEMBLY_OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for metadata in budget_metadata:

    budget = (
        metadata[
            "budget"
        ]
    )


    values = (
        assembly_df[
            assembly_df[
                "budget"
            ]
            ==
            budget
        ][
            "final_quality"
        ]
        .to_numpy(
            dtype=float
        )
    )


    summary_rows.append(
        {
            **metadata,

            "mean_final_quality":
                float(
                    np.mean(
                        values
                    )
                ),

            "median_final_quality":
                float(
                    np.median(
                        values
                    )
                ),

            "p95_final_quality":
                float(
                    np.quantile(
                        values,
                        0.95,
                    )
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


pivot = assembly_df.pivot(
    index="assembly_index",
    columns="budget",
    values="final_quality",
)


low = (
    pivot[
        "LOW_8x5"
    ]
    .to_numpy(
        dtype=float
    )
)


medium = (
    pivot[
        "MEDIUM_18x7"
    ]
    .to_numpy(
        dtype=float
    )
)


high = (
    pivot[
        "HIGH_30x10"
    ]
    .to_numpy(
        dtype=float
    )
)


medium_minus_low = (
    medium
    -
    low
)


high_minus_medium = (
    high
    -
    medium
)


high_minus_low = (
    high
    -
    low
)


comparison_rows = pd.DataFrame(
    [
        {
            "comparison":
                "MEDIUM_MINUS_LOW",

            "mean_difference":
                float(
                    np.mean(
                        medium_minus_low
                    )
                ),

            "median_difference":
                float(
                    np.median(
                        medium_minus_low
                    )
                ),

            "higher_budget_better_percent":
                float(
                    100.0
                    *
                    np.mean(
                        medium_minus_low
                        <
                        0.0
                    )
                ),
        },

        {
            "comparison":
                "HIGH_MINUS_MEDIUM",

            "mean_difference":
                float(
                    np.mean(
                        high_minus_medium
                    )
                ),

            "median_difference":
                float(
                    np.median(
                        high_minus_medium
                    )
                ),

            "higher_budget_better_percent":
                float(
                    100.0
                    *
                    np.mean(
                        high_minus_medium
                        <
                        0.0
                    )
                ),
        },

        {
            "comparison":
                "HIGH_MINUS_LOW",

            "mean_difference":
                float(
                    np.mean(
                        high_minus_low
                    )
                ),

            "median_difference":
                float(
                    np.median(
                        high_minus_low
                    )
                ),

            "higher_budget_better_percent":
                float(
                    100.0
                    *
                    np.mean(
                        high_minus_low
                        <
                        0.0
                    )
                ),
        },
    ]
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


comparison_output = (
    RESULTS_DIR
    / "v33_direct_reference_budget_comparisons.csv"
)


comparison_rows.to_csv(
    comparison_output,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "REFERENCE BUDGET RESULTS"
)

print(
    "============================================================"
)


for _, row in summary_df.iterrows():

    print(
        f"\n{row['budget']}"
    )


    print(
        f"  Mean final quality        : "
        f"{row['mean_final_quality']:.6f}"
    )


    print(
        f"  Median final quality      : "
        f"{row['median_final_quality']:.6f}"
    )


    print(
        f"  P95 final quality         : "
        f"{row['p95_final_quality']:.6f}"
    )


    print(
        f"  Mean evaluations/decision : "
        f"{row['mean_evaluations_per_decision']:.1f}"
    )


    print(
        f"  Formal success            : "
        f"{row['formal_success_percent']:.2f}%"
    )


    print(
        f"  Runtime                    : "
        f"{row['runtime_seconds']:.1f} s"
    )


print(
    "\n"
    "============================================================"
)

print(
    "BUDGET STABILITY COMPARISONS"
)

print(
    "============================================================"
)


for _, row in comparison_rows.iterrows():

    print(
        f"\n{row['comparison']}"
    )


    print(
        f"  Mean difference           : "
        f"{row['mean_difference']:.6f}"
    )


    print(
        f"  Median difference         : "
        f"{row['median_difference']:.6f}"
    )


    print(
        f"  Higher-budget better      : "
        f"{row['higher_budget_better_percent']:.2f}%"
    )


high_mean = float(
    summary_df[
        summary_df[
            "budget"
        ]
        ==
        "HIGH_30x10"
    ][
        "mean_final_quality"
    ]
    .iloc[0]
)


medium_mean = float(
    summary_df[
        summary_df[
            "budget"
        ]
        ==
        "MEDIUM_18x7"
    ][
        "mean_final_quality"
    ]
    .iloc[0]
)


relative_high_vs_medium = float(
    100.0
    *
    (
        medium_mean
        -
        high_mean
    )
    /
    max(
        medium_mean,
        1e-12,
    )
)


print(
    "\n"
    "============================================================"
)

print(
    "REFERENCE STABILITY INTERPRETATION"
)

print(
    "============================================================"
)


print(
    f"\nAdditional improvement HIGH vs MEDIUM: "
    f"{relative_high_vs_medium:.3f}%"
)


if abs(
    relative_high_vs_medium
) < 1.0:

    print(
        "\nSTABLE:"
    )

    print(
        "Increasing the numerical-reference budget from MEDIUM "
        "to HIGH changes mean sequential quality by less than 1%."
    )


else:

    print(
        "\nBUDGET SENSITIVE:"
    )

    print(
        "The direct numerical-reference estimate remains "
        "meaningfully sensitive to optimizer budget."
    )


print(
    "\nImportant:"
)

print(
    "Formal Differential Evolution success is not interpreted "
    "alone as evidence of reference validity."
)

print(
    "The primary diagnostic is quality stability as numerical "
    "search effort is increased."
)


print(
    "\nSaved assembly results:"
)

print(
    ASSEMBLY_OUTPUT
)


print(
    "\nSaved summary:"
)

print(
    SUMMARY_OUTPUT
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 DIRECT REFERENCE STABILITY AUDIT COMPLETED"
)

print(
    "============================================================"
)