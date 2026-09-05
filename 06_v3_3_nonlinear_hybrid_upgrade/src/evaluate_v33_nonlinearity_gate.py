"""
V3.3 nonlinearity gate evaluation.

Purpose
-------
Test whether the new nonlinear assembly response actually creates
meaningful analytical model mismatch for deterministic LSQ.

This script does NOT train ML.

It compares bounded analytical LSQ under controlled nonlinearity
levels using the same generated assemblies for every level.

The important scientific question is:

    Does LSQ performance degrade as the true assembly response
    becomes less consistent with the linear correction basis?

If not, the nonlinear extension is too weak or not meaningful.

The frozen V3.2 implementation is not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

from nonlinear_assembly_model import (
    create_initial_state,
    deviation_offset,
    deviation_tilt,
    deviation_bend,
    deviation_waviness,
    deviation_twist,
    deviation_local_bump,
    correction_profile_nonlinear,
    update_assembly_state,
    calculate_quality_metrics,
)

from v33_config import (
    N_COMPONENTS,
    REFERENCE_N_COMPONENTS,
    Z_ADJ_LIMIT_MM,
    THETA_ADJ_LIMIT_DEG,
    LOCATOR_OFFSET_LIMIT_MM,
    NONLINEARITY_LEVELS,
    BASE_SEED,
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
    / "nonlinearity_gate"
)

FIGURES_DIR = (
    V33_ROOT
    / "figures"
    / "nonlinearity_gate"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

N_ASSEMBLIES = 150

SEQUENCE_LENGTHS = [
    REFERENCE_N_COMPONENTS,
    N_COMPONENTS,
]

# Same assemblies are replayed under all nonlinearity levels.
RANDOM_SEED = BASE_SEED

# Small repeated process disturbance.
FIXTURE_DRIFT_STD_MM = 0.015

# Numerical LSQ solver starting point.
LSQ_INITIAL_GUESS = np.array(
    [
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)


# ============================================================
# RANDOM GENERATION
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


def generate_component_parameters():
    """
    Generate one controlled component deviation realization.

    These ranges are intentionally moderate and are used only
    for the V3.3 nonlinearity gate.
    """

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
    """
    Build one component deviation profile.
    """

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
# PRE-GENERATE IDENTICAL ASSEMBLIES
# ============================================================

def generate_experiment_cases(
    sequence_length,
):
    """
    Generate all assemblies once.

    The exact same cases will be replayed for every
    nonlinearity level so comparisons are paired.
    """

    cases = []

    for assembly_index in range(
        N_ASSEMBLIES
    ):

        component_records = []

        for component_index in range(
            sequence_length
        ):

            parameters = (
                generate_component_parameters()
            )

            component_profile = (
                build_component_profile(
                    parameters
                )
            )

            fixture_drift = rng.normal(
                loc=0.0,
                scale=FIXTURE_DRIFT_STD_MM,
                size=component_profile.shape,
            )

            component_records.append(
                {
                    "component_index":
                        component_index
                        +
                        1,

                    "parameters":
                        parameters,

                    "component_profile":
                        component_profile,

                    "fixture_drift":
                        fixture_drift,
                }
            )

        cases.append(
            {
                "assembly_index":
                    assembly_index
                    +
                    1,

                "components":
                    component_records,
            }
        )

    return cases


# ============================================================
# LSQ CORRECTION
# ============================================================

def solve_linear_lsq_correction(
    pre_correction_state,
):
    """
    Solve the bounded deterministic correction assuming the
    original linear correction response.

    Important:
    ----------
    The solver assumes the V3.2-style linear response even when
    the true simulator later applies a nonlinear response.

    This intentionally creates analytical model mismatch.
    """

    def residual_function(
        parameters,
    ):

        z_adj_mm = (
            parameters[0]
        )

        theta_adj_deg = (
            parameters[1]
        )

        locator_offset_mm = (
            parameters[2]
        )

        # The analytical LSQ model assumes zero nonlinearity.
        correction = (
            correction_profile_nonlinear(
                previous_state=pre_correction_state,
                z_adj_mm=z_adj_mm,
                theta_adj_deg=theta_adj_deg,
                locator_offset_mm=locator_offset_mm,
                nonlinearity_strength=0.0,
            )
        )

        corrected_state = (
            pre_correction_state
            +
            correction
        )

        return corrected_state


    lower_bounds = np.array(
        [
            -Z_ADJ_LIMIT_MM,
            -THETA_ADJ_LIMIT_DEG,
            -LOCATOR_OFFSET_LIMIT_MM,
        ],
        dtype=float,
    )

    upper_bounds = np.array(
        [
            Z_ADJ_LIMIT_MM,
            THETA_ADJ_LIMIT_DEG,
            LOCATOR_OFFSET_LIMIT_MM,
        ],
        dtype=float,
    )


    solution = least_squares(
        residual_function,
        x0=LSQ_INITIAL_GUESS,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        method="trf",
        max_nfev=100,
    )


    return {
        "z_adj_mm":
            float(
                solution.x[0]
            ),

        "theta_adj_deg":
            float(
                solution.x[1]
            ),

        "locator_offset_mm":
            float(
                solution.x[2]
            ),

        "solver_success":
            bool(
                solution.success
            ),

        "solver_cost":
            float(
                solution.cost
            ),
    }


# ============================================================
# RUN ONE ASSEMBLY
# ============================================================

def run_one_assembly(
    assembly_case,
    nonlinearity_strength,
):
    """
    Run one complete sequential assembly.

    LSQ computes its correction using the linear model.

    The actual correction effect is then applied through the
    nonlinear V3.3 simulator.
    """

    zero_state = (
        create_initial_state()
    )

    lsq_state = (
        create_initial_state()
    )

    component_results = []


    for component_record in (
        assembly_case[
            "components"
        ]
    ):

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

        component_index = (
            component_record[
                "component_index"
            ]
        )


        # ----------------------------------------------------
        # ZERO CORRECTION PATH
        # ----------------------------------------------------

        zero_state = (
            update_assembly_state(
                previous_state=zero_state,
                component_deviation=component_profile,
                correction=None,
                fixture_drift=fixture_drift,
            )
        )


        # ----------------------------------------------------
        # LSQ PRE-CORRECTION STATE
        # ----------------------------------------------------

        pre_correction_state = (
            lsq_state
            +
            component_profile
            +
            fixture_drift
        )


        # ----------------------------------------------------
        # LINEAR LSQ DECISION
        # ----------------------------------------------------

        lsq_solution = (
            solve_linear_lsq_correction(
                pre_correction_state
            )
        )


        # ----------------------------------------------------
        # TRUE NONLINEAR CORRECTION RESPONSE
        # ----------------------------------------------------

        true_correction = (
            correction_profile_nonlinear(
                previous_state=pre_correction_state,
                z_adj_mm=lsq_solution[
                    "z_adj_mm"
                ],
                theta_adj_deg=lsq_solution[
                    "theta_adj_deg"
                ],
                locator_offset_mm=lsq_solution[
                    "locator_offset_mm"
                ],
                nonlinearity_strength=nonlinearity_strength,
            )
        )


        # ----------------------------------------------------
        # UPDATE LSQ PATH
        # ----------------------------------------------------

        lsq_state = (
            update_assembly_state(
                previous_state=lsq_state,
                component_deviation=component_profile,
                correction=true_correction,
                fixture_drift=fixture_drift,
            )
        )


        zero_quality = (
            calculate_quality_metrics(
                zero_state
            )[
                "quality_score"
            ]
        )

        lsq_quality = (
            calculate_quality_metrics(
                lsq_state
            )[
                "quality_score"
            ]
        )


        component_results.append(
            {
                "assembly_index":
                    assembly_case[
                        "assembly_index"
                    ],

                "component_index":
                    component_index,

                "nonlinearity_strength":
                    nonlinearity_strength,

                "zero_quality":
                    zero_quality,

                "lsq_quality":
                    lsq_quality,

                "z_adj_mm":
                    lsq_solution[
                        "z_adj_mm"
                    ],

                "theta_adj_deg":
                    lsq_solution[
                        "theta_adj_deg"
                    ],

                "locator_offset_mm":
                    lsq_solution[
                        "locator_offset_mm"
                    ],

                "solver_success":
                    lsq_solution[
                        "solver_success"
                    ],
            }
        )


    final_zero_quality = (
        calculate_quality_metrics(
            zero_state
        )[
            "quality_score"
        ]
    )

    final_lsq_quality = (
        calculate_quality_metrics(
            lsq_state
        )[
            "quality_score"
        ]
    )


    improvement_percent = (
        100.0
        *
        (
            final_zero_quality
            -
            final_lsq_quality
        )
        /
        max(
            final_zero_quality,
            1e-12,
        )
    )


    return {
        "assembly_index":
            assembly_case[
                "assembly_index"
            ],

        "nonlinearity_strength":
            nonlinearity_strength,

        "zero_final_quality":
            final_zero_quality,

        "lsq_final_quality":
            final_lsq_quality,

        "lsq_improvement_percent":
            improvement_percent,

        "lsq_beats_zero":
            bool(
                final_lsq_quality
                <
                final_zero_quality
            ),

        "component_results":
            component_results,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

all_assembly_results = []

all_component_results = []

summary_rows = []


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 NONLINEARITY GATE EVALUATION"
)

print(
    "============================================================"
)

print(
    f"\nAssemblies per configuration : {N_ASSEMBLIES}"
)

print(
    f"Sequence lengths             : {SEQUENCE_LENGTHS}"
)

print(
    f"Nonlinearity levels          : {NONLINEARITY_LEVELS}"
)


for sequence_length in (
    SEQUENCE_LENGTHS
):

    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        f"Generating paired cases for K = {sequence_length}"
    )

    print(
        "------------------------------------------------------------"
    )


    experiment_cases = (
        generate_experiment_cases(
            sequence_length
        )
    )


    for nonlinearity_strength in (
        NONLINEARITY_LEVELS
    ):

        print(
            f"\nRunning K={sequence_length}, "
            f"alpha={nonlinearity_strength:.2f}"
        )


        configuration_results = []


        for assembly_case in (
            experiment_cases
        ):

            result = (
                run_one_assembly(
                    assembly_case=assembly_case,
                    nonlinearity_strength=nonlinearity_strength,
                )
            )


            assembly_record = {
                key:
                    value

                for key, value
                in result.items()

                if key
                !=
                "component_results"
            }


            assembly_record[
                "sequence_length"
            ] = (
                sequence_length
            )


            all_assembly_results.append(
                assembly_record
            )


            for component_record in (
                result[
                    "component_results"
                ]
            ):

                component_record[
                    "sequence_length"
                ] = (
                    sequence_length
                )

                all_component_results.append(
                    component_record
                )


            configuration_results.append(
                assembly_record
            )


        config_df = pd.DataFrame(
            configuration_results
        )


        mean_zero = float(
            config_df[
                "zero_final_quality"
            ]
            .mean()
        )


        mean_lsq = float(
            config_df[
                "lsq_final_quality"
            ]
            .mean()
        )


        median_lsq = float(
            config_df[
                "lsq_final_quality"
            ]
            .median()
        )


        p95_lsq = float(
            config_df[
                "lsq_final_quality"
            ]
            .quantile(
                0.95
            )
        )


        mean_improvement = float(
            config_df[
                "lsq_improvement_percent"
            ]
            .mean()
        )


        win_rate = float(
            100.0
            *
            config_df[
                "lsq_beats_zero"
            ]
            .mean()
        )


        summary_rows.append(
            {
                "sequence_length":
                    sequence_length,

                "nonlinearity_strength":
                    nonlinearity_strength,

                "assemblies":
                    N_ASSEMBLIES,

                "mean_zero_quality":
                    mean_zero,

                "mean_lsq_quality":
                    mean_lsq,

                "median_lsq_quality":
                    median_lsq,

                "p95_lsq_quality":
                    p95_lsq,

                "mean_lsq_improvement_percent":
                    mean_improvement,

                "lsq_beats_zero_percent":
                    win_rate,
            }
        )


        print(
            f"Mean zero quality : "
            f"{mean_zero:.4f}"
        )

        print(
            f"Mean LSQ quality  : "
            f"{mean_lsq:.4f}"
        )

        print(
            f"LSQ improvement   : "
            f"{mean_improvement:.2f}%"
        )

        print(
            f"LSQ beats zero    : "
            f"{win_rate:.2f}%"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

assembly_df = pd.DataFrame(
    all_assembly_results
)

component_df = pd.DataFrame(
    all_component_results
)

summary_df = pd.DataFrame(
    summary_rows
)


assembly_output = (
    RESULTS_DIR
    / "v33_nonlinearity_gate_assembly_results.csv"
)

component_output = (
    RESULTS_DIR
    / "v33_nonlinearity_gate_component_results.csv"
)

summary_output = (
    RESULTS_DIR
    / "v33_nonlinearity_gate_summary.csv"
)


assembly_df.to_csv(
    assembly_output,
    index=False,
)

component_df.to_csv(
    component_output,
    index=False,
)

summary_df.to_csv(
    summary_output,
    index=False,
)


# ============================================================
# RELATIVE LSQ DEGRADATION
# ============================================================

degradation_rows = []


for sequence_length in (
    SEQUENCE_LENGTHS
):

    sequence_summary = (
        summary_df[
            summary_df[
                "sequence_length"
            ]
            ==
            sequence_length
        ]
        .sort_values(
            "nonlinearity_strength"
        )
        .copy()
    )


    baseline_row = (
        sequence_summary[
            sequence_summary[
                "nonlinearity_strength"
            ]
            ==
            0.0
        ]
        .iloc[0]
    )


    baseline_lsq = float(
        baseline_row[
            "mean_lsq_quality"
        ]
    )


    for _, row in (
        sequence_summary
        .iterrows()
    ):

        current_lsq = float(
            row[
                "mean_lsq_quality"
            ]
        )


        relative_degradation = (
            100.0
            *
            (
                current_lsq
                -
                baseline_lsq
            )
            /
            max(
                baseline_lsq,
                1e-12,
            )
        )


        degradation_rows.append(
            {
                "sequence_length":
                    sequence_length,

                "nonlinearity_strength":
                    float(
                        row[
                            "nonlinearity_strength"
                        ]
                    ),

                "baseline_linear_lsq_quality":
                    baseline_lsq,

                "current_lsq_quality":
                    current_lsq,

                "relative_lsq_degradation_percent":
                    relative_degradation,
            }
        )


degradation_df = pd.DataFrame(
    degradation_rows
)


degradation_output = (
    RESULTS_DIR
    / "v33_nonlinearity_gate_lsq_degradation.csv"
)


degradation_df.to_csv(
    degradation_output,
    index=False,
)


# ============================================================
# PLOT
# ============================================================

for sequence_length in (
    SEQUENCE_LENGTHS
):

    plot_df = (
        summary_df[
            summary_df[
                "sequence_length"
            ]
            ==
            sequence_length
        ]
        .sort_values(
            "nonlinearity_strength"
        )
    )


    plt.figure(
        figsize=(
            7.5,
            4.5,
        )
    )


    plt.plot(
        plot_df[
            "nonlinearity_strength"
        ],
        plot_df[
            "mean_lsq_quality"
        ],
        marker="o",
        label="LSQ",
    )


    plt.plot(
        plot_df[
            "nonlinearity_strength"
        ],
        plot_df[
            "mean_zero_quality"
        ],
        marker="o",
        label="Zero correction",
    )


    plt.xlabel(
        "Nonlinearity strength"
    )

    plt.ylabel(
        "Mean final quality score"
    )

    plt.title(
        f"V3.3 LSQ Nonlinearity Gate - K={sequence_length}"
    )

    plt.legend()

    plt.tight_layout()


    figure_output = (
        FIGURES_DIR
        /
        (
            f"v33_lsq_nonlinearity_gate_K"
            f"{sequence_length}.png"
        )
    )


    plt.savefig(
        figure_output,
        dpi=250,
    )

    plt.close()


# ============================================================
# GATE INTERPRETATION
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "NONLINEARITY GATE INTERPRETATION"
)

print(
    "============================================================"
)


for sequence_length in (
    SEQUENCE_LENGTHS
):

    sequence_degradation = (
        degradation_df[
            degradation_df[
                "sequence_length"
            ]
            ==
            sequence_length
        ]
        .sort_values(
            "nonlinearity_strength"
        )
    )


    highest_row = (
        sequence_degradation
        .iloc[-1]
    )


    degradation_percent = float(
        highest_row[
            "relative_lsq_degradation_percent"
        ]
    )


    print(
        f"\nK={sequence_length}"
    )

    print(
        f"Linear LSQ quality       : "
        f"{highest_row['baseline_linear_lsq_quality']:.4f}"
    )

    print(
        f"Highest-alpha LSQ quality: "
        f"{highest_row['current_lsq_quality']:.4f}"
    )

    print(
        f"Relative LSQ degradation : "
        f"{degradation_percent:.2f}%"
    )


    if degradation_percent >= 10.0:

        print(
            "Gate status              : STRONG EFFECT"
        )

    elif degradation_percent >= 3.0:

        print(
            "Gate status              : MEANINGFUL EFFECT"
        )

    else:

        print(
            "Gate status              : WEAK EFFECT"
        )


print(
    "\nImportant:"
)

print(
    "This gate does NOT require ML to win."
)

print(
    "It only checks whether the nonlinear simulator creates "
    "measurable mismatch for the analytical LSQ assumption."
)


print(
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)


print(
    "\nFigures saved to:"
)

print(
    FIGURES_DIR
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 NONLINEARITY GATE COMPLETED"
)

print(
    "============================================================"
)