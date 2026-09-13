"""
Phase 8 final independent controller evaluation.

Purpose
-------
Perform the one-shot final evaluation of the three locked Phase 8
controller architectures:

1. Composite-Q
2. Composite-Q + Random Forest residual
3. Composite-Q + Gradient Boosting residual

The final evaluation uses the independent Phase 8 final population
defined in phase8_config.py.

No model training, feature selection, hyperparameter tuning, target
modification, or controller redesign is performed in this script.

Each controller follows its own complete sequential assembly trajectory.

The completed Version 3.3 framework and post-freeze Phase 1-7 results
remain unchanged.
"""

from pathlib import Path
import sys
import time

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# PHASE 8 CONFIGURATION
# ============================================================

from phase8_config import (  # noqa: E402
    FINAL_SEED,
    N_FINAL_ASSEMBLIES,
    TABLES_DIR,
    ensure_phase8_directories,
)


# ============================================================
# REUSE LOCKED PHASE 8 DEVELOPMENT-EVALUATION FUNCTIONS
# ============================================================

from evaluate_phase8_controllers import (  # noqa: E402
    RF_MODEL_FILES,
    GB_MODEL_FILES,
    load_model_family,
    run_controller_on_assembly,
)

from generate_residual_learning_dataset import (  # noqa: E402
    generate_cases,
)


# ============================================================
# OUTPUT FILES
# ============================================================

COMPONENT_OUTPUT_FILE = (
    TABLES_DIR
    / "phase8_final_controller_component_results.csv"
)

ASSEMBLY_OUTPUT_FILE = (
    TABLES_DIR
    / "phase8_final_controller_assembly_results.csv"
)

SUMMARY_OUTPUT_FILE = (
    TABLES_DIR
    / "phase8_final_controller_summary.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_phase8_directories()

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8: FINAL INDEPENDENT CONTROLLER EVALUATION"
    )

    print(
        "============================================================"
    )

    print(
        f"\nFinal assemblies : {N_FINAL_ASSEMBLIES}"
    )

    print(
        f"Final seed       : {FINAL_SEED}"
    )

    print(
        "\nMethods:"
    )

    print(
        "  1. Composite-Q"
    )

    print(
        "  2. Composite-Q + Random Forest residual"
    )

    print(
        "  3. Composite-Q + Gradient Boosting residual"
    )

    print(
        "\nNo tuning is performed in this run."
    )

    # ========================================================
    # LOAD ALREADY-TRAINED MODELS
    # ========================================================

    rf_models = load_model_family(
        RF_MODEL_FILES
    )

    gb_models = load_model_family(
        GB_MODEL_FILES
    )

    # ========================================================
    # GENERATE THE UNTOUCHED FINAL POPULATION
    # ========================================================

    cases = generate_cases(
        split_name=
            "final",

        n_assemblies=
            N_FINAL_ASSEMBLIES,

        seed=
            FINAL_SEED,
    )

    methods = [
        (
            "COMPOSITE_Q",
            None,
        ),
        (
            "COMPOSITE_Q_PLUS_RF",
            rf_models,
        ),
        (
            "COMPOSITE_Q_PLUS_GB",
            gb_models,
        ),
    ]

    component_rows = []
    assembly_rows = []

    start_time = time.time()

    # ========================================================
    # COMPLETE CLOSED-LOOP REPLAY
    # ========================================================

    for assembly_number, case in enumerate(
        cases,
        start=1,
    ):

        for method_name, models in methods:

            (
                local_component_rows,
                local_assembly_row,
            ) = run_controller_on_assembly(
                case=
                    case,

                method_name=
                    method_name,

                models=
                    models,
            )

            component_rows.extend(
                local_component_rows
            )

            assembly_rows.append(
                local_assembly_row
            )

        if (
            assembly_number == 1
            or
            assembly_number % 10 == 0
            or
            assembly_number == N_FINAL_ASSEMBLIES
        ):

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"completed "
                f"{assembly_number}/"
                f"{N_FINAL_ASSEMBLIES} assemblies "
                f"| elapsed {elapsed:.1f} s"
            )

    # ========================================================
    # DATA TABLES
    # ========================================================

    component_df = pd.DataFrame(
        component_rows
    )

    assembly_df = pd.DataFrame(
        assembly_rows
    )

    component_df.to_csv(
        COMPONENT_OUTPUT_FILE,
        index=False,
    )

    assembly_df.to_csv(
        ASSEMBLY_OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # METHOD SUMMARY
    # ========================================================

    summary_rows = []

    for method_name in [
        "COMPOSITE_Q",
        "COMPOSITE_Q_PLUS_RF",
        "COMPOSITE_Q_PLUS_GB",
    ]:

        method_df = (
            assembly_df[
                assembly_df[
                    "method"
                ]
                ==
                method_name
            ]
        )

        summary_rows.append(
            {
                "method":
                    method_name,

                "n_assemblies":
                    len(
                        method_df
                    ),

                "mean_final_quality":
                    float(
                        method_df[
                            "final_quality"
                        ].mean()
                    ),

                "median_final_quality":
                    float(
                        method_df[
                            "final_quality"
                        ].median()
                    ),

                "mean_final_mean_gap":
                    float(
                        method_df[
                            "final_mean_gap"
                        ].mean()
                    ),

                "mean_final_max_gap":
                    float(
                        method_df[
                            "final_max_gap"
                        ].mean()
                    ),

                "mean_final_parallelism":
                    float(
                        method_df[
                            "final_parallelism"
                        ].mean()
                    ),

                "mean_final_rms":
                    float(
                        method_df[
                            "final_rms"
                        ].mean()
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PAIRED FINAL COMPARISON
    # ========================================================

    pivot = (
        assembly_df.pivot(
            index=
                "assembly_index",

            columns=
                "method",

            values=
                "final_quality",
        )
    )

    expected_methods = {
        "COMPOSITE_Q",
        "COMPOSITE_Q_PLUS_RF",
        "COMPOSITE_Q_PLUS_GB",
    }

    missing_methods = (
        expected_methods
        -
        set(
            pivot.columns
        )
    )

    if missing_methods:
        raise RuntimeError(
            "Missing final methods: "
            f"{sorted(missing_methods)}"
        )

    if len(
        pivot
    ) != N_FINAL_ASSEMBLIES:
        raise RuntimeError(
            "Final assembly count mismatch: "
            f"{len(pivot)} != {N_FINAL_ASSEMBLIES}"
        )

    cq = pivot[
        "COMPOSITE_Q"
    ]

    rf = pivot[
        "COMPOSITE_Q_PLUS_RF"
    ]

    gb = pivot[
        "COMPOSITE_Q_PLUS_GB"
    ]

    rf_minus_cq = (
        rf
        -
        cq
    )

    gb_minus_cq = (
        gb
        -
        cq
    )

    rf_minus_gb = (
        rf
        -
        gb
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    elapsed_seconds = float(
        time.time()
        -
        start_time
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "FINAL PHASE 8 CONTROLLER RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        f"\nComposite-Q mean quality       : "
        f"{cq.mean():.12f}"
    )

    print(
        f"Composite-Q + RF mean quality  : "
        f"{rf.mean():.12f}"
    )

    print(
        f"Composite-Q + GB mean quality  : "
        f"{gb.mean():.12f}"
    )

    print(
        "\nMedian final quality:"
    )

    print(
        f"Composite-Q       : "
        f"{cq.median():.12f}"
    )

    print(
        f"Composite-Q + RF  : "
        f"{rf.median():.12f}"
    )

    print(
        f"Composite-Q + GB  : "
        f"{gb.median():.12f}"
    )

    print(
        "\nPaired mean differences "
        "(negative means first method is better):"
    )

    print(
        f"RF - Composite-Q : "
        f"{rf_minus_cq.mean():.12f}"
    )

    print(
        f"GB - Composite-Q : "
        f"{gb_minus_cq.mean():.12f}"
    )

    print(
        f"RF - GB          : "
        f"{rf_minus_gb.mean():.12f}"
    )

    print(
        "\nAssembly win rates:"
    )

    print(
        f"RF beats Composite-Q : "
        f"{100.0 * (rf_minus_cq < -1e-12).mean():.2f}%"
    )

    print(
        f"GB beats Composite-Q : "
        f"{100.0 * (gb_minus_cq < -1e-12).mean():.2f}%"
    )

    print(
        f"RF beats GB          : "
        f"{100.0 * (rf_minus_gb < -1e-12).mean():.2f}%"
    )

    print(
        "\nAssembly loss rates:"
    )

    print(
        f"RF worse than Composite-Q : "
        f"{100.0 * (rf_minus_cq > 1e-12).mean():.2f}%"
    )

    print(
        f"GB worse than Composite-Q : "
        f"{100.0 * (gb_minus_cq > 1e-12).mean():.2f}%"
    )

    print(
        f"\nRuntime: "
        f"{elapsed_seconds:.1f} s"
    )

    print(
        "\nSaved final component results:"
    )

    print(
        COMPONENT_OUTPUT_FILE
    )

    print(
        "\nSaved final assembly results:"
    )

    print(
        ASSEMBLY_OUTPUT_FILE
    )

    print(
        "\nSaved final summary:"
    )

    print(
        SUMMARY_OUTPUT_FILE
    )

    print(
        "\nImportant:"
    )

    print(
        "These are the untouched Phase 8 final-population results."
    )

    print(
        "No model or controller modification should be made after "
        "observing these results."
    )

    print(
        "\nStatistical testing is performed separately."
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 FINAL CONTROLLER EVALUATION COMPLETED"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()