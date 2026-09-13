"""
Phase 8 development controller evaluation.

Purpose
-------
Evaluate the three Phase 8 controller architectures in closed loop:

1. Composite-Q
2. Composite-Q + Random Forest residual
3. Composite-Q + Gradient Boosting residual

Each controller follows its own sequential assembly trajectory.

This development evaluation is performed before the final Phase 8
population is evaluated.

No Version 3.3 or Phase 1-7 files are modified.
"""

from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PHASE 8 MODULES
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase8_config import (  # noqa: E402
    DEVELOPMENT_SEED,
    N_DEVELOPMENT_ASSEMBLIES,
    MODEL_DIR,
    TABLES_DIR,
    DEVELOPMENT_DATA_DIR,
    ensure_phase8_directories,
)

from generate_residual_learning_dataset import (  # noqa: E402
    generate_cases,
    deterministic_lsq_correction,
    optimize_composite_q,
    calculate_observable_features,
    apply_true_correction,
    calculate_utilization,
    clip_correction,
    calculate_quality_metrics,
    create_initial_state,
    OBSERVABLE_FEATURE_COLUMNS,
)


# ============================================================
# MODEL FILES
# ============================================================

RF_MODEL_FILES = {
    "delta_z":
        MODEL_DIR
        / "phase8_random_forest_delta_z.joblib",

    "delta_theta":
        MODEL_DIR
        / "phase8_random_forest_delta_theta.joblib",

    "delta_locator":
        MODEL_DIR
        / "phase8_random_forest_delta_locator.joblib",
}

GB_MODEL_FILES = {
    "delta_z":
        MODEL_DIR
        / "phase8_gradient_boosting_delta_z.joblib",

    "delta_theta":
        MODEL_DIR
        / "phase8_gradient_boosting_delta_theta.joblib",

    "delta_locator":
        MODEL_DIR
        / "phase8_gradient_boosting_delta_locator.joblib",
}


# ============================================================
# OUTPUT FILES
# ============================================================

COMPONENT_OUTPUT_FILE = (
    TABLES_DIR
    / "phase8_development_controller_component_results.csv"
)

ASSEMBLY_OUTPUT_FILE = (
    TABLES_DIR
    / "phase8_development_controller_assembly_results.csv"
)

SUMMARY_OUTPUT_FILE = (
    TABLES_DIR
    / "phase8_development_controller_summary.csv"
)


# ============================================================
# LOAD MODELS
# ============================================================

def load_model_family(model_files):
    models = {}

    for target_name, model_file in model_files.items():
        if not model_file.exists():
            raise FileNotFoundError(
                f"Required model file not found:\n{model_file}"
            )

        models[target_name] = joblib.load(
            model_file
        )

    return models


# ============================================================
# ML RESIDUAL PREDICTION
# ============================================================

def predict_residual(
    models,
    feature_record,
):
    feature_vector = np.array(
        [
            feature_record[column]
            for column in OBSERVABLE_FEATURE_COLUMNS
        ],
        dtype=float,
    ).reshape(
        1,
        -1,
    )

    delta_z = float(
        models["delta_z"].predict(
            feature_vector
        )[0]
    )

    delta_theta = float(
        models["delta_theta"].predict(
            feature_vector
        )[0]
    )

    delta_locator = float(
        models["delta_locator"].predict(
            feature_vector
        )[0]
    )

    return np.array(
        [
            delta_z,
            delta_theta,
            delta_locator,
        ],
        dtype=float,
    )


# ============================================================
# RUN ONE CONTROLLER FOR ONE ASSEMBLY
# ============================================================

def run_controller_on_assembly(
    case,
    method_name,
    models=None,
):
    state = create_initial_state()

    component_rows = []

    for component in case["components"]:
        component_index = int(
            component["component_index"]
        )

        component_profile = (
            component["component_profile"]
        )

        fixture_drift = (
            component["fixture_drift"]
        )

        pre_state = (
            state
            +
            component_profile
            +
            fixture_drift
        )

        pre_quality = float(
            calculate_quality_metrics(
                pre_state
            )["quality_score"]
        )

        # ----------------------------------------------------
        # Composite-Q base
        # ----------------------------------------------------

        lsq_point = (
            deterministic_lsq_correction(
                pre_state
            )
        )

        composite_q = (
            optimize_composite_q(
                pre_state,
                lsq_point,
            )
        )

        composite_q_point = np.asarray(
            composite_q["point"],
            dtype=float,
        )

        # ----------------------------------------------------
        # Same observable features used during training
        # ----------------------------------------------------

        feature_record = (
            calculate_observable_features(
                pre_correction_state=
                    pre_state,

                component_profile=
                    component_profile,

                component_index=
                    component_index,

                lsq_point=
                    lsq_point,
            )
        )

        # ----------------------------------------------------
        # Controller-specific correction
        # ----------------------------------------------------

        if method_name == "COMPOSITE_Q":
            predicted_residual = np.zeros(
                3,
                dtype=float,
            )

            final_point = (
                composite_q_point.copy()
            )

        else:
            predicted_residual = (
                predict_residual(
                    models,
                    feature_record,
                )
            )

            final_point = (
                composite_q_point
                +
                predicted_residual
            )

            final_point = (
                clip_correction(
                    final_point
                )
            )

        applied_residual = (
            final_point
            -
            composite_q_point
        )

        state = (
            apply_true_correction(
                pre_state,
                final_point,
            )
        )

        post_quality = float(
            calculate_quality_metrics(
                state
            )["quality_score"]
        )

        component_rows.append(
            {
                "assembly_index":
                    int(
                        case["assembly_index"]
                    ),

                "component_index":
                    component_index,

                "method":
                    method_name,

                "pre_quality":
                    pre_quality,

                "post_quality":
                    post_quality,

                "base_composite_q_z_adj_mm":
                    float(
                        composite_q_point[0]
                    ),

                "base_composite_q_theta_adj_deg":
                    float(
                        composite_q_point[1]
                    ),

                "base_composite_q_locator_offset_mm":
                    float(
                        composite_q_point[2]
                    ),

                "predicted_delta_z_mm":
                    float(
                        predicted_residual[0]
                    ),

                "predicted_delta_theta_deg":
                    float(
                        predicted_residual[1]
                    ),

                "predicted_delta_locator_mm":
                    float(
                        predicted_residual[2]
                    ),

                "applied_delta_z_mm":
                    float(
                        applied_residual[0]
                    ),

                "applied_delta_theta_deg":
                    float(
                        applied_residual[1]
                    ),

                "applied_delta_locator_mm":
                    float(
                        applied_residual[2]
                    ),

                "final_z_adj_mm":
                    float(
                        final_point[0]
                    ),

                "final_theta_adj_deg":
                    float(
                        final_point[1]
                    ),

                "final_locator_offset_mm":
                    float(
                        final_point[2]
                    ),

                "final_utilization":
                    calculate_utilization(
                        final_point
                    ),
            }
        )

    final_metrics = (
        calculate_quality_metrics(
            state
        )
    )

    assembly_row = {
        "assembly_index":
            int(
                case["assembly_index"]
            ),

        "method":
            method_name,

        "final_quality":
            float(
                final_metrics[
                    "quality_score"
                ]
            ),

        "final_mean_gap":
            float(
                final_metrics[
                    "mean_gap"
                ]
            ),

        "final_max_gap":
            float(
                final_metrics[
                    "max_gap"
                ]
            ),

        "final_parallelism":
            float(
                final_metrics[
                    "parallelism_error"
                ]
            ),

        "final_rms":
            float(
                final_metrics[
                    "rms_deviation"
                ]
            ),
    }

    return (
        component_rows,
        assembly_row,
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
        "PHASE 8: DEVELOPMENT CLOSED-LOOP CONTROLLER EVALUATION"
    )

    print(
        "============================================================"
    )

    rf_models = (
        load_model_family(
            RF_MODEL_FILES
        )
    )

    gb_models = (
        load_model_family(
            GB_MODEL_FILES
        )
    )

    cases = (
        generate_cases(
            split_name=
                "development_controller",

            n_assemblies=
                N_DEVELOPMENT_ASSEMBLIES,

            seed=
                DEVELOPMENT_SEED,
        )
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

    start_time = (
        time.time()
    )

    for assembly_number, case in enumerate(
        cases,
        start=1,
    ):
        for method_name, models in methods:
            (
                local_component_rows,
                local_assembly_row,
            ) = (
                run_controller_on_assembly(
                    case=
                        case,

                    method_name=
                        method_name,

                    models=
                        models,
                )
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
            assembly_number
            ==
            N_DEVELOPMENT_ASSEMBLIES
        ):
            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"completed "
                f"{assembly_number}/"
                f"{N_DEVELOPMENT_ASSEMBLIES} assemblies "
                f"| elapsed {elapsed:.1f} s"
            )

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
    # SUMMARY
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
    # PAIRED DEVELOPMENT COMPARISONS
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

    cq = (
        pivot[
            "COMPOSITE_Q"
        ]
    )

    rf = (
        pivot[
            "COMPOSITE_Q_PLUS_RF"
        ]
    )

    gb = (
        pivot[
            "COMPOSITE_Q_PLUS_GB"
        ]
    )

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

    print(
        "\n"
        "============================================================"
    )

    print(
        "DEVELOPMENT CONTROLLER RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        f"\nComposite-Q mean quality       : "
        f"{cq.mean():.9f}"
    )

    print(
        f"Composite-Q + RF mean quality  : "
        f"{rf.mean():.9f}"
    )

    print(
        f"Composite-Q + GB mean quality  : "
        f"{gb.mean():.9f}"
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

    elapsed_seconds = float(
        time.time()
        -
        start_time
    )

    print(
        f"\nRuntime: "
        f"{elapsed_seconds:.1f} s"
    )

    print(
        "\nSaved component results:"
    )

    print(
        COMPONENT_OUTPUT_FILE
    )

    print(
        "\nSaved assembly results:"
    )

    print(
        ASSEMBLY_OUTPUT_FILE
    )

    print(
        "\nSaved summary:"
    )

    print(
        SUMMARY_OUTPUT_FILE
    )

    print(
        "\nImportant:"
    )

    print(
        "This is a development-set closed-loop comparison."
    )

    print(
        "The final Phase 8 population has not been evaluated here."
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 DEVELOPMENT CONTROLLER EVALUATION COMPLETED"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()