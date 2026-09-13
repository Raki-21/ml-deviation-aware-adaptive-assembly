"""
Phase 8 final integrity validation.

Purpose
-------
Verify that the completed Objective-Aligned Residual Learning experiment
is internally consistent before final documentation and freeze.

This script performs read-only checks on the completed Phase 8 outputs.

It verifies:

- expected training and development row counts;
- expected final assembly and component result counts;
- expected controller methods;
- absence of missing/non-finite values;
- uniqueness of assembly/method and component/method records;
- correction-bound compliance;
- best-reference safety relative to Composite-Q;
- existence of all six trained ML models;
- existence and structure of the final statistical results;
- consistency of the Phase 8 random seeds;
- expected observable feature count.

No models are trained.
No optimization is performed.
No previous Phase 8 result is modified.
No Version 3.3 or Phase 1-7 file is modified.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PHASE 8 CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase8_config import (  # noqa: E402
    TRAINING_DATA_DIR,
    DEVELOPMENT_DATA_DIR,
    MODEL_DIR,
    TABLES_DIR,
    N_TRAIN_ASSEMBLIES,
    N_DEVELOPMENT_ASSEMBLIES,
    N_FINAL_ASSEMBLIES,
    TRAIN_SEED,
    DEVELOPMENT_SEED,
    FINAL_SEED,
    RF_RANDOM_STATE,
    GB_RANDOM_STATE,
    BOOTSTRAP_RANDOM_STATE,
)

from generate_residual_learning_dataset import (  # noqa: E402
    SEQUENCE_LENGTH,
    OBSERVABLE_FEATURE_COLUMNS,
    Z_ADJ_LIMIT_MM,
    THETA_ADJ_LIMIT_DEG,
    LOCATOR_OFFSET_LIMIT_MM,
)


# ============================================================
# EXPECTED FILES
# ============================================================

TRAINING_FILE = (
    TRAINING_DATA_DIR
    / "phase8_residual_learning_training.csv"
)

DEVELOPMENT_FILE = (
    DEVELOPMENT_DATA_DIR
    / "phase8_residual_learning_development.csv"
)

FEATURE_FILE = (
    TRAINING_DATA_DIR
    / "phase8_observable_features.txt"
)

MODEL_FILES = [
    MODEL_DIR / "phase8_random_forest_delta_z.joblib",
    MODEL_DIR / "phase8_random_forest_delta_theta.joblib",
    MODEL_DIR / "phase8_random_forest_delta_locator.joblib",
    MODEL_DIR / "phase8_gradient_boosting_delta_z.joblib",
    MODEL_DIR / "phase8_gradient_boosting_delta_theta.joblib",
    MODEL_DIR / "phase8_gradient_boosting_delta_locator.joblib",
]

MODEL_METRICS_FILE = (
    TABLES_DIR
    / "phase8_residual_model_metrics.csv"
)

DEVELOPMENT_CONTROLLER_FILE = (
    TABLES_DIR
    / "phase8_development_controller_assembly_results.csv"
)

FINAL_COMPONENT_FILE = (
    TABLES_DIR
    / "phase8_final_controller_component_results.csv"
)

FINAL_ASSEMBLY_FILE = (
    TABLES_DIR
    / "phase8_final_controller_assembly_results.csv"
)

FINAL_SUMMARY_FILE = (
    TABLES_DIR
    / "phase8_final_controller_summary.csv"
)

STATISTICS_FILE = (
    TABLES_DIR
    / "phase8_final_statistical_results.csv"
)

TARGET_GENERATION_SUMMARY_FILE = (
    TABLES_DIR
    / "phase8_target_generation_summary.csv"
)


# ============================================================
# EXPECTED METHODS
# ============================================================

EXPECTED_METHODS = {
    "COMPOSITE_Q",
    "COMPOSITE_Q_PLUS_RF",
    "COMPOSITE_Q_PLUS_GB",
}


# ============================================================
# CHECK HELPERS
# ============================================================

checks = []


def record_check(
    name,
    passed,
    details,
):
    checks.append(
        {
            "check":
                name,

            "status":
                "PASS"
                if passed
                else
                "FAIL",

            "details":
                str(
                    details
                ),
        }
    )

    status_text = (
        "PASS"
        if passed
        else
        "FAIL"
    )

    print(
        f"{status_text:4s} | "
        f"{name} | "
        f"{details}"
    )


def require_file(
    file_path,
    label,
):
    exists = (
        file_path.exists()
    )

    record_check(
        f"file exists: {label}",
        exists,
        file_path,
    )

    return exists


def dataframe_is_finite(
    df,
    columns,
):
    values = (
        df[
            columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    return bool(
        np.isfinite(
            values
        ).all()
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8: FINAL INTEGRITY VALIDATION"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # FILE EXISTENCE
    # --------------------------------------------------------

    print(
        "\n[1] Required files"
    )

    required_files = [
        (
            TRAINING_FILE,
            "training dataset",
        ),
        (
            DEVELOPMENT_FILE,
            "development dataset",
        ),
        (
            FEATURE_FILE,
            "observable feature list",
        ),
        (
            MODEL_METRICS_FILE,
            "model metrics",
        ),
        (
            DEVELOPMENT_CONTROLLER_FILE,
            "development controller results",
        ),
        (
            FINAL_COMPONENT_FILE,
            "final component results",
        ),
        (
            FINAL_ASSEMBLY_FILE,
            "final assembly results",
        ),
        (
            FINAL_SUMMARY_FILE,
            "final controller summary",
        ),
        (
            STATISTICS_FILE,
            "final statistical results",
        ),
        (
            TARGET_GENERATION_SUMMARY_FILE,
            "target-generation summary",
        ),
    ]

    all_required_files_exist = True

    for file_path, label in required_files:
        if not require_file(
            file_path,
            label,
        ):
            all_required_files_exist = False

    for model_file in MODEL_FILES:
        if not require_file(
            model_file,
            model_file.name,
        ):
            all_required_files_exist = False

    if not all_required_files_exist:
        raise RuntimeError(
            "Required Phase 8 files are missing. "
            "Integrity validation cannot continue."
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAINING_FILE
    )

    development_df = pd.read_csv(
        DEVELOPMENT_FILE
    )

    final_component_df = pd.read_csv(
        FINAL_COMPONENT_FILE
    )

    final_assembly_df = pd.read_csv(
        FINAL_ASSEMBLY_FILE
    )

    statistics_df = pd.read_csv(
        STATISTICS_FILE
    )

    model_metrics_df = pd.read_csv(
        MODEL_METRICS_FILE
    )

    # --------------------------------------------------------
    # SEEDS
    # --------------------------------------------------------

    print(
        "\n[2] Seed consistency"
    )

    seed_values = [
        TRAIN_SEED,
        DEVELOPMENT_SEED,
        FINAL_SEED,
        RF_RANDOM_STATE,
        GB_RANDOM_STATE,
        BOOTSTRAP_RANDOM_STATE,
    ]

    seeds_unique = (
        len(
            seed_values
        )
        ==
        len(
            set(
                seed_values
            )
        )
    )

    record_check(
        "all Phase 8 configured seeds are distinct",
        seeds_unique,
        seed_values,
    )

    record_check(
        "training/development/final seeds differ",
        len(
            {
                TRAIN_SEED,
                DEVELOPMENT_SEED,
                FINAL_SEED,
            }
        )
        ==
        3,
        (
            f"train={TRAIN_SEED}, "
            f"development={DEVELOPMENT_SEED}, "
            f"final={FINAL_SEED}"
        ),
    )

    # --------------------------------------------------------
    # FEATURE SET
    # --------------------------------------------------------

    print(
        "\n[3] Observable feature definition"
    )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        saved_features = [
            line.strip()
            for line in file
            if line.strip()
        ]

    record_check(
        "observable feature count",
        len(
            saved_features
        )
        ==
        26,
        f"{len(saved_features)} features",
    )

    record_check(
        "saved feature list matches code",
        saved_features
        ==
        list(
            OBSERVABLE_FEATURE_COLUMNS
        ),
        "saved list compared with current Phase 8 definition",
    )

    # --------------------------------------------------------
    # TRAINING / DEVELOPMENT DATA
    # --------------------------------------------------------

    print(
        "\n[4] Training and development datasets"
    )

    expected_train_rows = (
        N_TRAIN_ASSEMBLIES
        *
        SEQUENCE_LENGTH
    )

    expected_development_rows = (
        N_DEVELOPMENT_ASSEMBLIES
        *
        SEQUENCE_LENGTH
    )

    record_check(
        "training row count",
        len(
            train_df
        )
        ==
        expected_train_rows,
        (
            f"{len(train_df)} / "
            f"{expected_train_rows}"
        ),
    )

    record_check(
        "development row count",
        len(
            development_df
        )
        ==
        expected_development_rows,
        (
            f"{len(development_df)} / "
            f"{expected_development_rows}"
        ),
    )

    record_check(
        "training dataset has no NaN",
        not train_df.isna().any().any(),
        (
            f"NaN cells = "
            f"{int(train_df.isna().sum().sum())}"
        ),
    )

    record_check(
        "development dataset has no NaN",
        not development_df.isna().any().any(),
        (
            f"NaN cells = "
            f"{int(development_df.isna().sum().sum())}"
        ),
    )

    train_keys_unique = (
        not train_df.duplicated(
            subset=[
                "assembly_index",
                "component_index",
            ]
        ).any()
    )

    development_keys_unique = (
        not development_df.duplicated(
            subset=[
                "assembly_index",
                "component_index",
            ]
        ).any()
    )

    record_check(
        "training decision keys unique",
        train_keys_unique,
        "assembly_index + component_index",
    )

    record_check(
        "development decision keys unique",
        development_keys_unique,
        "assembly_index + component_index",
    )

    target_columns = [
        "target_delta_z_mm",
        "target_delta_theta_deg",
        "target_delta_locator_mm",
    ]

    record_check(
        "training residual targets finite",
        dataframe_is_finite(
            train_df,
            target_columns,
        ),
        target_columns,
    )

    record_check(
        "development residual targets finite",
        dataframe_is_finite(
            development_df,
            target_columns,
        ),
        target_columns,
    )

    # --------------------------------------------------------
    # BEST-REFERENCE SAFETY
    # --------------------------------------------------------

    print(
        "\n[5] Best-reference safety"
    )

    train_reference_worse = (
        train_df[
            "best_reference_true_quality"
        ]
        >
        train_df[
            "composite_q_true_quality"
        ]
        +
        1e-9
    )

    development_reference_worse = (
        development_df[
            "best_reference_true_quality"
        ]
        >
        development_df[
            "composite_q_true_quality"
        ]
        +
        1e-9
    )

    record_check(
        "training reference never worse than Composite-Q",
        not train_reference_worse.any(),
        (
            f"violations = "
            f"{int(train_reference_worse.sum())}"
        ),
    )

    record_check(
        "development reference never worse than Composite-Q",
        not development_reference_worse.any(),
        (
            f"violations = "
            f"{int(development_reference_worse.sum())}"
        ),
    )

    # --------------------------------------------------------
    # MODEL FILES
    # --------------------------------------------------------

    print(
        "\n[6] Trained model files"
    )

    loaded_models = 0

    for model_file in MODEL_FILES:
        try:
            _ = joblib.load(
                model_file
            )

            loaded_models += 1

            record_check(
                f"model load: {model_file.name}",
                True,
                "loaded successfully",
            )

        except Exception as exc:
            record_check(
                f"model load: {model_file.name}",
                False,
                exc,
            )

    record_check(
        "six residual regressors available",
        loaded_models
        ==
        6,
        f"{loaded_models} / 6",
    )

    record_check(
        "model metrics contain six rows",
        len(
            model_metrics_df
        )
        ==
        6,
        f"{len(model_metrics_df)} / 6",
    )

    # --------------------------------------------------------
    # FINAL ASSEMBLY RESULTS
    # --------------------------------------------------------

    print(
        "\n[7] Final assembly results"
    )

    expected_final_assembly_rows = (
        N_FINAL_ASSEMBLIES
        *
        len(
            EXPECTED_METHODS
        )
    )

    record_check(
        "final assembly row count",
        len(
            final_assembly_df
        )
        ==
        expected_final_assembly_rows,
        (
            f"{len(final_assembly_df)} / "
            f"{expected_final_assembly_rows}"
        ),
    )

    final_methods = set(
        final_assembly_df[
            "method"
        ].unique()
    )

    record_check(
        "final method set",
        final_methods
        ==
        EXPECTED_METHODS,
        sorted(
            final_methods
        ),
    )

    final_method_counts = (
        final_assembly_df[
            "method"
        ]
        .value_counts()
        .to_dict()
    )

    method_counts_correct = all(
        final_method_counts.get(
            method,
            0,
        )
        ==
        N_FINAL_ASSEMBLIES
        for method in EXPECTED_METHODS
    )

    record_check(
        "300 final assemblies per method",
        method_counts_correct,
        final_method_counts,
    )

    assembly_method_unique = (
        not final_assembly_df.duplicated(
            subset=[
                "assembly_index",
                "method",
            ]
        ).any()
    )

    record_check(
        "final assembly/method keys unique",
        assembly_method_unique,
        "assembly_index + method",
    )

    final_quality_columns = [
        "final_quality",
        "final_mean_gap",
        "final_max_gap",
        "final_parallelism",
        "final_rms",
    ]

    record_check(
        "final assembly metrics finite",
        dataframe_is_finite(
            final_assembly_df,
            final_quality_columns,
        ),
        final_quality_columns,
    )

    record_check(
        "final assembly results contain no NaN",
        not final_assembly_df.isna().any().any(),
        (
            f"NaN cells = "
            f"{int(final_assembly_df.isna().sum().sum())}"
        ),
    )

    # --------------------------------------------------------
    # FINAL COMPONENT RESULTS
    # --------------------------------------------------------

    print(
        "\n[8] Final component results"
    )

    expected_final_component_rows = (
        N_FINAL_ASSEMBLIES
        *
        SEQUENCE_LENGTH
        *
        len(
            EXPECTED_METHODS
        )
    )

    record_check(
        "final component row count",
        len(
            final_component_df
        )
        ==
        expected_final_component_rows,
        (
            f"{len(final_component_df)} / "
            f"{expected_final_component_rows}"
        ),
    )

    final_component_key_unique = (
        not final_component_df.duplicated(
            subset=[
                "assembly_index",
                "component_index",
                "method",
            ]
        ).any()
    )

    record_check(
        "final component keys unique",
        final_component_key_unique,
        (
            "assembly_index + "
            "component_index + method"
        ),
    )

    record_check(
        "final component results contain no NaN",
        not final_component_df.isna().any().any(),
        (
            f"NaN cells = "
            f"{int(final_component_df.isna().sum().sum())}"
        ),
    )

    # --------------------------------------------------------
    # CORRECTION BOUNDS
    # --------------------------------------------------------

    print(
        "\n[9] Correction-bound compliance"
    )

    z_violation = (
        final_component_df[
            "final_z_adj_mm"
        ].abs()
        >
        Z_ADJ_LIMIT_MM
        +
        1e-9
    )

    theta_violation = (
        final_component_df[
            "final_theta_adj_deg"
        ].abs()
        >
        THETA_ADJ_LIMIT_DEG
        +
        1e-9
    )

    locator_violation = (
        final_component_df[
            "final_locator_offset_mm"
        ].abs()
        >
        LOCATOR_OFFSET_LIMIT_MM
        +
        1e-9
    )

    record_check(
        "z correction within bounds",
        not z_violation.any(),
        (
            f"violations = "
            f"{int(z_violation.sum())}"
        ),
    )

    record_check(
        "theta correction within bounds",
        not theta_violation.any(),
        (
            f"violations = "
            f"{int(theta_violation.sum())}"
        ),
    )

    record_check(
        "locator correction within bounds",
        not locator_violation.any(),
        (
            f"violations = "
            f"{int(locator_violation.sum())}"
        ),
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print(
        "\n[10] Final statistical analysis"
    )

    expected_hypotheses = {
        "H1",
        "H2",
        "H3",
    }

    observed_hypotheses = set(
        statistics_df[
            "hypothesis"
        ].astype(
            str
        )
    )

    record_check(
        "statistical hypothesis set",
        observed_hypotheses
        ==
        expected_hypotheses,
        sorted(
            observed_hypotheses
        ),
    )

    record_check(
        "three predefined statistical comparisons",
        len(
            statistics_df
        )
        ==
        3,
        f"{len(statistics_df)} / 3",
    )

    statistical_numeric_columns = [
        "mean_difference_a_minus_b",
        "bootstrap_95_ci_low",
        "bootstrap_95_ci_high",
        "raw_p_value",
        "holm_adjusted_p_value",
        "paired_standardized_effect",
    ]

    record_check(
        "statistical results finite",
        dataframe_is_finite(
            statistics_df,
            statistical_numeric_columns,
        ),
        statistical_numeric_columns,
    )

    adjusted_p_valid = bool(
        (
            (
                statistics_df[
                    "holm_adjusted_p_value"
                ]
                >=
                0.0
            )
            &
            (
                statistics_df[
                    "holm_adjusted_p_value"
                ]
                <=
                1.0
            )
        ).all()
    )

    record_check(
        "Holm-adjusted p-values valid",
        adjusted_p_valid,
        "all values within [0, 1]",
    )

    # --------------------------------------------------------
    # FINAL RESULT RECONCILIATION
    # --------------------------------------------------------

    print(
        "\n[11] Final result reconciliation"
    )

    pivot = (
        final_assembly_df.pivot(
            index=
                "assembly_index",

            columns=
                "method",

            values=
                "final_quality",
        )
    )

    cq_mean = float(
        pivot[
            "COMPOSITE_Q"
        ].mean()
    )

    rf_mean = float(
        pivot[
            "COMPOSITE_Q_PLUS_RF"
        ].mean()
    )

    gb_mean = float(
        pivot[
            "COMPOSITE_Q_PLUS_GB"
        ].mean()
    )

    record_check(
        "Composite-Q is lowest final mean quality",
        (
            cq_mean
            <
            rf_mean
            and
            cq_mean
            <
            gb_mean
        ),
        (
            f"CQ={cq_mean:.12f}, "
            f"RF={rf_mean:.12f}, "
            f"GB={gb_mean:.12f}"
        ),
    )

    h1 = (
        statistics_df[
            statistics_df[
                "hypothesis"
            ]
            ==
            "H1"
        ].iloc[
            0
        ]
    )

    h2 = (
        statistics_df[
            statistics_df[
                "hypothesis"
            ]
            ==
            "H2"
        ].iloc[
            0
        ]
    )

    h3 = (
        statistics_df[
            statistics_df[
                "hypothesis"
            ]
            ==
            "H3"
        ].iloc[
            0
        ]
    )

    record_check(
        "H1 direction matches final means",
        float(
            h1[
                "mean_difference_a_minus_b"
            ]
        )
        >
        0.0,
        (
            "RF - Composite-Q > 0 "
            "(RF worse)"
        ),
    )

    record_check(
        "H2 direction matches final means",
        float(
            h2[
                "mean_difference_a_minus_b"
            ]
        )
        >
        0.0,
        (
            "GB - Composite-Q > 0 "
            "(GB worse)"
        ),
    )

    record_check(
        "H3 not significant after Holm",
        not bool(
            h3[
                "significant_after_holm"
            ]
        ),
        (
            f"Holm p = "
            f"{float(h3['holm_adjusted_p_value']):.6g}"
        ),
    )

    # --------------------------------------------------------
    # FINAL GATE
    # --------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 INTEGRITY GATE"
    )

    print(
        "============================================================"
    )

    check_df = pd.DataFrame(
        checks
    )

    failed_checks = (
        check_df[
            check_df[
                "status"
            ]
            ==
            "FAIL"
        ]
    )

    print(
        f"\nTotal checks : "
        f"{len(check_df)}"
    )

    print(
        f"Passed       : "
        f"{int((check_df['status'] == 'PASS').sum())}"
    )

    print(
        f"Failed       : "
        f"{len(failed_checks)}"
    )

    if len(
        failed_checks
    ) == 0:

        print(
            "\nFINAL STATUS: PASS"
        )

        print(
            "Phase 8 outputs are internally consistent "
            "and ready for final documentation/freeze."
        )

    else:

        print(
            "\nFINAL STATUS: FAIL"
        )

        print(
            "Do not freeze Phase 8 until all failed "
            "integrity checks are resolved."
        )

        print(
            "\nFailed checks:"
        )

        print(
            failed_checks.to_string(
                index=False
            )
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 FINAL INTEGRITY VALIDATION COMPLETED"
    )

    print(
        "============================================================"
    )

    if len(
        failed_checks
    ) > 0:
        raise RuntimeError(
            "Phase 8 integrity validation failed."
        )


if __name__ == "__main__":
    main()