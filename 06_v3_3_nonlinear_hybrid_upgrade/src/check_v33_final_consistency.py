"""
V3.3 final scientific consistency checker.

Purpose
-------
Verify that the completed V3.3 nonlinear hybrid upgrade is internally
consistent before documentation cleanup and freezing.

This script does NOT:
- retrain any model,
- tune any controller,
- change architecture,
- generate new scientific results.

It checks the already-generated evidence package.

Major evidence checked
----------------------
1. Architecture lock exists and is LSQ_PLUS_ML_ALL.
2. Full residual-learning dataset sizes are correct.
3. Training/development assembly separation is preserved.
4. Full residual model uses 26 observable features and 3 outputs.
5. Independent validation contains 300 assemblies.
6. Locked hybrid beats LSQ on mean final quality.
7. Independent paired bootstrap CI supports hybrid advantage.
8. Hybrid win rate and relative improvement match final statistics.
9. Stage-wise results contain all 10 stages.
10. Robustness regime contains 9 configurations.
11. Hybrid remains better than LSQ in all tested robustness regimes.
12. Sequence scaling contains K=5 and K=10.
13. Hybrid remains better than LSQ at both sequence lengths.
14. Direct reference medium/high-budget stability is below 1%.
15. Required output files exist.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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


FREEZE_DIR = (
    V33_ROOT
    / "freeze"
)


ARCHITECTURE_LOCK_FILE = (
    FREEZE_DIR
    / "v33_hybrid_architecture_lock.txt"
)


FULL_DATA_DIR = (
    V33_ROOT
    / "data"
    / "residual_learning_full"
)


TRAIN_FILE = (
    FULL_DATA_DIR
    / "v33_residual_learning_full_train.csv"
)


DEVELOPMENT_FILE = (
    FULL_DATA_DIR
    / "v33_residual_learning_full_development.csv"
)


FEATURE_FILE = (
    FULL_DATA_DIR
    / "v33_residual_learning_full_observable_features.txt"
)


MODEL_FILE = (
    V33_ROOT
    / "models"
    / "residual_learning_full"
    / "v33_residual_rf_full.joblib"
)


FINAL_RESULTS_DIR = (
    V33_ROOT
    / "results"
    / "final_independent_validation_v33"
)


FINAL_ASSEMBLY_FILE = (
    FINAL_RESULTS_DIR
    / "v33_final_independent_assembly_results.csv"
)


FINAL_COMPONENT_FILE = (
    FINAL_RESULTS_DIR
    / "v33_final_independent_component_results.csv"
)


FINAL_STATISTICS_FILE = (
    FINAL_RESULTS_DIR
    / "v33_final_independent_statistics.csv"
)


FINAL_STAGE_FILE = (
    FINAL_RESULTS_DIR
    / "v33_final_independent_stage_statistics.csv"
)


FINAL_UTILIZATION_FILE = (
    FINAL_RESULTS_DIR
    / "v33_final_independent_utilization_statistics.csv"
)


ROBUSTNESS_DIR = (
    V33_ROOT
    / "results"
    / "regime_robustness"
)


ROBUSTNESS_FILE = (
    ROBUSTNESS_DIR
    / "v33_regime_robustness_summary.csv"
)


SCALING_DIR = (
    V33_ROOT
    / "results"
    / "sequence_length_scaling"
)


SCALING_FILE = (
    SCALING_DIR
    / "v33_sequence_length_scaling_summary.csv"
)


SCALING_STAGE_FILE = (
    SCALING_DIR
    / "v33_sequence_length_scaling_stage_results.csv"
)


REFERENCE_DIR = (
    V33_ROOT
    / "results"
    / "direct_reference_stability"
)


REFERENCE_SUMMARY_FILE = (
    REFERENCE_DIR
    / "v33_direct_reference_stability_summary.csv"
)


REFERENCE_COMPARISON_FILE = (
    REFERENCE_DIR
    / "v33_direct_reference_budget_comparisons.csv"
)


CHECK_OUTPUT_FILE = (
    V33_ROOT
    / "results"
    / "v33_final_consistency_checks.csv"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_TRAIN_ROWS = 3000

EXPECTED_DEVELOPMENT_ROWS = 600

EXPECTED_FEATURES = 26

EXPECTED_MODEL_OUTPUTS = 3

EXPECTED_FINAL_ASSEMBLIES = 300

EXPECTED_SEQUENCE_LENGTH = 10

EXPECTED_ROBUSTNESS_CONFIGURATIONS = 9

EXPECTED_SCALING_LENGTHS = {
    5,
    10,
}

LOCKED_ARCHITECTURE = (
    "LSQ_PLUS_ML_ALL"
)


TOLERANCE = 1e-6


# ============================================================
# CHECK STORAGE
# ============================================================

checks = []


def add_check(
    name,
    passed,
    observed="",
    expected="",
):
    """
    Store one consistency check.
    """

    checks.append(
        {
            "check":
                name,

            "status":
                "PASS"
                if bool(
                    passed
                )
                else
                "FAIL",

            "observed":
                str(
                    observed
                ),

            "expected":
                str(
                    expected
                ),
        }
    )


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
    ARCHITECTURE_LOCK_FILE,
    TRAIN_FILE,
    DEVELOPMENT_FILE,
    FEATURE_FILE,
    MODEL_FILE,
    FINAL_ASSEMBLY_FILE,
    FINAL_COMPONENT_FILE,
    FINAL_STATISTICS_FILE,
    FINAL_STAGE_FILE,
    FINAL_UTILIZATION_FILE,
    ROBUSTNESS_FILE,
    SCALING_FILE,
    SCALING_STAGE_FILE,
    REFERENCE_SUMMARY_FILE,
    REFERENCE_COMPARISON_FILE,
]


for path in required_files:

    add_check(
        name=(
            f"required_file_exists:"
            f"{path.name}"
        ),
        passed=path.exists(),
        observed=path.exists(),
        expected=True,
    )


# ============================================================
# STOP IF REQUIRED FILES ARE MISSING
# ============================================================

missing_required_files = [
    path
    for path in required_files
    if not path.exists()
]


if missing_required_files:

    check_df = pd.DataFrame(
        checks
    )

    CHECK_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_df.to_csv(
        CHECK_OUTPUT_FILE,
        index=False,
    )

    print(
        "\nMissing required files:"
    )

    for path in missing_required_files:

        print(
            path
        )

    raise FileNotFoundError(
        "\nV3.3 consistency checker stopped because "
        "required evidence files are missing."
    )


# ============================================================
# ARCHITECTURE LOCK
# ============================================================

lock_text = (
    ARCHITECTURE_LOCK_FILE
    .read_text(
        encoding="utf-8-sig"
    )
)


add_check(
    name="architecture_lock_is_all_residuals",
    passed=(
        LOCKED_ARCHITECTURE
        in
        lock_text
    ),
    observed=(
        LOCKED_ARCHITECTURE
        if LOCKED_ARCHITECTURE in lock_text
        else "not found"
    ),
    expected=LOCKED_ARCHITECTURE,
)


add_check(
    name="architecture_locked_before_final_validation_statement",
    passed=(
        "Do not change residual dimensions"
        in
        lock_text
    ),
    observed=(
        "statement found"
        if "Do not change residual dimensions" in lock_text
        else "statement missing"
    ),
    expected="statement found",
)


# ============================================================
# FULL DATASET
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)


development_df = pd.read_csv(
    DEVELOPMENT_FILE
)


add_check(
    name="full_training_rows",
    passed=(
        len(
            train_df
        )
        ==
        EXPECTED_TRAIN_ROWS
    ),
    observed=len(
        train_df
    ),
    expected=EXPECTED_TRAIN_ROWS,
)


add_check(
    name="full_development_rows",
    passed=(
        len(
            development_df
        )
        ==
        EXPECTED_DEVELOPMENT_ROWS
    ),
    observed=len(
        development_df
    ),
    expected=EXPECTED_DEVELOPMENT_ROWS,
)


train_ids = set(
    train_df[
        "assembly_id"
    ]
    .astype(
        str
    )
)


development_ids = set(
    development_df[
        "assembly_id"
    ]
    .astype(
        str
    )
)


dataset_overlap = (
    train_ids
    &
    development_ids
)


add_check(
    name="train_development_assembly_overlap",
    passed=(
        len(
            dataset_overlap
        )
        ==
        0
    ),
    observed=len(
        dataset_overlap
    ),
    expected=0,
)


# ============================================================
# FEATURE LIST
# ============================================================

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8",
) as file:

    feature_columns = [
        line.strip()
        for line in file
        if line.strip()
    ]


add_check(
    name="observable_feature_count",
    passed=(
        len(
            feature_columns
        )
        ==
        EXPECTED_FEATURES
    ),
    observed=len(
        feature_columns
    ),
    expected=EXPECTED_FEATURES,
)


privileged_feature_names = {
    "offset_mm",
    "tilt_deg",
    "bend_mm",
    "waviness_mm",
    "twist_mm",
    "local_bump_mm",
    "variation_multiplier",
    "batch_offset_bias_mm",
    "batch_angular_bias_deg",
    "fixture_drift_mm",
}


privileged_present = sorted(
    privileged_feature_names
    &
    set(
        feature_columns
    )
)


add_check(
    name="no_known_generator_privileged_features",
    passed=(
        len(
            privileged_present
        )
        ==
        0
    ),
    observed=privileged_present,
    expected=[],
)


# ============================================================
# MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)


model_feature_count = getattr(
    model,
    "n_features_in_",
    None,
)


model_output_count = getattr(
    model,
    "n_outputs_",
    None,
)


add_check(
    name="model_feature_count",
    passed=(
        model_feature_count
        ==
        EXPECTED_FEATURES
    ),
    observed=model_feature_count,
    expected=EXPECTED_FEATURES,
)


add_check(
    name="model_output_count",
    passed=(
        model_output_count
        ==
        EXPECTED_MODEL_OUTPUTS
    ),
    observed=model_output_count,
    expected=EXPECTED_MODEL_OUTPUTS,
)


# ============================================================
# FINAL INDEPENDENT VALIDATION
# ============================================================

final_assembly_df = pd.read_csv(
    FINAL_ASSEMBLY_FILE
)


final_component_df = pd.read_csv(
    FINAL_COMPONENT_FILE
)


final_statistics_df = pd.read_csv(
    FINAL_STATISTICS_FILE
)


final_stage_df = pd.read_csv(
    FINAL_STAGE_FILE
)


final_n_assemblies = int(
    final_assembly_df[
        "assembly_index"
    ]
    .nunique()
)


add_check(
    name="final_independent_assembly_count",
    passed=(
        final_n_assemblies
        ==
        EXPECTED_FINAL_ASSEMBLIES
    ),
    observed=final_n_assemblies,
    expected=EXPECTED_FINAL_ASSEMBLIES,
)


final_max_stage = int(
    final_component_df[
        "component_index"
    ]
    .max()
)


add_check(
    name="final_sequence_length",
    passed=(
        final_max_stage
        ==
        EXPECTED_SEQUENCE_LENGTH
    ),
    observed=final_max_stage,
    expected=EXPECTED_SEQUENCE_LENGTH,
)


# ============================================================
# FINAL METHOD MEANS
# ============================================================

final_method_means = (
    final_assembly_df
    .groupby(
        "method"
    )[
        "final_quality"
    ]
    .mean()
)


required_final_methods = [
    "LSQ",
    "LSQ_PLUS_ML_ALL",
    "DIRECT_NONLINEAR_REFERENCE",
]


for method in required_final_methods:

    add_check(
        name=(
            f"final_method_present:"
            f"{method}"
        ),
        passed=(
            method
            in
            final_method_means.index
        ),
        observed=(
            method
            in
            final_method_means.index
        ),
        expected=True,
    )


mean_lsq = float(
    final_method_means[
        "LSQ"
    ]
)


mean_hybrid = float(
    final_method_means[
        "LSQ_PLUS_ML_ALL"
    ]
)


add_check(
    name="final_locked_hybrid_mean_better_than_lsq",
    passed=(
        mean_hybrid
        <
        mean_lsq
    ),
    observed=(
        f"LSQ={mean_lsq:.6f}, "
        f"Hybrid={mean_hybrid:.6f}"
    ),
    expected="Hybrid < LSQ",
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
    mean_lsq
)


add_check(
    name="final_relative_improvement_positive",
    passed=(
        relative_improvement
        >
        0.0
    ),
    observed=(
        f"{relative_improvement:.4f}%"
    ),
    expected="> 0%",
)


# ============================================================
# STATISTICS LOOKUP
# ============================================================

statistics_lookup = {
    str(
        row[
            "metric"
        ]
    ):
        float(
            row[
                "value"
            ]
        )

    for _, row in (
        final_statistics_df
        .iterrows()
    )
}


required_stat_metrics = [
    "mean_lsq_quality",
    "mean_locked_hybrid_quality",
    "paired_bootstrap_95_ci_lower",
    "paired_bootstrap_95_ci_upper",
    "wilcoxon_p_value",
    "hybrid_beats_lsq_percent",
    "relative_hybrid_improvement_percent",
]


for metric in required_stat_metrics:

    add_check(
        name=(
            f"final_stat_metric_present:"
            f"{metric}"
        ),
        passed=(
            metric
            in
            statistics_lookup
        ),
        observed=(
            metric
            in
            statistics_lookup
        ),
        expected=True,
    )


if all(
    metric
    in
    statistics_lookup
    for metric in required_stat_metrics
):

    ci_lower = (
        statistics_lookup[
            "paired_bootstrap_95_ci_lower"
        ]
    )


    ci_upper = (
        statistics_lookup[
            "paired_bootstrap_95_ci_upper"
        ]
    )


    wilcoxon_p = (
        statistics_lookup[
            "wilcoxon_p_value"
        ]
    )


    final_win_rate = (
        statistics_lookup[
            "hybrid_beats_lsq_percent"
        ]
    )


    reported_relative_improvement = (
        statistics_lookup[
            "relative_hybrid_improvement_percent"
        ]
    )


    add_check(
        name="final_bootstrap_ci_entirely_below_zero",
        passed=(
            ci_upper
            <
            0.0
        ),
        observed=(
            f"[{ci_lower:.6f}, "
            f"{ci_upper:.6f}]"
        ),
        expected="upper < 0",
    )


    add_check(
        name="final_wilcoxon_significant",
        passed=(
            wilcoxon_p
            <
            0.05
        ),
        observed=wilcoxon_p,
        expected="< 0.05",
    )


    add_check(
        name="final_hybrid_win_rate_above_50_percent",
        passed=(
            final_win_rate
            >
            50.0
        ),
        observed=(
            f"{final_win_rate:.2f}%"
        ),
        expected="> 50%",
    )


    add_check(
        name="reported_relative_improvement_matches_recomputed",
        passed=(
            abs(
                reported_relative_improvement
                -
                relative_improvement
            )
            <=
            0.01
        ),
        observed=(
            f"reported="
            f"{reported_relative_improvement:.4f}%, "
            f"recomputed="
            f"{relative_improvement:.4f}%"
        ),
        expected="difference <= 0.01 percentage point",
    )


    add_check(
        name="reported_lsq_mean_matches_assembly_results",
        passed=(
            abs(
                statistics_lookup[
                    "mean_lsq_quality"
                ]
                -
                mean_lsq
            )
            <=
            TOLERANCE
        ),
        observed=(
            statistics_lookup[
                "mean_lsq_quality"
            ]
        ),
        expected=mean_lsq,
    )


    add_check(
        name="reported_hybrid_mean_matches_assembly_results",
        passed=(
            abs(
                statistics_lookup[
                    "mean_locked_hybrid_quality"
                ]
                -
                mean_hybrid
            )
            <=
            TOLERANCE
        ),
        observed=(
            statistics_lookup[
                "mean_locked_hybrid_quality"
            ]
        ),
        expected=mean_hybrid,
    )


# ============================================================
# STAGE-WISE FINAL EVIDENCE
# ============================================================

stage_indices = set(
    final_stage_df[
        "component_index"
    ]
    .astype(
        int
    )
)


expected_stage_indices = set(
    range(
        1,
        EXPECTED_SEQUENCE_LENGTH
        +
        1
    )
)


add_check(
    name="final_stage_indices_complete",
    passed=(
        stage_indices
        ==
        expected_stage_indices
    ),
    observed=sorted(
        stage_indices
    ),
    expected=sorted(
        expected_stage_indices
    ),
)


all_stage_means_better = bool(
    (
        final_stage_df[
            "mean_hybrid_minus_lsq"
        ]
        <
        0.0
    )
    .all()
)


add_check(
    name="hybrid_mean_better_at_all_10_stages",
    passed=all_stage_means_better,
    observed=all_stage_means_better,
    expected=True,
)


# ============================================================
# ROBUSTNESS REGIME
# ============================================================

robustness_df = pd.read_csv(
    ROBUSTNESS_FILE
)


add_check(
    name="robustness_configuration_count",
    passed=(
        len(
            robustness_df
        )
        ==
        EXPECTED_ROBUSTNESS_CONFIGURATIONS
    ),
    observed=len(
        robustness_df
    ),
    expected=EXPECTED_ROBUSTNESS_CONFIGURATIONS,
)


robustness_all_positive = bool(
    (
        robustness_df[
            "relative_hybrid_improvement_percent"
        ]
        >
        0.0
    )
    .all()
)


add_check(
    name="hybrid_positive_in_all_tested_robustness_regimes",
    passed=robustness_all_positive,
    observed=robustness_all_positive,
    expected=True,
)


robustness_all_mean_better = bool(
    (
        robustness_df[
            "mean_hybrid_minus_lsq"
        ]
        <
        0.0
    )
    .all()
)


add_check(
    name="hybrid_mean_better_than_lsq_in_all_regimes",
    passed=robustness_all_mean_better,
    observed=robustness_all_mean_better,
    expected=True,
)


# ============================================================
# SEQUENCE-LENGTH SCALING
# ============================================================

scaling_df = pd.read_csv(
    SCALING_FILE
)


observed_sequence_lengths = set(
    scaling_df[
        "sequence_length"
    ]
    .astype(
        int
    )
)


add_check(
    name="scaling_sequence_lengths",
    passed=(
        observed_sequence_lengths
        ==
        EXPECTED_SCALING_LENGTHS
    ),
    observed=sorted(
        observed_sequence_lengths
    ),
    expected=sorted(
        EXPECTED_SCALING_LENGTHS
    ),
)


scaling_all_positive = bool(
    (
        scaling_df[
            "relative_hybrid_improvement_percent"
        ]
        >
        0.0
    )
    .all()
)


add_check(
    name="hybrid_positive_at_k5_and_k10",
    passed=scaling_all_positive,
    observed=scaling_all_positive,
    expected=True,
)


# ============================================================
# PREFIX CONSISTENCY
# ============================================================

scaling_stage_df = pd.read_csv(
    SCALING_STAGE_FILE
)


stage5_k5 = (
    scaling_stage_df[
        (
            scaling_stage_df[
                "sequence_length"
            ]
            ==
            5
        )
        &
        (
            scaling_stage_df[
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
    scaling_stage_df[
        (
            scaling_stage_df[
                "sequence_length"
            ]
            ==
            10
        )
        &
        (
            scaling_stage_df[
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


add_check(
    name="k5_k10_lsq_prefix_consistency",
    passed=(
        prefix_lsq_difference
        <=
        1e-12
    ),
    observed=prefix_lsq_difference,
    expected="<= 1e-12",
)


add_check(
    name="k5_k10_hybrid_prefix_consistency",
    passed=(
        prefix_hybrid_difference
        <=
        1e-12
    ),
    observed=prefix_hybrid_difference,
    expected="<= 1e-12",
)


# ============================================================
# DIRECT REFERENCE STABILITY
# ============================================================

reference_df = pd.read_csv(
    REFERENCE_SUMMARY_FILE
)


reference_comparison_df = pd.read_csv(
    REFERENCE_COMPARISON_FILE
)


required_budgets = {
    "LOW_8x5",
    "MEDIUM_18x7",
    "HIGH_30x10",
}


observed_budgets = set(
    reference_df[
        "budget"
    ]
    .astype(
        str
    )
)


add_check(
    name="direct_reference_budgets_present",
    passed=(
        observed_budgets
        ==
        required_budgets
    ),
    observed=sorted(
        observed_budgets
    ),
    expected=sorted(
        required_budgets
    ),
)


medium_quality = float(
    reference_df[
        reference_df[
            "budget"
        ]
        ==
        "MEDIUM_18x7"
    ][
        "mean_final_quality"
    ]
    .iloc[0]
)


high_quality = float(
    reference_df[
        reference_df[
            "budget"
        ]
        ==
        "HIGH_30x10"
    ][
        "mean_final_quality"
    ]
    .iloc[0]
)


high_vs_medium_relative_change = float(
    100.0
    *
    abs(
        medium_quality
        -
        high_quality
    )
    /
    medium_quality
)


add_check(
    name="direct_reference_medium_high_stability_below_1_percent",
    passed=(
        high_vs_medium_relative_change
        <
        1.0
    ),
    observed=(
        f"{high_vs_medium_relative_change:.4f}%"
    ),
    expected="< 1%",
)


# ============================================================
# SAVE + PRINT
# ============================================================

check_df = pd.DataFrame(
    checks
)


CHECK_OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


check_df.to_csv(
    CHECK_OUTPUT_FILE,
    index=False,
)


passed_count = int(
    (
        check_df[
            "status"
        ]
        ==
        "PASS"
    )
    .sum()
)


failed_count = int(
    (
        check_df[
            "status"
        ]
        ==
        "FAIL"
    )
    .sum()
)


total_count = len(
    check_df
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 FINAL CONSISTENCY CHECK"
)

print(
    "============================================================"
)


for _, row in check_df.iterrows():

    print(
        f"\n[{row['status']}] "
        f"{row['check']}"
    )


    if (
        str(
            row[
                "observed"
            ]
        )
        !=
        ""
    ):

        print(
            f"  observed : "
            f"{row['observed']}"
        )


    if (
        str(
            row[
                "expected"
            ]
        )
        !=
        ""
    ):

        print(
            f"  expected : "
            f"{row['expected']}"
        )


print(
    "\n"
    "============================================================"
)

print(
    f"Total checks : {total_count}"
)

print(
    f"Passed       : {passed_count}"
)

print(
    f"Failed       : {failed_count}"
)


if failed_count == 0:

    print(
        "\n"
        "V3.3 FINAL CONSISTENCY: PASS"
    )


    print(
        "\n"
        "The V3.3 nonlinear hybrid upgrade evidence package "
        "is internally consistent."
    )


else:

    print(
        "\n"
        "V3.3 FINAL CONSISTENCY: FAIL"
    )


    print(
        "\n"
        "Do not freeze V3.3 until every failed check has "
        "been investigated."
    )


print(
    "\nSaved consistency table:"
)

print(
    CHECK_OUTPUT_FILE
)


print(
    "\n"
    "============================================================"
)