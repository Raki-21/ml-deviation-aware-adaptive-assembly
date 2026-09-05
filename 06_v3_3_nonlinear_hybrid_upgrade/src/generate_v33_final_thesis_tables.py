"""
Generate final thesis tables for V3.3 from validated result files only.

This script does NOT:
- rerun experiments,
- retrain models,
- tune controllers,
- alter frozen evidence.

Outputs are written to:
06_v3_3_nonlinear_hybrid_upgrade/results/final_thesis_tables
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
V33_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = V33_ROOT / "results"

FINAL_DIR = RESULTS_ROOT / "final_independent_validation_v33"
ROBUSTNESS_DIR = RESULTS_ROOT / "regime_robustness"
SCALING_DIR = RESULTS_ROOT / "sequence_length_scaling"
REFERENCE_DIR = RESULTS_ROOT / "direct_reference_stability"
RESIDUAL_DIR = RESULTS_ROOT / "residual_learning_full"

OUTPUT_DIR = RESULTS_ROOT / "final_thesis_tables"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

FINAL_SUMMARY_FILE = (
    FINAL_DIR
    / "v33_final_independent_summary.csv"
)

FINAL_STATISTICS_FILE = (
    FINAL_DIR
    / "v33_final_independent_statistics.csv"
)

FINAL_STAGE_FILE = (
    FINAL_DIR
    / "v33_final_independent_stage_statistics.csv"
)

FINAL_UTILIZATION_FILE = (
    FINAL_DIR
    / "v33_final_independent_utilization_statistics.csv"
)

ROBUSTNESS_FILE = (
    ROBUSTNESS_DIR
    / "v33_regime_robustness_summary.csv"
)

SCALING_FILE = (
    SCALING_DIR
    / "v33_sequence_length_scaling_summary.csv"
)

REFERENCE_FILE = (
    REFERENCE_DIR
    / "v33_direct_reference_stability_summary.csv"
)

RESIDUAL_SUMMARY_FILE = (
    RESIDUAL_DIR
    / "v33_residual_rf_full_summary.csv"
)


required_files = [
    FINAL_SUMMARY_FILE,
    FINAL_STATISTICS_FILE,
    FINAL_STAGE_FILE,
    FINAL_UTILIZATION_FILE,
    ROBUSTNESS_FILE,
    SCALING_FILE,
    REFERENCE_FILE,
    RESIDUAL_SUMMARY_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required result file missing:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

final_summary_df = pd.read_csv(
    FINAL_SUMMARY_FILE
)

final_statistics_df = pd.read_csv(
    FINAL_STATISTICS_FILE
)

final_stage_df = pd.read_csv(
    FINAL_STAGE_FILE
)

final_utilization_df = pd.read_csv(
    FINAL_UTILIZATION_FILE
)

robustness_df = pd.read_csv(
    ROBUSTNESS_FILE
)

scaling_df = pd.read_csv(
    SCALING_FILE
)

reference_df = pd.read_csv(
    REFERENCE_FILE
)

residual_summary_df = pd.read_csv(
    RESIDUAL_SUMMARY_FILE
)


# ============================================================
# TABLE 1
# FINAL INDEPENDENT METHOD COMPARISON
# ============================================================

preferred_methods = [
    "ZERO",
    "LSQ",
    "LSQ_PLUS_ML_ALL",
    "DIRECT_NONLINEAR_REFERENCE",
]


table1 = (
    final_summary_df[
        final_summary_df[
            "method"
        ]
        .isin(
            preferred_methods
        )
    ]
    .copy()
)


method_label_map = {
    "ZERO":
        "Zero correction",

    "LSQ":
        "Deterministic LSQ",

    "LSQ_PLUS_ML_ALL":
        "Locked hybrid",

    "DIRECT_NONLINEAR_REFERENCE":
        "Direct nonlinear reference",
}


table1[
    "Method"
] = table1[
    "method"
].map(
    method_label_map
)


column_candidates = {
    "mean_final_quality":
        "Mean final quality",

    "median_final_quality":
        "Median final quality",

    "p95_final_quality":
        "P95 final quality",

    "beats_lsq_percent":
        "Beats LSQ (%)",

    "mean_difference_vs_lsq":
        "Mean difference vs LSQ",

    "mean_utilization":
        "Mean utilization",
}


existing_columns = [
    column
    for column in column_candidates
    if column in table1.columns
]


table1 = table1[
    [
        "Method"
    ]
    +
    existing_columns
]


table1 = table1.rename(
    columns={
        column:
            column_candidates[
                column
            ]
        for column in existing_columns
    }
)


table1.to_csv(
    OUTPUT_DIR
    / "table_v33_final_independent_method_comparison.csv",
    index=False,
)


# ============================================================
# TABLE 2
# FINAL PAIRED STATISTICAL EVIDENCE
# ============================================================

stats_lookup = {
    str(
        row[
            "metric"
        ]
    ):
        row[
            "value"
        ]

    for _, row in (
        final_statistics_df
        .iterrows()
    )
}


statistics_rows = [
    {
        "Metric":
            "Independent assemblies",

        "Value":
            stats_lookup.get(
                "n_independent_assemblies",
                "",
            ),
    },

    {
        "Metric":
            "Sequence length",

        "Value":
            stats_lookup.get(
                "sequence_length",
                "",
            ),
    },

    {
        "Metric":
            "LSQ mean quality",

        "Value":
            stats_lookup.get(
                "mean_lsq_quality",
                "",
            ),
    },

    {
        "Metric":
            "Locked hybrid mean quality",

        "Value":
            stats_lookup.get(
                "mean_locked_hybrid_quality",
                "",
            ),
    },

    {
        "Metric":
            "Mean hybrid - LSQ",

        "Value":
            stats_lookup.get(
                "mean_hybrid_minus_lsq",
                "",
            ),
    },

    {
        "Metric":
            "Median hybrid - LSQ",

        "Value":
            stats_lookup.get(
                "median_hybrid_minus_lsq",
                "",
            ),
    },

    {
        "Metric":
            "Relative hybrid improvement (%)",

        "Value":
            stats_lookup.get(
                "relative_hybrid_improvement_percent",
                "",
            ),
    },

    {
        "Metric":
            "Hybrid beats LSQ (%)",

        "Value":
            stats_lookup.get(
                "hybrid_beats_lsq_percent",
                "",
            ),
    },

    {
        "Metric":
            "Bootstrap 95% CI lower",

        "Value":
            stats_lookup.get(
                "paired_bootstrap_95_ci_lower",
                "",
            ),
    },

    {
        "Metric":
            "Bootstrap 95% CI upper",

        "Value":
            stats_lookup.get(
                "paired_bootstrap_95_ci_upper",
                "",
            ),
    },

    {
        "Metric":
            "Wilcoxon p-value",

        "Value":
            stats_lookup.get(
                "wilcoxon_p_value",
                "",
            ),
    },

    {
        "Metric":
            "Paired standardized effect",

        "Value":
            stats_lookup.get(
                "paired_standardized_effect",
                "",
            ),
    },
]


table2 = pd.DataFrame(
    statistics_rows
)


table2.to_csv(
    OUTPUT_DIR
    / "table_v33_final_paired_statistics.csv",
    index=False,
)


# ============================================================
# TABLE 3
# STAGE-WISE K=10 PERFORMANCE
# ============================================================

table3 = final_stage_df[
    [
        "component_index",
        "mean_lsq_quality",
        "mean_hybrid_quality",
        "mean_hybrid_minus_lsq",
        "hybrid_beats_lsq_percent",
    ]
].copy()


table3 = table3.rename(
    columns={
        "component_index":
            "Stage",

        "mean_lsq_quality":
            "LSQ mean quality",

        "mean_hybrid_quality":
            "Hybrid mean quality",

        "mean_hybrid_minus_lsq":
            "Hybrid - LSQ",

        "hybrid_beats_lsq_percent":
            "Hybrid wins (%)",
    }
)


table3.to_csv(
    OUTPUT_DIR
    / "table_v33_stagewise_K10.csv",
    index=False,
)


# ============================================================
# TABLE 4
# ROBUSTNESS REGIME
# ============================================================

table4 = robustness_df[
    [
        "nonlinearity_strength",
        "measurement_noise_mm",
        "mean_lsq_quality",
        "mean_locked_hybrid_quality",
        "mean_hybrid_minus_lsq",
        "relative_hybrid_improvement_percent",
        "hybrid_beats_lsq_percent",
    ]
].copy()


table4 = table4.rename(
    columns={
        "nonlinearity_strength":
            "Nonlinearity strength",

        "measurement_noise_mm":
            "Measurement noise (mm)",

        "mean_lsq_quality":
            "LSQ mean quality",

        "mean_locked_hybrid_quality":
            "Hybrid mean quality",

        "mean_hybrid_minus_lsq":
            "Hybrid - LSQ",

        "relative_hybrid_improvement_percent":
            "Hybrid improvement (%)",

        "hybrid_beats_lsq_percent":
            "Hybrid wins (%)",
    }
)


table4.to_csv(
    OUTPUT_DIR
    / "table_v33_robustness_regime.csv",
    index=False,
)


# ============================================================
# TABLE 5
# SEQUENCE LENGTH SCALING
# ============================================================

table5 = scaling_df[
    [
        "sequence_length",
        "mean_lsq_quality",
        "mean_hybrid_quality",
        "relative_hybrid_improvement_percent",
        "hybrid_beats_lsq_percent",
        "median_lsq_quality",
        "median_hybrid_quality",
        "p95_lsq_quality",
        "p95_hybrid_quality",
    ]
].copy()


table5 = table5.rename(
    columns={
        "sequence_length":
            "Sequence length K",

        "mean_lsq_quality":
            "LSQ mean",

        "mean_hybrid_quality":
            "Hybrid mean",

        "relative_hybrid_improvement_percent":
            "Hybrid improvement (%)",

        "hybrid_beats_lsq_percent":
            "Hybrid wins (%)",

        "median_lsq_quality":
            "LSQ median",

        "median_hybrid_quality":
            "Hybrid median",

        "p95_lsq_quality":
            "LSQ P95",

        "p95_hybrid_quality":
            "Hybrid P95",
    }
)


table5.to_csv(
    OUTPUT_DIR
    / "table_v33_sequence_length_scaling.csv",
    index=False,
)


# ============================================================
# TABLE 6
# DIRECT REFERENCE STABILITY
# ============================================================

table6 = reference_df[
    [
        "budget",
        "maxiter",
        "popsize",
        "mean_final_quality",
        "median_final_quality",
        "p95_final_quality",
        "mean_evaluations_per_decision",
        "formal_success_percent",
    ]
].copy()


table6 = table6.rename(
    columns={
        "budget":
            "Reference budget",

        "maxiter":
            "Max iterations",

        "popsize":
            "Population size",

        "mean_final_quality":
            "Mean final quality",

        "median_final_quality":
            "Median final quality",

        "p95_final_quality":
            "P95 final quality",

        "mean_evaluations_per_decision":
            "Mean evaluations / decision",

        "formal_success_percent":
            "Formal success (%)",
    }
)


table6.to_csv(
    OUTPUT_DIR
    / "table_v33_direct_reference_stability.csv",
    index=False,
)


# ============================================================
# TABLE 7
# FINAL UTILIZATION
# ============================================================

table7 = final_utilization_df.copy()


table7.to_csv(
    OUTPUT_DIR
    / "table_v33_correction_utilization.csv",
    index=False,
)


# ============================================================
# TABLE 8
# RESIDUAL ML PERFORMANCE
# ============================================================

table8 = residual_summary_df.copy()


table8.to_csv(
    OUTPUT_DIR
    / "table_v33_residual_ml_performance.csv",
    index=False,
)


# ============================================================
# README / INDEX
# ============================================================

index_text = """
V3.3 FINAL THESIS TABLE PACKAGE

Generated from validated V3.3 result CSV files only.

Tables:

1. table_v33_final_independent_method_comparison.csv
   Primary final controller comparison.

2. table_v33_final_paired_statistics.csv
   Final paired statistical evidence for locked hybrid vs LSQ.

3. table_v33_stagewise_K10.csv
   Stage-wise performance across the ten-component sequential assembly.

4. table_v33_robustness_regime.csv
   Nonlinearity and measurement-noise robustness results.

5. table_v33_sequence_length_scaling.csv
   K=5 vs K=10 scaling results.

6. table_v33_direct_reference_stability.csv
   Numerical-reference budget stability.

7. table_v33_correction_utilization.csv
   Correction-utilization comparison.

8. table_v33_residual_ml_performance.csv
   Residual-model prediction performance.

No technical experiment was rerun to generate these tables.
"""


(
    OUTPUT_DIR
    / "README_TABLES.txt"
).write_text(
    index_text.strip(),
    encoding="utf-8",
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 FINAL THESIS TABLE GENERATION COMPLETED"
)

print(
    "============================================================"
)

print(
    f"\nOutput folder:\n{OUTPUT_DIR}"
)

print(
    "\nGenerated files:"
)


for path in sorted(
    OUTPUT_DIR.iterdir()
):

    if path.is_file():

        print(
            f"  {path.name}"
        )