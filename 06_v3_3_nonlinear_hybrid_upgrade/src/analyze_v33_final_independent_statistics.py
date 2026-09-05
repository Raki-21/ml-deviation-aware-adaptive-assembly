"""
V3.3 final independent statistical analysis.

Purpose
-------
Perform the final paired statistical evaluation of the already-locked
V3.3 hybrid architecture:

    LSQ_PLUS_ML_ALL

against deterministic LSQ on the untouched 300-assembly independent
validation population.

The architecture was locked before these validation results were seen.

This script does NOT retrain, tune, or select a controller.

Primary comparison
------------------
    Locked Hybrid vs LSQ

Secondary contextual comparison
-------------------------------
    Finite-budget nonlinear numerical reference

The numerical reference is not described as a proven global optimum.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


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
    / "final_independent_validation_v33"
)

ASSEMBLY_FILE = (
    RESULTS_DIR
    / "v33_final_independent_assembly_results.csv"
)

COMPONENT_FILE = (
    RESULTS_DIR
    / "v33_final_independent_component_results.csv"
)

STATISTICS_FILE = (
    RESULTS_DIR
    / "v33_final_independent_statistics.csv"
)

PAIRED_FILE = (
    RESULTS_DIR
    / "v33_final_independent_paired_results.csv"
)

STAGE_FILE = (
    RESULTS_DIR
    / "v33_final_independent_stage_statistics.csv"
)

UTILIZATION_FILE = (
    RESULTS_DIR
    / "v33_final_independent_utilization_statistics.csv"
)


# ============================================================
# METHODS
# ============================================================

METHOD_LSQ = "LSQ"

METHOD_HYBRID = "LSQ_PLUS_ML_ALL"

METHOD_DIRECT = "DIRECT_NONLINEAR_REFERENCE"


# ============================================================
# STATISTICAL SETTINGS
# ============================================================

BOOTSTRAP_REPETITIONS = 50000

BOOTSTRAP_SEED = 20260913

EXPECTED_ASSEMBLIES = 300

EXPECTED_SEQUENCE_LENGTH = 10


# ============================================================
# LOAD DATA
# ============================================================

if not ASSEMBLY_FILE.exists():

    raise FileNotFoundError(
        f"\nAssembly result file not found:\n{ASSEMBLY_FILE}"
    )


if not COMPONENT_FILE.exists():

    raise FileNotFoundError(
        f"\nComponent result file not found:\n{COMPONENT_FILE}"
    )


assembly_df = pd.read_csv(
    ASSEMBLY_FILE
)

component_df = pd.read_csv(
    COMPONENT_FILE
)


# ============================================================
# BASIC CONSISTENCY CHECKS
# ============================================================

required_assembly_methods = {
    METHOD_LSQ,
    METHOD_HYBRID,
    METHOD_DIRECT,
}


available_assembly_methods = set(
    assembly_df[
        "method"
    ]
    .astype(
        str
    )
)


missing_methods = (
    required_assembly_methods
    -
    available_assembly_methods
)


if missing_methods:

    raise RuntimeError(
        "\nMissing required methods:\n"
        +
        "\n".join(
            sorted(
                missing_methods
            )
        )
    )


n_assemblies = int(
    assembly_df[
        "assembly_index"
    ]
    .nunique()
)


if n_assemblies != EXPECTED_ASSEMBLIES:

    raise RuntimeError(
        f"\nExpected {EXPECTED_ASSEMBLIES} assemblies, "
        f"found {n_assemblies}."
    )


max_component_index = int(
    component_df[
        "component_index"
    ]
    .max()
)


if max_component_index != EXPECTED_SEQUENCE_LENGTH:

    raise RuntimeError(
        f"\nExpected sequence length {EXPECTED_SEQUENCE_LENGTH}, "
        f"found maximum component index {max_component_index}."
    )


# ============================================================
# ASSEMBLY-LEVEL PIVOT
# ============================================================

assembly_pivot = (
    assembly_df[
        assembly_df[
            "method"
        ]
        .isin(
            [
                METHOD_LSQ,
                METHOD_HYBRID,
                METHOD_DIRECT,
            ]
        )
    ]
    .pivot(
        index="assembly_index",
        columns="method",
        values="final_quality",
    )
)


assembly_pivot = (
    assembly_pivot
    .dropna(
        subset=[
            METHOD_LSQ,
            METHOD_HYBRID,
            METHOD_DIRECT,
        ]
    )
    .sort_index()
)


if len(
    assembly_pivot
) != EXPECTED_ASSEMBLIES:

    raise RuntimeError(
        "\nPaired assembly count is not 300."
    )


lsq_quality = (
    assembly_pivot[
        METHOD_LSQ
    ]
    .to_numpy(
        dtype=float
    )
)


hybrid_quality = (
    assembly_pivot[
        METHOD_HYBRID
    ]
    .to_numpy(
        dtype=float
    )
)


direct_quality = (
    assembly_pivot[
        METHOD_DIRECT
    ]
    .to_numpy(
        dtype=float
    )
)


# ============================================================
# PRIMARY PAIRED DIFFERENCE
# ============================================================

# Negative values mean the hybrid is better.
hybrid_minus_lsq = (
    hybrid_quality
    -
    lsq_quality
)


# Negative values mean direct reference is better than hybrid.
direct_minus_hybrid = (
    direct_quality
    -
    hybrid_quality
)


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


def paired_bootstrap_mean_ci(
    differences,
    repetitions=BOOTSTRAP_REPETITIONS,
):
    """
    Paired bootstrap 95% confidence interval for mean difference.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )


    n = len(
        differences
    )


    bootstrap_means = np.empty(
        repetitions,
        dtype=float,
    )


    for iteration in range(
        repetitions
    ):

        sampled_indices = rng.integers(
            0,
            n,
            size=n,
        )


        bootstrap_means[
            iteration
        ] = float(
            np.mean(
                differences[
                    sampled_indices
                ]
            )
        )


    lower = float(
        np.quantile(
            bootstrap_means,
            0.025,
        )
    )


    upper = float(
        np.quantile(
            bootstrap_means,
            0.975,
        )
    )


    return (
        lower,
        upper,
    )


hybrid_ci_lower, hybrid_ci_upper = (
    paired_bootstrap_mean_ci(
        hybrid_minus_lsq
    )
)


# ============================================================
# WILCOXON SIGNED-RANK TEST
# ============================================================

wilcoxon_result = wilcoxon(
    hybrid_quality,
    lsq_quality,
    alternative="two-sided",
    zero_method="wilcox",
)


wilcoxon_statistic = float(
    wilcoxon_result.statistic
)


wilcoxon_p_value = float(
    wilcoxon_result.pvalue
)


# ============================================================
# EFFECT SIZE
# ============================================================

difference_std = float(
    np.std(
        hybrid_minus_lsq,
        ddof=1,
    )
)


if difference_std > 0.0:

    paired_standardized_effect = float(
        np.mean(
            hybrid_minus_lsq
        )
        /
        difference_std
    )

else:

    paired_standardized_effect = np.nan


# ============================================================
# ASSEMBLY-LEVEL METRICS
# ============================================================

mean_lsq = float(
    np.mean(
        lsq_quality
    )
)


mean_hybrid = float(
    np.mean(
        hybrid_quality
    )
)


mean_direct = float(
    np.mean(
        direct_quality
    )
)


median_lsq = float(
    np.median(
        lsq_quality
    )
)


median_hybrid = float(
    np.median(
        hybrid_quality
    )
)


p95_lsq = float(
    np.quantile(
        lsq_quality,
        0.95,
    )
)


p95_hybrid = float(
    np.quantile(
        hybrid_quality,
        0.95,
    )
)


mean_difference = float(
    np.mean(
        hybrid_minus_lsq
    )
)


median_difference = float(
    np.median(
        hybrid_minus_lsq
    )
)


hybrid_win_percent = float(
    100.0
    *
    np.mean(
        hybrid_minus_lsq
        <
        -1e-12
    )
)


lsq_win_percent = float(
    100.0
    *
    np.mean(
        hybrid_minus_lsq
        >
        1e-12
    )
)


tie_percent = float(
    100.0
    -
    hybrid_win_percent
    -
    lsq_win_percent
)


relative_improvement_percent = float(
    100.0
    *
    (
        mean_lsq
        -
        mean_hybrid
    )
    /
    max(
        mean_lsq,
        1e-12,
    )
)


# ============================================================
# REFERENCE-GAP CONTEXT
# ============================================================

available_improvement_to_reference = float(
    mean_lsq
    -
    mean_direct
)


hybrid_improvement_over_lsq = float(
    mean_lsq
    -
    mean_hybrid
)


if available_improvement_to_reference > 0.0:

    reference_improvement_capture_percent = float(
        100.0
        *
        hybrid_improvement_over_lsq
        /
        available_improvement_to_reference
    )

else:

    reference_improvement_capture_percent = np.nan


hybrid_gap_to_reference = float(
    mean_hybrid
    -
    mean_direct
)


# ============================================================
# SAVE PAIRED ASSEMBLY RESULTS
# ============================================================

paired_df = pd.DataFrame(
    {
        "assembly_index":
            assembly_pivot.index.to_numpy(),

        "lsq_final_quality":
            lsq_quality,

        "hybrid_final_quality":
            hybrid_quality,

        "direct_reference_final_quality":
            direct_quality,

        "hybrid_minus_lsq":
            hybrid_minus_lsq,

        "direct_minus_hybrid":
            direct_minus_hybrid,

        "hybrid_beats_lsq":
            (
                hybrid_minus_lsq
                <
                -1e-12
            ),
    }
)


paired_df.to_csv(
    PAIRED_FILE,
    index=False,
)


# ============================================================
# STAGE-WISE ANALYSIS
# ============================================================

stage_rows = []


for component_index in range(
    1,
    EXPECTED_SEQUENCE_LENGTH
    +
    1
):

    stage_df = (
        component_df[
            component_df[
                "component_index"
            ]
            ==
            component_index
        ]
    )


    stage_pivot = (
        stage_df[
            stage_df[
                "method"
            ]
            .isin(
                [
                    METHOD_LSQ,
                    METHOD_HYBRID,
                ]
            )
        ]
        .pivot(
            index="assembly_index",
            columns="method",
            values="post_quality",
        )
        .dropna(
            subset=[
                METHOD_LSQ,
                METHOD_HYBRID,
            ]
        )
    )


    stage_lsq = (
        stage_pivot[
            METHOD_LSQ
        ]
        .to_numpy(
            dtype=float
        )
    )


    stage_hybrid = (
        stage_pivot[
            METHOD_HYBRID
        ]
        .to_numpy(
            dtype=float
        )
    )


    stage_difference = (
        stage_hybrid
        -
        stage_lsq
    )


    stage_rows.append(
        {
            "component_index":
                component_index,

            "assemblies":
                len(
                    stage_pivot
                ),

            "mean_lsq_quality":
                float(
                    np.mean(
                        stage_lsq
                    )
                ),

            "mean_hybrid_quality":
                float(
                    np.mean(
                        stage_hybrid
                    )
                ),

            "mean_hybrid_minus_lsq":
                float(
                    np.mean(
                        stage_difference
                    )
                ),

            "median_hybrid_minus_lsq":
                float(
                    np.median(
                        stage_difference
                    )
                ),

            "hybrid_beats_lsq_percent":
                float(
                    100.0
                    *
                    np.mean(
                        stage_difference
                        <
                        -1e-12
                    )
                ),
        }
    )


stage_stats_df = pd.DataFrame(
    stage_rows
)


stage_stats_df.to_csv(
    STAGE_FILE,
    index=False,
)


# ============================================================
# UTILIZATION ANALYSIS
# ============================================================

utilization_rows = []


for method in [
    METHOD_LSQ,
    METHOD_HYBRID,
    METHOD_DIRECT,
]:

    method_component_df = (
        component_df[
            component_df[
                "method"
            ]
            ==
            method
        ]
    )


    utilization_values = (
        method_component_df[
            "utilization"
        ]
        .to_numpy(
            dtype=float
        )
    )


    utilization_rows.append(
        {
            "method":
                method,

            "decisions":
                len(
                    utilization_values
                ),

            "mean_utilization":
                float(
                    np.mean(
                        utilization_values
                    )
                ),

            "median_utilization":
                float(
                    np.median(
                        utilization_values
                    )
                ),

            "p95_utilization":
                float(
                    np.quantile(
                        utilization_values,
                        0.95,
                    )
                ),

            "near_90_percent_capability_percent":
                float(
                    100.0
                    *
                    np.mean(
                        utilization_values
                        >=
                        0.90
                    )
                ),
        }
    )


utilization_df = pd.DataFrame(
    utilization_rows
)


utilization_df.to_csv(
    UTILIZATION_FILE,
    index=False,
)


# ============================================================
# FINAL STATISTICS TABLE
# ============================================================

statistics_rows = [
    {
        "metric":
            "n_independent_assemblies",

        "value":
            len(
                assembly_pivot
            ),
    },

    {
        "metric":
            "sequence_length",

        "value":
            EXPECTED_SEQUENCE_LENGTH,
    },

    {
        "metric":
            "mean_lsq_quality",

        "value":
            mean_lsq,
    },

    {
        "metric":
            "mean_locked_hybrid_quality",

        "value":
            mean_hybrid,
    },

    {
        "metric":
            "mean_direct_reference_quality",

        "value":
            mean_direct,
    },

    {
        "metric":
            "median_lsq_quality",

        "value":
            median_lsq,
    },

    {
        "metric":
            "median_locked_hybrid_quality",

        "value":
            median_hybrid,
    },

    {
        "metric":
            "p95_lsq_quality",

        "value":
            p95_lsq,
    },

    {
        "metric":
            "p95_locked_hybrid_quality",

        "value":
            p95_hybrid,
    },

    {
        "metric":
            "mean_hybrid_minus_lsq",

        "value":
            mean_difference,
    },

    {
        "metric":
            "median_hybrid_minus_lsq",

        "value":
            median_difference,
    },

    {
        "metric":
            "paired_bootstrap_95_ci_lower",

        "value":
            hybrid_ci_lower,
    },

    {
        "metric":
            "paired_bootstrap_95_ci_upper",

        "value":
            hybrid_ci_upper,
    },

    {
        "metric":
            "wilcoxon_statistic",

        "value":
            wilcoxon_statistic,
    },

    {
        "metric":
            "wilcoxon_p_value",

        "value":
            wilcoxon_p_value,
    },

    {
        "metric":
            "paired_standardized_effect",

        "value":
            paired_standardized_effect,
    },

    {
        "metric":
            "hybrid_beats_lsq_percent",

        "value":
            hybrid_win_percent,
    },

    {
        "metric":
            "lsq_beats_hybrid_percent",

        "value":
            lsq_win_percent,
    },

    {
        "metric":
            "tie_percent",

        "value":
            tie_percent,
    },

    {
        "metric":
            "relative_hybrid_improvement_percent",

        "value":
            relative_improvement_percent,
    },

    {
        "metric":
            "hybrid_gap_to_direct_reference",

        "value":
            hybrid_gap_to_reference,
    },

    {
        "metric":
            "reference_improvement_capture_percent",

        "value":
            reference_improvement_capture_percent,
    },
]


statistics_df = pd.DataFrame(
    statistics_rows
)


statistics_df.to_csv(
    STATISTICS_FILE,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 FINAL INDEPENDENT STATISTICAL ANALYSIS"
)

print(
    "============================================================"
)


print(
    f"\nIndependent assemblies      : "
    f"{len(assembly_pivot)}"
)


print(
    f"Sequence length             : "
    f"{EXPECTED_SEQUENCE_LENGTH}"
)


print(
    "\nPrimary locked comparison:"
)


print(
    f"LSQ mean quality            : "
    f"{mean_lsq:.6f}"
)


print(
    f"Locked hybrid mean quality  : "
    f"{mean_hybrid:.6f}"
)


print(
    f"Mean paired difference      : "
    f"{mean_difference:.6f}"
)


print(
    f"Median paired difference    : "
    f"{median_difference:.6f}"
)


print(
    f"Relative improvement        : "
    f"{relative_improvement_percent:.2f}%"
)


print(
    f"Hybrid beats LSQ            : "
    f"{hybrid_win_percent:.2f}%"
)


print(
    f"LSQ beats hybrid            : "
    f"{lsq_win_percent:.2f}%"
)


print(
    "\nStatistical evidence:"
)


print(
    f"95% paired bootstrap CI     : "
    f"[{hybrid_ci_lower:.6f}, "
    f"{hybrid_ci_upper:.6f}]"
)


print(
    f"Wilcoxon p                  : "
    f"{wilcoxon_p_value:.8g}"
)


print(
    f"Paired standardized effect  : "
    f"{paired_standardized_effect:.4f}"
)


print(
    "\nDistribution metrics:"
)


print(
    f"LSQ median                  : "
    f"{median_lsq:.6f}"
)


print(
    f"Hybrid median               : "
    f"{median_hybrid:.6f}"
)


print(
    f"LSQ P95                     : "
    f"{p95_lsq:.6f}"
)


print(
    f"Hybrid P95                  : "
    f"{p95_hybrid:.6f}"
)


print(
    "\nFinite-budget reference context:"
)


print(
    f"Direct reference mean       : "
    f"{mean_direct:.6f}"
)


print(
    f"Hybrid-reference gap        : "
    f"{hybrid_gap_to_reference:.6f}"
)


print(
    f"Reference improvement captured: "
    f"{reference_improvement_capture_percent:.2f}%"
)


print(
    "\n"
    "============================================================"
)

print(
    "STAGE-WISE LOCKED HYBRID VS LSQ"
)

print(
    "============================================================"
)


for _, row in stage_stats_df.iterrows():

    print(
        f"\nStage {int(row['component_index'])}"
    )


    print(
        f"  LSQ mean quality          : "
        f"{row['mean_lsq_quality']:.6f}"
    )


    print(
        f"  Hybrid mean quality       : "
        f"{row['mean_hybrid_quality']:.6f}"
    )


    print(
        f"  Hybrid - LSQ              : "
        f"{row['mean_hybrid_minus_lsq']:.6f}"
    )


    print(
        f"  Hybrid wins               : "
        f"{row['hybrid_beats_lsq_percent']:.2f}%"
    )


print(
    "\n"
    "============================================================"
)

print(
    "FINAL STATISTICAL GATE"
)

print(
    "============================================================"
)


ci_supports_hybrid = bool(
    hybrid_ci_upper
    <
    0.0
)


if (
    ci_supports_hybrid
    and
    mean_difference
    <
    0.0
):

    print(
        "\nPASS:"
    )


    print(
        "The pre-locked V3.3 hybrid shows a statistically "
        "supported mean improvement over deterministic LSQ "
        "on the 300-assembly independent validation population."
    )


else:

    print(
        "\nNOT SUPPORTED:"
    )


    print(
        "The independent paired evidence does not establish "
        "a statistically supported mean advantage over LSQ."
    )


print(
    "\nNo model retraining, architecture reselection, or "
    "post-validation controller tuning was performed."
)


print(
    "\nSaved final statistics:"
)

print(
    STATISTICS_FILE
)


print(
    "\nSaved stage statistics:"
)

print(
    STAGE_FILE
)


print(
    "\nSaved utilization statistics:"
)

print(
    UTILIZATION_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 FINAL INDEPENDENT STATISTICAL ANALYSIS COMPLETED"
)

print(
    "============================================================"
)