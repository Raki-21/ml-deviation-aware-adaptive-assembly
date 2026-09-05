"""
V3.3 hybrid architecture development analysis.

Purpose
-------
Analyze the 60-assembly development experiment before locking
the final V3.3 hybrid architecture.

This script compares:

    LSQ
    LSQ + ML theta
    LSQ + ML z + theta
    LSQ + ML all

using paired assembly-level statistics.

Important
---------
The goal is NOT simply to select the method with the smallest
sample mean.

If a more complex hybrid provides only negligible or statistically
uncertain improvement over a simpler hybrid, the simpler architecture
may be preferred before final independent validation.
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
    / "residual_hybrid_closed_loop_development"
)


INPUT_FILE = (
    RESULTS_DIR
    / "v33_residual_hybrid_development_assembly_results.csv"
)


OUTPUT_FILE = (
    RESULTS_DIR
    / "v33_hybrid_architecture_development_statistics.csv"
)


PAIRWISE_FILE = (
    RESULTS_DIR
    / "v33_hybrid_architecture_pairwise_assembly_differences.csv"
)


# ============================================================
# METHODS
# ============================================================

LSQ = "LSQ"

THETA = "LSQ_PLUS_ML_THETA"

Z_THETA = "LSQ_PLUS_ML_Z_THETA"

ALL = "LSQ_PLUS_ML_ALL"


METHODS = [
    LSQ,
    THETA,
    Z_THETA,
    ALL,
]


# ============================================================
# SETTINGS
# ============================================================

BOOTSTRAP_REPETITIONS = 20000

BOOTSTRAP_SEED = 20260911


# ============================================================
# LOAD RESULTS
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nDevelopment assembly result file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


development_df = (
    df[
        df[
            "method"
        ]
        .isin(
            METHODS
        )
    ]
    .copy()
)


pivot = development_df.pivot(
    index="assembly_index",
    columns="method",
    values="final_quality",
)


missing_methods = [
    method
    for method in METHODS
    if method not in pivot.columns
]


if missing_methods:

    raise RuntimeError(
        "\nMissing required methods:\n"
        +
        "\n".join(
            missing_methods
        )
    )


pivot = pivot.dropna(
    subset=METHODS
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 HYBRID ARCHITECTURE DEVELOPMENT ANALYSIS"
)

print(
    "============================================================"
)


print(
    f"\nPaired assemblies : {len(pivot)}"
)


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


def paired_bootstrap_ci(
    differences,
):
    """
    95% paired bootstrap confidence interval for the mean
    difference.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )


    n = len(
        differences
    )


    bootstrap_means = np.empty(
        BOOTSTRAP_REPETITIONS,
        dtype=float,
    )


    for i in range(
        BOOTSTRAP_REPETITIONS
    ):

        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )


        bootstrap_means[i] = np.mean(
            differences[
                sample_indices
            ]
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


# ============================================================
# PAIRED COMPARISON
# ============================================================

def compare_methods(
    reference_method,
    candidate_method,
):
    """
    Difference definition:

        candidate - reference

    Negative values mean candidate is better because lower
    quality score is preferred.
    """

    reference = (
        pivot[
            reference_method
        ]
        .to_numpy(
            dtype=float
        )
    )


    candidate = (
        pivot[
            candidate_method
        ]
        .to_numpy(
            dtype=float
        )
    )


    difference = (
        candidate
        -
        reference
    )


    mean_difference = float(
        np.mean(
            difference
        )
    )


    median_difference = float(
        np.median(
            difference
        )
    )


    ci_lower, ci_upper = (
        paired_bootstrap_ci(
            difference
        )
    )


    candidate_win_rate = float(
        100.0
        *
        np.mean(
            difference
            <
            -1e-12
        )
    )


    reference_win_rate = float(
        100.0
        *
        np.mean(
            difference
            >
            1e-12
        )
    )


    tie_rate = float(
        100.0
        -
        candidate_win_rate
        -
        reference_win_rate
    )


    # Wilcoxon may fail if every paired difference is exactly zero.
    try:

        wilcoxon_result = wilcoxon(
            candidate,
            reference,
            alternative="two-sided",
            zero_method="wilcox",
        )


        wilcoxon_statistic = float(
            wilcoxon_result.statistic
        )


        wilcoxon_p_value = float(
            wilcoxon_result.pvalue
        )


    except ValueError:

        wilcoxon_statistic = np.nan

        wilcoxon_p_value = 1.0


    return {
        "reference_method":
            reference_method,

        "candidate_method":
            candidate_method,

        "n_assemblies":
            len(
                difference
            ),

        "reference_mean_quality":
            float(
                np.mean(
                    reference
                )
            ),

        "candidate_mean_quality":
            float(
                np.mean(
                    candidate
                )
            ),

        "mean_difference_candidate_minus_reference":
            mean_difference,

        "median_difference_candidate_minus_reference":
            median_difference,

        "bootstrap_95_ci_lower":
            ci_lower,

        "bootstrap_95_ci_upper":
            ci_upper,

        "candidate_beats_reference_percent":
            candidate_win_rate,

        "reference_beats_candidate_percent":
            reference_win_rate,

        "tie_percent":
            tie_rate,

        "wilcoxon_statistic":
            wilcoxon_statistic,

        "wilcoxon_p_value":
            wilcoxon_p_value,
    }


# ============================================================
# COMPARISONS
# ============================================================

comparison_pairs = [
    (
        LSQ,
        THETA,
    ),

    (
        LSQ,
        Z_THETA,
    ),

    (
        LSQ,
        ALL,
    ),

    (
        THETA,
        Z_THETA,
    ),

    (
        Z_THETA,
        ALL,
    ),

    (
        THETA,
        ALL,
    ),
]


rows = []


for reference_method, candidate_method in comparison_pairs:

    rows.append(
        compare_methods(
            reference_method=reference_method,
            candidate_method=candidate_method,
        )
    )


stats_df = pd.DataFrame(
    rows
)


stats_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SAVE ASSEMBLY-LEVEL DIFFERENCES
# ============================================================

pairwise_df = pd.DataFrame(
    {
        "assembly_index":
            pivot.index,

        "lsq_quality":
            pivot[
                LSQ
            ].values,

        "theta_quality":
            pivot[
                THETA
            ].values,

        "z_theta_quality":
            pivot[
                Z_THETA
            ].values,

        "all_quality":
            pivot[
                ALL
            ].values,
    }
)


pairwise_df[
    "theta_minus_lsq"
] = (
    pairwise_df[
        "theta_quality"
    ]
    -
    pairwise_df[
        "lsq_quality"
    ]
)


pairwise_df[
    "z_theta_minus_lsq"
] = (
    pairwise_df[
        "z_theta_quality"
    ]
    -
    pairwise_df[
        "lsq_quality"
    ]
)


pairwise_df[
    "all_minus_lsq"
] = (
    pairwise_df[
        "all_quality"
    ]
    -
    pairwise_df[
        "lsq_quality"
    ]
)


pairwise_df[
    "all_minus_z_theta"
] = (
    pairwise_df[
        "all_quality"
    ]
    -
    pairwise_df[
        "z_theta_quality"
    ]
)


pairwise_df.to_csv(
    PAIRWISE_FILE,
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
    "PAIRED METHOD COMPARISONS"
)

print(
    "============================================================"
)


for _, row in stats_df.iterrows():

    print(
        "\n"
        f"{row['candidate_method']} "
        f"vs {row['reference_method']}"
    )


    print(
        f"  Reference mean           : "
        f"{row['reference_mean_quality']:.6f}"
    )


    print(
        f"  Candidate mean           : "
        f"{row['candidate_mean_quality']:.6f}"
    )


    print(
        f"  Mean difference          : "
        f"{row['mean_difference_candidate_minus_reference']:.6f}"
    )


    print(
        f"  Median difference        : "
        f"{row['median_difference_candidate_minus_reference']:.6f}"
    )


    print(
        f"  95% bootstrap CI         : "
        f"[{row['bootstrap_95_ci_lower']:.6f}, "
        f"{row['bootstrap_95_ci_upper']:.6f}]"
    )


    print(
        f"  Candidate wins           : "
        f"{row['candidate_beats_reference_percent']:.2f}%"
    )


    print(
        f"  Reference wins           : "
        f"{row['reference_beats_candidate_percent']:.2f}%"
    )


    print(
        f"  Wilcoxon p               : "
        f"{row['wilcoxon_p_value']:.6g}"
    )


# ============================================================
# ARCHITECTURE LOCK RECOMMENDATION
# ============================================================

all_vs_ztheta = (
    stats_df[
        (
            stats_df[
                "reference_method"
            ]
            ==
            Z_THETA
        )
        &
        (
            stats_df[
                "candidate_method"
            ]
            ==
            ALL
        )
    ]
    .iloc[0]
)


ztheta_vs_theta = (
    stats_df[
        (
            stats_df[
                "reference_method"
            ]
            ==
            THETA
        )
        &
        (
            stats_df[
                "candidate_method"
            ]
            ==
            Z_THETA
        )
    ]
    .iloc[0]
)


print(
    "\n"
    "============================================================"
)

print(
    "ARCHITECTURE LOCK ANALYSIS"
)

print(
    "============================================================"
)


print(
    "\nALL versus Z+THETA:"
)


print(
    f"Mean gain from locator residual : "
    f"{-all_vs_ztheta['mean_difference_candidate_minus_reference']:.6f}"
)


print(
    f"95% CI for ALL-Z_THETA          : "
    f"[{all_vs_ztheta['bootstrap_95_ci_lower']:.6f}, "
    f"{all_vs_ztheta['bootstrap_95_ci_upper']:.6f}]"
)


print(
    f"Wilcoxon p                      : "
    f"{all_vs_ztheta['wilcoxon_p_value']:.6g}"
)


print(
    "\nZ+THETA versus THETA:"
)


print(
    f"Mean gain from z residual       : "
    f"{-ztheta_vs_theta['mean_difference_candidate_minus_reference']:.6f}"
)


print(
    f"95% CI for Z_THETA-THETA        : "
    f"[{ztheta_vs_theta['bootstrap_95_ci_lower']:.6f}, "
    f"{ztheta_vs_theta['bootstrap_95_ci_upper']:.6f}]"
)


print(
    f"Wilcoxon p                      : "
    f"{ztheta_vs_theta['wilcoxon_p_value']:.6g}"
)


# ============================================================
# CONSERVATIVE DECISION RULE
# ============================================================

locator_ci_entirely_better = bool(
    all_vs_ztheta[
        "bootstrap_95_ci_upper"
    ]
    <
    0.0
)


z_ci_entirely_better = bool(
    ztheta_vs_theta[
        "bootstrap_95_ci_upper"
    ]
    <
    0.0
)


print(
    "\n"
    "============================================================"
)

print(
    "PRE-FINAL ARCHITECTURE DECISION"
)

print(
    "============================================================"
)


if locator_ci_entirely_better:

    recommended_method = ALL

    print(
        "\nRecommendation: LOCK LSQ_PLUS_ML_ALL"
    )

    print(
        "The locator residual provides a paired development "
        "advantage whose 95% bootstrap CI remains below zero."
    )


elif z_ci_entirely_better:

    recommended_method = Z_THETA

    print(
        "\nRecommendation: LOCK LSQ_PLUS_ML_Z_THETA"
    )

    print(
        "The z residual provides a supported improvement over "
        "theta-only, while the extra locator contribution is "
        "not sufficiently supported."
    )


else:

    recommended_method = THETA

    print(
        "\nRecommendation: LOCK LSQ_PLUS_ML_THETA"
    )

    print(
        "The additional residual dimensions do not show a "
        "sufficiently supported paired advantage to justify "
        "the extra complexity."
    )


print(
    f"\nRecommended architecture: {recommended_method}"
)


print(
    "\nImportant:"
)


print(
    "This decision uses development data only."
)


print(
    "Once the architecture is locked, it must not be changed "
    "after seeing the final independent validation results."
)


print(
    "\nSaved statistics:"
)


print(
    OUTPUT_FILE
)


print(
    "\nSaved paired assembly differences:"
)


print(
    PAIRWISE_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.3 HYBRID ARCHITECTURE DEVELOPMENT ANALYSIS COMPLETED"
)

print(
    "============================================================"
)