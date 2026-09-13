"""
Phase 8 final statistical analysis.

Purpose
-------
Perform the predefined paired statistical analysis for the independent
Phase 8 final controller evaluation.

The three comparisons form a separate Phase 8 statistical family:

H1: Composite-Q + Random Forest versus Composite-Q
H2: Composite-Q + Gradient Boosting versus Composite-Q
H3: Composite-Q + Random Forest versus Composite-Q + Gradient Boosting

For every comparison:

    difference = quality_A - quality_B

Therefore:

    negative difference -> method A is better
    positive difference -> method B is better

The analysis reports:

- paired mean difference;
- paired median difference;
- assembly-level win/loss/tie rates;
- paired bootstrap 95% confidence interval for the mean difference;
- Wilcoxon signed-rank test;
- paired standardized mean effect;
- Holm-adjusted p-value.

Statistical significance is interpreted separately from practical
engineering magnitude.

No model retraining, tuning, or controller modification is performed.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


# ============================================================
# PHASE 8 CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase8_config import (  # noqa: E402
    TABLES_DIR,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_RANDOM_STATE,
    SIGNIFICANCE_LEVEL,
    N_FINAL_ASSEMBLIES,
    ensure_phase8_directories,
)


# ============================================================
# INPUT / OUTPUT
# ============================================================

FINAL_ASSEMBLY_FILE = (
    TABLES_DIR
    / "phase8_final_controller_assembly_results.csv"
)

STATISTICS_OUTPUT_FILE = (
    TABLES_DIR
    / "phase8_final_statistical_results.csv"
)


# ============================================================
# COMPARISONS
# ============================================================

COMPARISONS = [
    {
        "hypothesis":
            "H1",

        "method_a":
            "COMPOSITE_Q_PLUS_RF",

        "method_b":
            "COMPOSITE_Q",

        "label":
            "Composite-Q + RF vs Composite-Q",
    },
    {
        "hypothesis":
            "H2",

        "method_a":
            "COMPOSITE_Q_PLUS_GB",

        "method_b":
            "COMPOSITE_Q",

        "label":
            "Composite-Q + GB vs Composite-Q",
    },
    {
        "hypothesis":
            "H3",

        "method_a":
            "COMPOSITE_Q_PLUS_RF",

        "method_b":
            "COMPOSITE_Q_PLUS_GB",

        "label":
            "Composite-Q + RF vs Composite-Q + GB",
    },
]


# ============================================================
# BOOTSTRAP
# ============================================================

def paired_bootstrap_ci(
    differences,
    n_bootstrap,
    seed,
):
    """
    Paired bootstrap confidence interval for the mean difference.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(
        differences
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for index in range(
        n_bootstrap
    ):
        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_means[
            index
        ] = float(
            np.mean(
                differences[
                    sample_indices
                ]
            )
        )

    lower = float(
        np.percentile(
            bootstrap_means,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            97.5,
        )
    )

    return lower, upper


# ============================================================
# EFFECT SIZE
# ============================================================

def paired_standardized_effect(
    differences,
):
    """
    Standardized paired mean difference:

        mean(difference) / sd(difference)

    Negative -> method A tends to be better.
    Positive -> method B tends to be better.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    standard_deviation = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    if np.isclose(
        standard_deviation,
        0.0,
        atol=1e-30,
    ):
        return np.nan

    return float(
        np.mean(
            differences
        )
        /
        standard_deviation
    )


# ============================================================
# WILCOXON
# ============================================================

def paired_wilcoxon(
    differences,
):
    """
    Two-sided Wilcoxon signed-rank test.

    Returns p = 1.0 when all paired differences are numerically zero.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    nonzero = (
        np.abs(
            differences
        )
        >
        1e-15
    )

    if not np.any(
        nonzero
    ):
        return np.nan, 1.0

    result = wilcoxon(
        differences,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_adjust(
    p_values,
):
    """
    Holm-Bonferroni adjustment for one family of hypotheses.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, original_index in enumerate(
        order
    ):
        multiplier = (
            m
            -
            rank
        )

        adjusted_value = min(
            1.0,
            multiplier
            *
            p_values[
                original_index
            ],
        )

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted_sorted[
            rank
        ] = running_max

    adjusted = np.empty(
        m,
        dtype=float,
    )

    for rank, original_index in enumerate(
        order
    ):
        adjusted[
            original_index
        ] = adjusted_sorted[
            rank
        ]

    return adjusted


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_phase8_directories()

    if not FINAL_ASSEMBLY_FILE.exists():
        raise FileNotFoundError(
            f"Final assembly results not found:\n"
            f"{FINAL_ASSEMBLY_FILE}"
        )

    final_df = pd.read_csv(
        FINAL_ASSEMBLY_FILE
    )

    # ========================================================
    # BASIC FINAL-DATA VALIDATION
    # ========================================================

    required_columns = {
        "assembly_index",
        "method",
        "final_quality",
    }

    missing_columns = (
        required_columns
        -
        set(
            final_df.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Missing columns in final assembly results: "
            f"{sorted(missing_columns)}"
        )

    pivot = final_df.pivot(
        index=
            "assembly_index",

        columns=
            "method",

        values=
            "final_quality",
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
            f"Expected {N_FINAL_ASSEMBLIES} final assemblies, "
            f"found {len(pivot)}."
        )

    if pivot.isna().any().any():
        raise RuntimeError(
            "NaN detected in paired final-quality table."
        )

    # ========================================================
    # ANALYZE PREDEFINED COMPARISONS
    # ========================================================

    result_rows = []

    raw_p_values = []

    for comparison_index, comparison in enumerate(
        COMPARISONS
    ):

        method_a = comparison[
            "method_a"
        ]

        method_b = comparison[
            "method_b"
        ]

        a = pivot[
            method_a
        ].to_numpy(
            dtype=float
        )

        b = pivot[
            method_b
        ].to_numpy(
            dtype=float
        )

        differences = (
            a
            -
            b
        )

        mean_a = float(
            np.mean(
                a
            )
        )

        mean_b = float(
            np.mean(
                b
            )
        )

        median_a = float(
            np.median(
                a
            )
        )

        median_b = float(
            np.median(
                b
            )
        )

        mean_difference = float(
            np.mean(
                differences
            )
        )

        median_difference = float(
            np.median(
                differences
            )
        )

        wins = int(
            np.sum(
                differences
                <
                -1e-12
            )
        )

        losses = int(
            np.sum(
                differences
                >
                1e-12
            )
        )

        ties = int(
            len(
                differences
            )
            -
            wins
            -
            losses
        )

        win_rate = float(
            100.0
            *
            wins
            /
            len(
                differences
            )
        )

        loss_rate = float(
            100.0
            *
            losses
            /
            len(
                differences
            )
        )

        tie_rate = float(
            100.0
            *
            ties
            /
            len(
                differences
            )
        )

        bootstrap_low, bootstrap_high = (
            paired_bootstrap_ci(
                differences=
                    differences,

                n_bootstrap=
                    BOOTSTRAP_SAMPLES,

                seed=
                    BOOTSTRAP_RANDOM_STATE
                    +
                    comparison_index,
            )
        )

        wilcoxon_statistic, raw_p_value = (
            paired_wilcoxon(
                differences
            )
        )

        effect_size = (
            paired_standardized_effect(
                differences
            )
        )

        relative_difference_percent = float(
            100.0
            *
            mean_difference
            /
            max(
                abs(
                    mean_b
                ),
                1e-30,
            )
        )

        raw_p_values.append(
            raw_p_value
        )

        result_rows.append(
            {
                "hypothesis":
                    comparison[
                        "hypothesis"
                    ],

                "comparison":
                    comparison[
                        "label"
                    ],

                "method_a":
                    method_a,

                "method_b":
                    method_b,

                "n_pairs":
                    len(
                        differences
                    ),

                "mean_method_a":
                    mean_a,

                "mean_method_b":
                    mean_b,

                "median_method_a":
                    median_a,

                "median_method_b":
                    median_b,

                "mean_difference_a_minus_b":
                    mean_difference,

                "median_difference_a_minus_b":
                    median_difference,

                "relative_mean_difference_percent":
                    relative_difference_percent,

                "method_a_wins":
                    wins,

                "method_a_losses":
                    losses,

                "ties":
                    ties,

                "method_a_win_rate_percent":
                    win_rate,

                "method_a_loss_rate_percent":
                    loss_rate,

                "tie_rate_percent":
                    tie_rate,

                "bootstrap_95_ci_low":
                    bootstrap_low,

                "bootstrap_95_ci_high":
                    bootstrap_high,

                "wilcoxon_statistic":
                    wilcoxon_statistic,

                "raw_p_value":
                    raw_p_value,

                "paired_standardized_effect":
                    effect_size,
            }
        )

    # ========================================================
    # HOLM FAMILY ADJUSTMENT
    # ========================================================

    adjusted_p_values = (
        holm_adjust(
            raw_p_values
        )
    )

    for row, adjusted_p in zip(
        result_rows,
        adjusted_p_values,
    ):

        row[
            "holm_adjusted_p_value"
        ] = float(
            adjusted_p
        )

        row[
            "significant_after_holm"
        ] = bool(
            adjusted_p
            <
            SIGNIFICANCE_LEVEL
        )

    result_df = pd.DataFrame(
        result_rows
    )

    result_df.to_csv(
        STATISTICS_OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8: FINAL PAIRED STATISTICAL ANALYSIS"
    )

    print(
        "============================================================"
    )

    print(
        f"\nFinal assemblies : {len(pivot)}"
    )

    print(
        f"Bootstrap samples: {BOOTSTRAP_SAMPLES}"
    )

    print(
        f"Alpha            : {SIGNIFICANCE_LEVEL}"
    )

    print(
        "\nDifference convention:"
    )

    print(
        "  difference = method A - method B"
    )

    print(
        "  negative -> method A better"
    )

    print(
        "  positive -> method B better"
    )

    for row in result_rows:

        print(
            "\n"
            "------------------------------------------------------------"
        )

        print(
            f"{row['hypothesis']}: "
            f"{row['comparison']}"
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            f"Mean A                    : "
            f"{row['mean_method_a']:.12f}"
        )

        print(
            f"Mean B                    : "
            f"{row['mean_method_b']:.12f}"
        )

        print(
            f"Mean difference A-B       : "
            f"{row['mean_difference_a_minus_b']:.12e}"
        )

        print(
            f"Median difference A-B     : "
            f"{row['median_difference_a_minus_b']:.12e}"
        )

        print(
            f"Relative mean difference  : "
            f"{row['relative_mean_difference_percent']:.8f}%"
        )

        print(
            f"A win / loss / tie        : "
            f"{row['method_a_wins']} / "
            f"{row['method_a_losses']} / "
            f"{row['ties']}"
        )

        print(
            f"A win rate                : "
            f"{row['method_a_win_rate_percent']:.2f}%"
        )

        print(
            f"Bootstrap 95% CI          : "
            f"[{row['bootstrap_95_ci_low']:.12e}, "
            f"{row['bootstrap_95_ci_high']:.12e}]"
        )

        print(
            f"Wilcoxon raw p            : "
            f"{row['raw_p_value']:.12e}"
        )

        print(
            f"Holm-adjusted p           : "
            f"{row['holm_adjusted_p_value']:.12e}"
        )

        print(
            f"Paired standardized effect: "
            f"{row['paired_standardized_effect']:.6f}"
        )

        print(
            f"Significant after Holm    : "
            f"{row['significant_after_holm']}"
        )

    print(
        "\nSaved statistical results:"
    )

    print(
        STATISTICS_OUTPUT_FILE
    )

    print(
        "\nImportant:"
    )

    print(
        "Statistical significance and practical engineering magnitude "
        "must be interpreted separately."
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 8 FINAL STATISTICAL ANALYSIS COMPLETED"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()