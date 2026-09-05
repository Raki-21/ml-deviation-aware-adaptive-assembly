import os
import numpy as np
import pandas as pd


# ============================================================
# VERSION 3.2
# CORRECTED SELECTIVE BO TRIGGER CALIBRATION
# ============================================================
#
# WHY THIS SCRIPT EXISTS
#
# The first trigger calibration used:
#
#       high_regret = regret >= P90(regret)
#
# But the regret distribution contains many exact zeros.
#
# Therefore:
#
#       P90(regret) = 0
#
# and every non-negative decision was incorrectly classified
# as "high regret".
#
#
# This corrected calibration DOES NOT use an arbitrary binary
# high-regret threshold.
#
#
# Instead it measures:
#
# 1. Exact-best-miss capture
#
# 2. Positive-regret decision capture
#
# 3. TOTAL REGRET CAPTURE
#
#       sum(regret on triggered decisions)
#       ----------------------------------
#       sum(regret on all decisions)
#
#
# Total regret capture is particularly useful because it asks:
#
#   "How much of the actual decision-quality loss is located
#    inside the subset that the online trigger identifies?"
#
#
# ONLINE TRIGGER INPUTS REMAIN ONLY:
#
#   - predicted best-vs-second-best margin
#   - Random-Forest tree disagreement
#   - correction utilization
#
#
# Actual regret and exact-best information remain POST-HOC
# calibration labels only.
# ============================================================


# ============================================================
# PATHS
# ============================================================

SIGNAL_FILE = (
    "data/processed/"
    "v3_2_predecision_ambiguity_signals.csv"
)

GRID_OUTPUT = (
    "results/tables/"
    "v3_2_corrected_selective_bo_trigger_grid.csv"
)

PARETO_OUTPUT = (
    "results/tables/"
    "v3_2_corrected_selective_bo_trigger_pareto.csv"
)

SUMMARY_OUTPUT = (
    "results/tables/"
    "v3_2_corrected_selective_bo_trigger_summary.csv"
)

DECISION_OUTPUT = (
    "data/processed/"
    "v3_2_corrected_selective_bo_trigger_labels.csv"
)


# ============================================================
# FILE CHECK
# ============================================================

if not os.path.exists(
    SIGNAL_FILE
):

    raise FileNotFoundError(
        "\nPre-decision ambiguity signal dataset was not found."
    )


# ============================================================
# SETTINGS
# ============================================================

MAX_TRIGGER_RATE_PERCENT = 20.0

REGRET_EPSILON = 1e-12


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CORRECTED SELECTIVE BO TRIGGER CALIBRATION"
)

print(
    "============================================================"
)


df = pd.read_csv(
    SIGNAL_FILE
)


required_columns = [
    "assembly_id",
    "component_index",
    "predicted_quality_margin",
    "selected_tree_std",
    "selected_utilization",
    "actual_regret",
    "exact_best_miss",
]


missing_columns = [
    column
    for column
    in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required signal columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# CLEAN LABELS
# ============================================================

df[
    "actual_regret"
] = (
    df[
        "actual_regret"
    ]
    .clip(
        lower=0.0
    )
)


df[
    "positive_regret"
] = (
    df[
        "actual_regret"
    ]
    >
    REGRET_EPSILON
)


df[
    "exact_best_miss"
] = (
    df[
        "exact_best_miss"
    ]
    .astype(
        bool
    )
)


n_decisions = len(
    df
)


n_positive_regret = int(
    df[
        "positive_regret"
    ]
    .sum()
)


n_exact_misses = int(
    df[
        "exact_best_miss"
    ]
    .sum()
)


total_regret = float(
    df[
        "actual_regret"
    ]
    .sum()
)


mean_regret = float(
    df[
        "actual_regret"
    ]
    .mean()
)


positive_regret_values = (
    df.loc[
        df[
            "positive_regret"
        ],
        "actual_regret",
    ]
)


print(
    f"\nDecisions                   : "
    f"{n_decisions}"
)


print(
    f"Positive-regret decisions   : "
    f"{n_positive_regret}"
)


print(
    f"Exact-best misses           : "
    f"{n_exact_misses}"
)


print(
    f"Total accumulated regret    : "
    f"{total_regret:.6f}"
)


print(
    f"Mean regret / decision      : "
    f"{mean_regret:.6f}"
)


if len(
    positive_regret_values
) > 0:

    print(
        "\nPositive-regret distribution:"
    )


    print(
        positive_regret_values
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
        .round(
            6
        )
        .to_string()
    )


# ============================================================
# BASE RATES
# ============================================================

base_positive_regret_rate = (
    n_positive_regret
    /
    n_decisions
    *
    100.0
)


base_exact_miss_rate = (
    n_exact_misses
    /
    n_decisions
    *
    100.0
)


# ============================================================
# TRIGGER EVALUATION
# ============================================================

def evaluate_trigger(
    trigger_mask,
    rule_name,
    margin_threshold=np.nan,
    uncertainty_threshold=np.nan,
    utilization_threshold=np.nan,
):


    trigger_mask = pd.Series(
        trigger_mask,
        index=df.index,
    ).astype(
        bool
    )


    n_triggered = int(
        trigger_mask.sum()
    )


    trigger_rate = (
        n_triggered
        /
        n_decisions
        *
        100.0
    )


    # --------------------------------------------------------
    # Positive-regret capture
    # --------------------------------------------------------

    positive_regret_capture = (
        (
            trigger_mask
            &
            df[
                "positive_regret"
            ]
        )
        .sum()
        /
        max(
            n_positive_regret,
            1,
        )
        *
        100.0
    )


    # --------------------------------------------------------
    # Exact-best-miss capture
    # --------------------------------------------------------

    exact_miss_capture = (
        (
            trigger_mask
            &
            df[
                "exact_best_miss"
            ]
        )
        .sum()
        /
        max(
            n_exact_misses,
            1,
        )
        *
        100.0
    )


    # --------------------------------------------------------
    # TOTAL REGRET CAPTURE
    #
    # Main continuous decision-loss metric.
    # --------------------------------------------------------

    triggered_regret_sum = float(
        df.loc[
            trigger_mask,
            "actual_regret",
        ]
        .sum()
    )


    regret_capture = (
        triggered_regret_sum
        /
        max(
            total_regret,
            REGRET_EPSILON,
        )
        *
        100.0
    )


    # --------------------------------------------------------
    # Precision-like metrics
    # --------------------------------------------------------

    positive_regret_precision = (
        (
            trigger_mask
            &
            df[
                "positive_regret"
            ]
        )
        .sum()
        /
        max(
            n_triggered,
            1,
        )
        *
        100.0
    )


    exact_miss_precision = (
        (
            trigger_mask
            &
            df[
                "exact_best_miss"
            ]
        )
        .sum()
        /
        max(
            n_triggered,
            1,
        )
        *
        100.0
    )


    # --------------------------------------------------------
    # LIFT
    #
    # A useful trigger should concentrate difficult cases above
    # their base prevalence.
    # --------------------------------------------------------

    positive_regret_lift = (
        positive_regret_precision
        /
        max(
            base_positive_regret_rate,
            REGRET_EPSILON,
        )
    )


    exact_miss_lift = (
        exact_miss_precision
        /
        max(
            base_exact_miss_rate,
            REGRET_EPSILON,
        )
    )


    # --------------------------------------------------------
    # Mean regret inside/outside trigger
    # --------------------------------------------------------

    triggered_mean_regret = (
        df.loc[
            trigger_mask,
            "actual_regret",
        ]
        .mean()
        if n_triggered > 0
        else np.nan
    )


    nontriggered_mean_regret = (
        df.loc[
            ~trigger_mask,
            "actual_regret",
        ]
        .mean()
        if (
            ~trigger_mask
        ).sum() > 0
        else np.nan
    )


    regret_concentration = (
        triggered_mean_regret
        /
        max(
            mean_regret,
            REGRET_EPSILON,
        )
    )


    return {

        "rule_name":
            rule_name,

        "margin_threshold":
            margin_threshold,

        "uncertainty_threshold":
            uncertainty_threshold,

        "utilization_threshold":
            utilization_threshold,

        "n_triggered":
            n_triggered,

        "trigger_rate_percent":
            trigger_rate,

        "regret_capture_percent":
            regret_capture,

        "positive_regret_capture_percent":
            positive_regret_capture,

        "exact_miss_capture_percent":
            exact_miss_capture,

        "positive_regret_precision_percent":
            positive_regret_precision,

        "exact_miss_precision_percent":
            exact_miss_precision,

        "positive_regret_lift":
            positive_regret_lift,

        "exact_miss_lift":
            exact_miss_lift,

        "triggered_mean_regret":
            triggered_mean_regret,

        "nontriggered_mean_regret":
            nontriggered_mean_regret,

        "regret_concentration":
            regret_concentration,
    }


# ============================================================
# DATA-DRIVEN THRESHOLD QUANTILES
# ============================================================

margin_quantiles = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]


uncertainty_quantiles = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


utilization_quantiles = [
    0.80,
    0.85,
    0.90,
    0.95,
]


grid_records = []


# ============================================================
# MARGIN ONLY
# ============================================================

for q_margin in margin_quantiles:


    margin_threshold = float(
        df[
            "predicted_quality_margin"
        ]
        .quantile(
            q_margin
        )
    )


    trigger = (
        df[
            "predicted_quality_margin"
        ]
        <=
        margin_threshold
    )


    grid_records.append(
        evaluate_trigger(

            trigger_mask=trigger,

            rule_name=(
                f"margin_only_q{q_margin:.2f}"
            ),

            margin_threshold=(
                margin_threshold
            ),
        )
    )


# ============================================================
# TREE STD ONLY
# ============================================================

for q_std in uncertainty_quantiles:


    threshold = float(
        df[
            "selected_tree_std"
        ]
        .quantile(
            q_std
        )
    )


    trigger = (
        df[
            "selected_tree_std"
        ]
        >=
        threshold
    )


    grid_records.append(
        evaluate_trigger(

            trigger_mask=trigger,

            rule_name=(
                f"tree_std_only_q{q_std:.2f}"
            ),

            uncertainty_threshold=(
                threshold
            ),
        )
    )


# ============================================================
# MARGIN OR TREE STD
# ============================================================

for q_margin in margin_quantiles:


    margin_threshold = float(
        df[
            "predicted_quality_margin"
        ]
        .quantile(
            q_margin
        )
    )


    for q_std in uncertainty_quantiles:


        std_threshold = float(
            df[
                "selected_tree_std"
            ]
            .quantile(
                q_std
            )
        )


        trigger = (

            (
                df[
                    "predicted_quality_margin"
                ]
                <=
                margin_threshold
            )

            |

            (
                df[
                    "selected_tree_std"
                ]
                >=
                std_threshold
            )
        )


        grid_records.append(
            evaluate_trigger(

                trigger_mask=trigger,

                rule_name=(
                    "margin_or_tree_std"
                ),

                margin_threshold=(
                    margin_threshold
                ),

                uncertainty_threshold=(
                    std_threshold
                ),
            )
        )


# ============================================================
# ADD UTILIZATION
# ============================================================

for q_margin in [
    0.10,
    0.15,
    0.20,
]:


    margin_threshold = float(
        df[
            "predicted_quality_margin"
        ]
        .quantile(
            q_margin
        )
    )


    for q_std in [
        0.80,
        0.85,
        0.90,
    ]:


        std_threshold = float(
            df[
                "selected_tree_std"
            ]
            .quantile(
                q_std
            )
        )


        for q_util in utilization_quantiles:


            utilization_threshold = float(
                df[
                    "selected_utilization"
                ]
                .quantile(
                    q_util
                )
            )


            trigger = (

                (
                    df[
                        "predicted_quality_margin"
                    ]
                    <=
                    margin_threshold
                )

                |

                (
                    df[
                        "selected_tree_std"
                    ]
                    >=
                    std_threshold
                )

                |

                (
                    df[
                        "selected_utilization"
                    ]
                    >=
                    utilization_threshold
                )
            )


            grid_records.append(
                evaluate_trigger(

                    trigger_mask=trigger,

                    rule_name=(
                        "margin_or_tree_std_or_utilization"
                    ),

                    margin_threshold=(
                        margin_threshold
                    ),

                    uncertainty_threshold=(
                        std_threshold
                    ),

                    utilization_threshold=(
                        utilization_threshold
                    ),
                )
            )


# ============================================================
# GRID DATAFRAME
# ============================================================

grid_df = pd.DataFrame(
    grid_records
)


# ============================================================
# LOW-FREQUENCY RULES
# ============================================================

feasible_df = grid_df[
    grid_df[
        "trigger_rate_percent"
    ]
    <=
    MAX_TRIGGER_RATE_PERCENT
].copy()


if len(
    feasible_df
) == 0:

    raise RuntimeError(
        "\nNo rule satisfies the selective-BO trigger-rate "
        "constraint."
    )


# ============================================================
# PARETO ANALYSIS
# ============================================================
#
# Desired:
#
#   lower trigger rate
#   higher regret capture
#   higher exact-miss capture
# ============================================================

pareto_indices = []


for idx, row in feasible_df.iterrows():


    dominated = False


    for other_idx, other in feasible_df.iterrows():


        if idx == other_idx:
            continue


        no_more_cost = (
            other[
                "trigger_rate_percent"
            ]
            <=
            row[
                "trigger_rate_percent"
            ]
        )


        no_worse_regret = (
            other[
                "regret_capture_percent"
            ]
            >=
            row[
                "regret_capture_percent"
            ]
        )


        no_worse_miss = (
            other[
                "exact_miss_capture_percent"
            ]
            >=
            row[
                "exact_miss_capture_percent"
            ]
        )


        strictly_better = (

            other[
                "trigger_rate_percent"
            ]
            <
            row[
                "trigger_rate_percent"
            ]

            or

            other[
                "regret_capture_percent"
            ]
            >
            row[
                "regret_capture_percent"
            ]

            or

            other[
                "exact_miss_capture_percent"
            ]
            >
            row[
                "exact_miss_capture_percent"
            ]
        )


        if (
            no_more_cost
            and
            no_worse_regret
            and
            no_worse_miss
            and
            strictly_better
        ):

            dominated = True

            break


    if not dominated:

        pareto_indices.append(
            idx
        )


pareto_df = (
    feasible_df
    .loc[
        pareto_indices
    ]
    .copy()
)


# ============================================================
# PROVISIONAL BEST RULE
# ============================================================
#
# Transparent lexicographic rule:
#
# 1. maximize TOTAL REGRET CAPTURE
# 2. maximize exact-best-miss capture
# 3. minimize trigger frequency
#
# No invented weighted score.
# ============================================================

ranked_df = feasible_df.sort_values(

    by=[
        "regret_capture_percent",
        "exact_miss_capture_percent",
        "trigger_rate_percent",
    ],

    ascending=[
        False,
        False,
        True,
    ],
)


best_rule = ranked_df.iloc[
    0
]


# ============================================================
# SAVE LABEL DATA
# ============================================================

df.to_csv(
    DECISION_OUTPUT,
    index=False,
)


os.makedirs(
    "results/tables",
    exist_ok=True,
)


grid_df.to_csv(
    GRID_OUTPUT,
    index=False,
)


pareto_df.to_csv(
    PARETO_OUTPUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    {

        "metric": [

            "n_decisions",

            "n_positive_regret_decisions",

            "n_exact_best_misses",

            "total_accumulated_regret",

            "base_positive_regret_rate_percent",

            "base_exact_miss_rate_percent",

            "maximum_trigger_rate_percent",

            "selected_trigger_rate_percent",

            "selected_regret_capture_percent",

            "selected_positive_regret_capture_percent",

            "selected_exact_miss_capture_percent",

            "selected_positive_regret_precision_percent",

            "selected_exact_miss_precision_percent",

            "selected_positive_regret_lift",

            "selected_exact_miss_lift",

            "selected_regret_concentration",

            "selected_margin_threshold",

            "selected_tree_std_threshold",

            "selected_utilization_threshold",
        ],

        "value": [

            n_decisions,

            n_positive_regret,

            n_exact_misses,

            total_regret,

            base_positive_regret_rate,

            base_exact_miss_rate,

            MAX_TRIGGER_RATE_PERCENT,

            best_rule[
                "trigger_rate_percent"
            ],

            best_rule[
                "regret_capture_percent"
            ],

            best_rule[
                "positive_regret_capture_percent"
            ],

            best_rule[
                "exact_miss_capture_percent"
            ],

            best_rule[
                "positive_regret_precision_percent"
            ],

            best_rule[
                "exact_miss_precision_percent"
            ],

            best_rule[
                "positive_regret_lift"
            ],

            best_rule[
                "exact_miss_lift"
            ],

            best_rule[
                "regret_concentration"
            ],

            best_rule[
                "margin_threshold"
            ],

            best_rule[
                "uncertainty_threshold"
            ],

            best_rule[
                "utilization_threshold"
            ],
        ],
    }
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
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
    "CORRECTED SELECTIVE-BO CALIBRATION RESULT"
)

print(
    "============================================================"
)


print(
    f"\nRule:"
)

print(
    best_rule[
        "rule_name"
    ]
)


print(
    f"\nTrigger rate                    : "
    f"{best_rule['trigger_rate_percent']:.2f}%"
)


print(
    f"TOTAL REGRET CAPTURE            : "
    f"{best_rule['regret_capture_percent']:.2f}%"
)


print(
    f"Positive-regret decision capture: "
    f"{best_rule['positive_regret_capture_percent']:.2f}%"
)


print(
    f"Exact-best-miss capture         : "
    f"{best_rule['exact_miss_capture_percent']:.2f}%"
)


print(
    f"\nPositive-regret precision       : "
    f"{best_rule['positive_regret_precision_percent']:.2f}%"
)


print(
    f"Exact-best-miss precision       : "
    f"{best_rule['exact_miss_precision_percent']:.2f}%"
)


print(
    f"\nPositive-regret lift            : "
    f"{best_rule['positive_regret_lift']:.2f}x"
)


print(
    f"Exact-best-miss lift            : "
    f"{best_rule['exact_miss_lift']:.2f}x"
)


print(
    f"Regret concentration            : "
    f"{best_rule['regret_concentration']:.2f}x"
)


print(
    "\nThresholds:"
)


print(
    f"Margin                          : "
    f"{best_rule['margin_threshold']}"
)


print(
    f"Tree disagreement               : "
    f"{best_rule['uncertainty_threshold']}"
)


print(
    f"Utilization                     : "
    f"{best_rule['utilization_threshold']}"
)


print(
    "\n"
    "------------------------------------------------------------"
)

print(
    "TOP PARETO RULES"
)

print(
    "------------------------------------------------------------\n"
)


columns_to_show = [

    "rule_name",

    "trigger_rate_percent",

    "regret_capture_percent",

    "positive_regret_capture_percent",

    "exact_miss_capture_percent",

    "exact_miss_precision_percent",

    "exact_miss_lift",

    "regret_concentration",

    "margin_threshold",

    "uncertainty_threshold",

    "utilization_threshold",
]


print(
    pareto_df[
        columns_to_show
    ]
    .sort_values(
        [
            "regret_capture_percent",
            "exact_miss_capture_percent",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(
        15
    )
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL DECISION
# ============================================================

trigger_rate = float(
    best_rule[
        "trigger_rate_percent"
    ]
)


regret_capture = float(
    best_rule[
        "regret_capture_percent"
    ]
)


miss_capture = float(
    best_rule[
        "exact_miss_capture_percent"
    ]
)


print(
    "\n"
    "============================================================"
)

print(
    "CORRECTED SELECTIVE-BO VERDICT"
)

print(
    "============================================================"
)


if (
    trigger_rate
    <=
    20.0
    and
    regret_capture
    >=
    60.0
):


    print(
        "\nRESULT A:"
    )


    print(
        "A low-frequency online trigger identifies a large "
        "share of total decision regret."
    )


    print(
        "\nSelective BO is strongly justified for a controlled "
        "100-assembly validation."
    )


elif (
    trigger_rate
    <=
    20.0
    and
    (
        regret_capture
        >=
        40.0
        or
        miss_capture
        >=
        50.0
    )
):


    print(
        "\nRESULT B:"
    )


    print(
        "The trigger provides useful but moderate "
        "discrimination."
    )


    print(
        "\nOne controlled selective-BO validation is justified, "
        "but it should not yet be treated as the final "
        "architecture."
    )


else:


    print(
        "\nRESULT C:"
    )


    print(
        "The online signals do not concentrate enough "
        "decision loss to justify another selective-BO "
        "controller experiment."
    )


    print(
        "\nFreeze Structured-20 as the operational controller "
        "and move directly to robustness validation."
    )


print(
    "\nIMPORTANT:"
)


print(
    "This corrected analysis replaces the previous binary "
    "high-regret calibration because the previous P90 regret "
    "threshold collapsed to zero."
)


print(
    "\nSaved:"
)


print(
    GRID_OUTPUT
)

print(
    PARETO_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print(
    DECISION_OUTPUT
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 CORRECTED SELECTIVE-BO CALIBRATION COMPLETED"
)

print(
    "============================================================"
)