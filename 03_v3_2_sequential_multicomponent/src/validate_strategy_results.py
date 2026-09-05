import os
import pandas as pd
import numpy as np


# ============================================================
# VERSION 3.2 - STRATEGY RESULT VALIDATION
# ============================================================

ASSEMBLY_FILE = (
    "data/processed/"
    "v3_2_strategy_assembly_results.csv"
)

COMPONENT_FILE = (
    "data/processed/"
    "v3_2_strategy_component_results.csv"
)

SUMMARY_FILE = (
    "results/tables/"
    "v3_2_strategy_summary.csv"
)


# ------------------------------------------------------------
# FILE CHECK
# ------------------------------------------------------------

required_files = [
    ASSEMBLY_FILE,
    COMPONENT_FILE,
    SUMMARY_FILE,
]

missing = [
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing:

    print("\nMissing required files:")

    for file in missing:
        print(file)

    raise SystemExit(
        "\nRun strategy_comparison_v3_2.py first."
    )


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

assembly_df = pd.read_csv(
    ASSEMBLY_FILE
)

component_df = pd.read_csv(
    COMPONENT_FILE
)

summary_df = pd.read_csv(
    SUMMARY_FILE
)


print(
    "\n"
    "============================================================"
)

print(
    "V3.2 STRATEGY RESULT VALIDATION"
)

print(
    "============================================================"
)


# ============================================================
# 1. BASIC DATA CHECK
# ============================================================

print(
    "\n1. BASIC DATA CHECK"
)

print(
    "------------------------------------------------------------"
)

print(
    f"Assemblies evaluated : {len(assembly_df)}"
)

print(
    f"Component decisions  : {len(component_df)}"
)

missing_values = (
    assembly_df.isna().sum().sum()
    +
    component_df.isna().sum().sum()
)

print(
    f"Missing values       : {missing_values}"
)

if missing_values == 0:

    print(
        "PASS - No missing strategy-result values."
    )

else:

    print(
        "WARNING - Missing values detected."
    )


# ============================================================
# 2. FINAL QUALITY SUMMARY
# ============================================================

print(
    "\n2. FINAL QUALITY SUMMARY"
)

print(
    "------------------------------------------------------------"
)


quality_columns = {
    "global":
        "global_final_quality",

    "batch":
        "batch_final_quality",

    "individual":
        "individual_final_quality",
}


for name, column in quality_columns.items():

    print(
        f"\n{name.upper()}"
    )

    print(
        assembly_df[
            column
        ]
        .describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(4)
    )


# ============================================================
# 3. STRATEGY ORDER CHECK
# ============================================================

print(
    "\n3. STRATEGY ORDER CHECK"
)

print(
    "------------------------------------------------------------"
)


global_mean = float(
    assembly_df[
        "global_final_quality"
    ].mean()
)

batch_mean = float(
    assembly_df[
        "batch_final_quality"
    ].mean()
)

individual_mean = float(
    assembly_df[
        "individual_final_quality"
    ].mean()
)


print(
    f"Global mean quality     : {global_mean:.4f}"
)

print(
    f"Batch mean quality      : {batch_mean:.4f}"
)

print(
    f"Individual mean quality : {individual_mean:.4f}"
)


if (
    individual_mean
    < global_mean
):

    print(
        "\nPASS - Individual strategy improves over global."
    )

else:

    print(
        "\nWARNING - Individual strategy does not improve over global."
    )


if (
    batch_mean
    < global_mean
):

    print(
        "PASS - Batch strategy improves over global."
    )

else:

    print(
        "NOTE - Batch strategy does not improve over global."
    )


# ============================================================
# 4. ASSEMBLY-BY-ASSEMBLY IMPROVEMENT
# ============================================================

print(
    "\n4. ASSEMBLY-BY-ASSEMBLY IMPROVEMENT"
)

print(
    "------------------------------------------------------------"
)


assembly_df[
    "individual_better_than_global"
] = (
    assembly_df[
        "individual_final_quality"
    ]
    <
    assembly_df[
        "global_final_quality"
    ]
)


assembly_df[
    "batch_better_than_global"
] = (
    assembly_df[
        "batch_final_quality"
    ]
    <
    assembly_df[
        "global_final_quality"
    ]
)


individual_win_rate = (
    assembly_df[
        "individual_better_than_global"
    ].mean()
    * 100.0
)


batch_win_rate = (
    assembly_df[
        "batch_better_than_global"
    ].mean()
    * 100.0
)


print(
    f"Individual better than global : "
    f"{individual_win_rate:.2f}%"
)

print(
    f"Batch better than global      : "
    f"{batch_win_rate:.2f}%"
)


# ============================================================
# 5. QUALITY BY BATCH CONDITION
# ============================================================

print(
    "\n5. QUALITY BY BATCH CONDITION"
)

print(
    "------------------------------------------------------------"
)


batch_summary = (
    assembly_df
    .groupby(
        "batch_condition"
    )
    .agg(
        global_mean=(
            "global_final_quality",
            "mean",
        ),

        batch_mean=(
            "batch_final_quality",
            "mean",
        ),

        individual_mean=(
            "individual_final_quality",
            "mean",
        ),
    )
)


batch_summary[
    "individual_improvement_percent"
] = (
    (
        batch_summary[
            "global_mean"
        ]
        -
        batch_summary[
            "individual_mean"
        ]
    )
    /
    batch_summary[
        "global_mean"
    ]
    * 100.0
)


print(
    batch_summary.round(4)
)


# ============================================================
# 6. PARALLELISM COMPARISON
# ============================================================

print(
    "\n6. PARALLELISM COMPARISON"
)

print(
    "------------------------------------------------------------"
)


parallelism_summary = pd.DataFrame(
    {
        "strategy": [
            "global",
            "batch",
            "individual",
        ],

        "mean_parallelism": [
            assembly_df[
                "global_final_parallelism"
            ].mean(),

            assembly_df[
                "batch_final_parallelism"
            ].mean(),

            assembly_df[
                "individual_final_parallelism"
            ].mean(),
        ],

        "p95_parallelism": [
            assembly_df[
                "global_final_parallelism"
            ].quantile(0.95),

            assembly_df[
                "batch_final_parallelism"
            ].quantile(0.95),

            assembly_df[
                "individual_final_parallelism"
            ].quantile(0.95),
        ],
    }
)


print(
    parallelism_summary.round(4)
)


# ============================================================
# 7. CORRECTION UTILIZATION CHECK
# ============================================================

print(
    "\n7. CORRECTION UTILIZATION"
)

print(
    "------------------------------------------------------------"
)


utilization = (
    component_df[
        "correction_utilization"
    ]
)


print(
    utilization
    .describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.99,
        ]
    )
    .round(4)
)


near_limit = (
    utilization
    >= 0.90
).mean() * 100.0


at_limit = (
    utilization
    >= 0.999
).mean() * 100.0


print(
    f"\nCorrections >= 90% capability : "
    f"{near_limit:.2f}%"
)

print(
    f"Corrections at limit          : "
    f"{at_limit:.2f}%"
)


# ============================================================
# 8. COMPONENT-STAGE QUALITY
# ============================================================

print(
    "\n8. QUALITY THROUGH ASSEMBLY SEQUENCE"
)

print(
    "------------------------------------------------------------"
)


stage_summary = (
    component_df
    .groupby(
        "component_index"
    )
    .agg(
        global_quality=(
            "global_quality",
            "mean",
        ),

        batch_quality=(
            "batch_quality",
            "mean",
        ),

        individual_quality=(
            "individual_quality",
            "mean",
        ),
    )
)


print(
    stage_summary.round(4)
)


# ============================================================
# 9. SAVE VALIDATION TABLES
# ============================================================

os.makedirs(
    "results/tables",
    exist_ok=True,
)


batch_summary.to_csv(
    "results/tables/"
    "v3_2_strategy_by_batch_condition.csv"
)


parallelism_summary.to_csv(
    "results/tables/"
    "v3_2_strategy_parallelism_summary.csv",
    index=False,
)


stage_summary.to_csv(
    "results/tables/"
    "v3_2_strategy_quality_by_stage.csv"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "STRATEGY RESULT VALIDATION COMPLETED"
)

print(
    "============================================================"
)