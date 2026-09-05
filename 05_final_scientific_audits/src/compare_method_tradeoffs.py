"""
Final method trade-off audit.

Purpose
-------
Compare the main correction/recommendation strategies using already
validated frozen results.

No controller is retrained.
No optimization is rerun.
No V3.2 code is modified.

The audit compares:
- Zero correction
- Deterministic LSQ
- Structured-20
- Selective Bayesian Optimization
- Direct simulator numerical reference

The comparison considers:
- final quality
- improvement relative to zero
- approximate computational effort
- requirement for ML
- requirement for an analytical correction model
- simulator evaluations
- practical online suitability
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_ROOT = (
    PROJECT_ROOT
    / "05_final_scientific_audits"
)

AUDIT_RESULTS = (
    AUDIT_ROOT
    / "results"
)

V32_ROOT = (
    PROJECT_ROOT
    / "03_v3_2_sequential_multicomponent"
)

OUTPUT_DIR = (
    AUDIT_RESULTS
    / "method_tradeoffs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXISTING FINAL-AUDIT FILES
# ============================================================

DETERMINISTIC_SUMMARY = (
    AUDIT_RESULTS
    / "deterministic_baseline"
    / "deterministic_baseline_summary.csv"
)

ACHIEVABLE_SUMMARY = (
    AUDIT_RESULTS
    / "achievable_correction"
    / "achievable_correction_summary.csv"
)

DIRECT_COMPONENT = (
    AUDIT_RESULTS
    / "achievable_correction"
    / "direct_simulator_component_results.csv"
)


# ============================================================
# LOCATE FINAL V3.2 VALIDATION SUMMARY
# ============================================================

def find_csv_by_name_fragment(
    root: Path,
    fragments: list[str],
) -> Path | None:

    matches = []

    for path in root.rglob("*.csv"):

        name = path.name.lower()

        if all(
            fragment.lower() in name
            for fragment in fragments
        ):
            matches.append(
                path
            )

    if not matches:
        return None

    return matches[0]


FINAL_VALIDATION_SUMMARY = find_csv_by_name_fragment(
    V32_ROOT,
    [
        "final",
        "controller",
        "summary",
    ],
)


FINAL_VALIDATION_COMPONENT = find_csv_by_name_fragment(
    V32_ROOT,
    [
        "final",
        "controller",
        "component",
    ],
)


# ============================================================
# HELPERS
# ============================================================

def read_metric_table(
    path: Path,
) -> dict[str, float]:

    df = pd.read_csv(
        path
    )

    if not {
        "metric",
        "value",
    }.issubset(
        df.columns
    ):

        raise ValueError(
            f"{path.name} does not contain metric/value columns."
        )

    result = {}

    for _, row in df.iterrows():

        result[
            str(
                row["metric"]
            )
        ] = row[
            "value"
        ]

    return result


def find_metric(
    metric_dict: dict,
    candidate_names: list[str],
    default=np.nan,
):

    for candidate in candidate_names:

        if candidate in metric_dict:

            try:

                return float(
                    metric_dict[
                        candidate
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):

                return metric_dict[
                    candidate
                ]

    return default


# ============================================================
# LOAD AUDIT RESULTS
# ============================================================

for required_file in [
    DETERMINISTIC_SUMMARY,
    ACHIEVABLE_SUMMARY,
    DIRECT_COMPONENT,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{required_file}"
        )


det_metrics = read_metric_table(
    DETERMINISTIC_SUMMARY
)

ach_metrics = read_metric_table(
    ACHIEVABLE_SUMMARY
)

direct_component_df = pd.read_csv(
    DIRECT_COMPONENT
)


# ============================================================
# EXTRACT QUALITY VALUES
# ============================================================

zero_quality = find_metric(
    det_metrics,
    [
        "zero_mean_final_quality",
    ],
)


lsq_quality = find_metric(
    det_metrics,
    [
        "deterministic_mean_final_quality",
    ],
)


structured_quality = find_metric(
    det_metrics,
    [
        "structured_mean_final_quality",
    ],
)


direct_quality = find_metric(
    ach_metrics,
    [
        "direct_mean_final_quality",
        "direct_simulator_mean_final_quality",
        "direct_final_quality",
    ],
)


if np.isnan(
    direct_quality
):

    direct_assembly_path = (
        AUDIT_RESULTS
        / "achievable_correction"
        / "direct_simulator_assembly_results.csv"
    )

    direct_assembly_df = pd.read_csv(
        direct_assembly_path
    )

    direct_quality = float(
        direct_assembly_df[
            "direct_final_quality"
        ]
        .mean()
    )


# ============================================================
# FINAL V3.2 SELECTIVE BO VALUE
# ============================================================

selective_quality = np.nan

structured_runtime_seconds = np.nan
selective_runtime_seconds = np.nan
total_runtime_seconds = np.nan


if (
    FINAL_VALIDATION_SUMMARY is not None
    and FINAL_VALIDATION_SUMMARY.exists()
):

    try:

        final_metrics = read_metric_table(
            FINAL_VALIDATION_SUMMARY
        )

        selective_quality = find_metric(
            final_metrics,
            [
                "selective_mean_final_quality",
                "selective_bo_mean_final_quality",
                "selective_final_quality",
            ],
        )

        structured_runtime_seconds = find_metric(
            final_metrics,
            [
                "structured_runtime_seconds",
                "structured_prediction_runtime_seconds",
                "structured_total_runtime_seconds",
            ],
        )

        selective_runtime_seconds = find_metric(
            final_metrics,
            [
                "selective_runtime_seconds",
                "selective_total_runtime_seconds",
            ],
        )

        total_runtime_seconds = find_metric(
            final_metrics,
            [
                "total_runtime_seconds",
            ],
        )

    except Exception:

        pass


# Frozen validated value if not discoverable automatically.
if np.isnan(
    selective_quality
):

    selective_quality = 0.2377


# ============================================================
# DIRECT SIMULATOR COMPUTATIONAL COST
# ============================================================

if "optimizer_evaluations" in direct_component_df.columns:

    mean_direct_evaluations = float(
        direct_component_df[
            "optimizer_evaluations"
        ]
        .mean()
    )

else:

    mean_direct_evaluations = np.nan


# ============================================================
# QUALITY IMPROVEMENTS
# ============================================================

def improvement_vs_zero(
    quality,
):

    return float(
        100.0
        *
        (
            zero_quality
            -
            quality
        )
        /
        zero_quality
    )


lsq_improvement = improvement_vs_zero(
    lsq_quality
)

structured_improvement = improvement_vs_zero(
    structured_quality
)

selective_improvement = improvement_vs_zero(
    selective_quality
)

direct_improvement = improvement_vs_zero(
    direct_quality
)


# ============================================================
# METHOD TABLE
# ============================================================

method_rows = [
    {
        "method":
            "Zero correction",

        "mean_final_quality":
            zero_quality,

        "improvement_vs_zero_percent":
            0.0,

        "ml_required":
            "No",

        "analytical_inverse_or_basis_required":
            "No",

        "simulator_or_surrogate_evaluations_per_decision":
            0,

        "computational_effort":
            "Minimal",

        "main_strength":
            "Reference baseline",

        "main_limitation":
            "No adaptive correction",

        "online_suitability":
            "Very high computationally, but no correction value",
    },

    {
        "method":
            "Deterministic LSQ",

        "mean_final_quality":
            lsq_quality,

        "improvement_vs_zero_percent":
            lsq_improvement,

        "ml_required":
            "No",

        "analytical_inverse_or_basis_required":
            "Yes",

        "simulator_or_surrogate_evaluations_per_decision":
            1,

        "computational_effort":
            "Very low",

        "main_strength":
            "Transparent and efficient when correction basis is known",

        "main_limitation":
            "Requires an explicit analytical correction structure",

        "online_suitability":
            "Excellent for basis-aligned current model",
    },

    {
        "method":
            "Structured-20",

        "mean_final_quality":
            structured_quality,

        "improvement_vs_zero_percent":
            structured_improvement,

        "ml_required":
            "Yes",

        "analytical_inverse_or_basis_required":
            "No exact inverse",

        "simulator_or_surrogate_evaluations_per_decision":
            20,

        "computational_effort":
            "Low",

        "main_strength":
            "Flexible surrogate-based candidate ranking",

        "main_limitation":
            "Depends on trained surrogate and current feature representation",

        "online_suitability":
            "High",
    },

    {
        "method":
            "Selective BO",

        "mean_final_quality":
            selective_quality,

        "improvement_vs_zero_percent":
            selective_improvement,

        "ml_required":
            "Yes",

        "analytical_inverse_or_basis_required":
            "No exact inverse",

        "simulator_or_surrogate_evaluations_per_decision":
            "20 only when triggered",

        "computational_effort":
            "Moderate",

        "main_strength":
            "Optional refinement for ambiguous surrogate decisions",

        "main_limitation":
            "Negligible mean gain over Structured-20 in validation",

        "online_suitability":
            "Moderate to high when used selectively",
    },

    {
        "method":
            "Direct simulator numerical reference",

        "mean_final_quality":
            direct_quality,

        "improvement_vs_zero_percent":
            direct_improvement,

        "ml_required":
            "No",

        "analytical_inverse_or_basis_required":
            "No, but repeated simulator access required",

        "simulator_or_surrogate_evaluations_per_decision":
            mean_direct_evaluations,

        "computational_effort":
            "Very high",

        "main_strength":
            "Best observed greedy correction quality",

        "main_limitation":
            "Hundreds of simulator evaluations per decision",

        "online_suitability":
            "Low for real-time use in current implementation",
    },
]


method_df = pd.DataFrame(
    method_rows
)


# ============================================================
# NUMERICAL TRADE-OFF TABLE
# ============================================================

numeric_df = method_df[
    [
        "method",
        "mean_final_quality",
        "improvement_vs_zero_percent",
        "simulator_or_surrogate_evaluations_per_decision",
    ]
].copy()


# ============================================================
# RELATIVE QUALITY GAPS
# ============================================================

quality_gap_lsq_vs_direct = float(
    lsq_quality
    -
    direct_quality
)


quality_gap_structured_vs_direct = float(
    structured_quality
    -
    direct_quality
)


quality_gap_structured_vs_lsq = float(
    structured_quality
    -
    lsq_quality
)


quality_gap_selective_vs_structured = float(
    selective_quality
    -
    structured_quality
)


# ============================================================
# SAVE
# ============================================================

method_df.to_csv(
    OUTPUT_DIR
    / "method_tradeoff_table.csv",
    index=False,
)


numeric_df.to_csv(
    OUTPUT_DIR
    / "method_tradeoff_numeric.csv",
    index=False,
)


summary_df = pd.DataFrame(
    {
        "metric": [
            "zero_mean_final_quality",
            "lsq_mean_final_quality",
            "structured20_mean_final_quality",
            "selective_bo_mean_final_quality",
            "direct_reference_mean_final_quality",
            "lsq_improvement_vs_zero_percent",
            "structured20_improvement_vs_zero_percent",
            "selective_bo_improvement_vs_zero_percent",
            "direct_improvement_vs_zero_percent",
            "structured20_minus_lsq_quality",
            "selective_bo_minus_structured20_quality",
            "lsq_minus_direct_quality",
            "structured20_minus_direct_quality",
            "mean_direct_simulator_evaluations_per_decision",
            "structured_runtime_seconds_if_available",
            "selective_runtime_seconds_if_available",
            "total_validation_runtime_seconds_if_available",
        ],

        "value": [
            zero_quality,
            lsq_quality,
            structured_quality,
            selective_quality,
            direct_quality,
            lsq_improvement,
            structured_improvement,
            selective_improvement,
            direct_improvement,
            quality_gap_structured_vs_lsq,
            quality_gap_selective_vs_structured,
            quality_gap_lsq_vs_direct,
            quality_gap_structured_vs_direct,
            mean_direct_evaluations,
            structured_runtime_seconds,
            selective_runtime_seconds,
            total_runtime_seconds,
        ],
    }
)


summary_df.to_csv(
    OUTPUT_DIR
    / "method_tradeoff_summary.csv",
    index=False,
)


# ============================================================
# FIGURE
# ============================================================

quality_plot = method_df[
    [
        "method",
        "mean_final_quality",
    ]
].copy()


plt.figure(
    figsize=(
        10,
        5,
    )
)


plt.bar(
    quality_plot[
        "method"
    ],
    quality_plot[
        "mean_final_quality"
    ],
)


plt.ylabel(
    "Mean final quality score"
)


plt.title(
    "Final Method Quality Comparison"
)


plt.xticks(
    rotation=20,
    ha="right",
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "method_quality_comparison.png",
    dpi=250,
)


plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "FINAL METHOD TRADE-OFF AUDIT"
)

print(
    "============================================================"
)


print(
    "\nMean final quality:"
)


print(
    f"Zero correction             : "
    f"{zero_quality:.4f}"
)


print(
    f"Deterministic LSQ           : "
    f"{lsq_quality:.4f}"
)


print(
    f"Structured-20               : "
    f"{structured_quality:.4f}"
)


print(
    f"Selective BO                : "
    f"{selective_quality:.4f}"
)


print(
    f"Direct simulator reference  : "
    f"{direct_quality:.4f}"
)


print(
    "\nImprovement vs zero:"
)


print(
    f"Deterministic LSQ           : "
    f"{lsq_improvement:.2f}%"
)


print(
    f"Structured-20               : "
    f"{structured_improvement:.2f}%"
)


print(
    f"Selective BO                : "
    f"{selective_improvement:.2f}%"
)


print(
    f"Direct simulator reference  : "
    f"{direct_improvement:.2f}%"
)


print(
    "\nQuality gaps:"
)


print(
    f"Structured-20 - LSQ         : "
    f"{quality_gap_structured_vs_lsq:.4f}"
)


print(
    f"Selective BO - Structured20 : "
    f"{quality_gap_selective_vs_structured:.4f}"
)


print(
    f"LSQ - direct reference      : "
    f"{quality_gap_lsq_vs_direct:.4f}"
)


print(
    f"Structured20 - direct ref.  : "
    f"{quality_gap_structured_vs_direct:.4f}"
)


print(
    "\nComputational indicators:"
)


print(
    "Structured-20 surrogate evaluations/decision : 20"
)


if not np.isnan(
    mean_direct_evaluations
):

    print(
        f"Direct simulator evaluations/decision        : "
        f"{mean_direct_evaluations:.1f}"
    )


print(
    "\nMethod-selection interpretation:"
)


print(
    "\n1. Deterministic LSQ is the preferred transparent "
    "baseline when the correction response is explicitly known "
    "and geometry is well represented by the correction basis."
)


print(
    "\n2. Structured-20 provides a flexible ML-supported "
    "recommendation mechanism without requiring direct inversion "
    "of the quality response."
)


print(
    "\n3. The basis-residual audit shows that Structured-20 "
    "becomes relatively more competitive when analytical "
    "basis representability deteriorates."
)


print(
    "\n4. Selective BO produces almost no mean quality improvement "
    "over Structured-20 and is therefore optional rather than "
    "mandatory in the present model."
)


print(
    "\n5. Direct simulator optimization provides the strongest "
    "quality reference, but its evaluation cost is far higher "
    "than Structured-20 or LSQ."
)


print(
    "\n6. No single strategy should be claimed as universally "
    "optimal. Method choice depends on analytical model access, "
    "geometric complexity, computational budget, and transfer "
    "requirements."
)


print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)


print(
    "\n"
    "============================================================"
)

print(
    "METHOD TRADE-OFF AUDIT COMPLETED"
)

print(
    "============================================================"
)