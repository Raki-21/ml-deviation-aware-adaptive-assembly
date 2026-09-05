"""
Generate final thesis-quality figures for V3.3.

This script only reads already validated V3.3 CSV outputs.

It does NOT:
- retrain models,
- rerun experiments,
- tune controllers,
- modify frozen results.

Outputs are written to:
06_v3_3_nonlinear_hybrid_upgrade/figures/final_thesis_figures
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

FIGURE_DIR = V33_ROOT / "figures" / "final_thesis_figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

FINAL_ASSEMBLY_FILE = (
    FINAL_DIR
    / "v33_final_independent_assembly_results.csv"
)

PAIRED_FILE = (
    FINAL_DIR
    / "v33_final_independent_paired_results.csv"
)

STAGE_FILE = (
    FINAL_DIR
    / "v33_final_independent_stage_statistics.csv"
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


required_files = [
    FINAL_ASSEMBLY_FILE,
    PAIRED_FILE,
    STAGE_FILE,
    ROBUSTNESS_FILE,
    SCALING_FILE,
    REFERENCE_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required validated result file missing:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

final_df = pd.read_csv(
    FINAL_ASSEMBLY_FILE
)

paired_df = pd.read_csv(
    PAIRED_FILE
)

stage_df = pd.read_csv(
    STAGE_FILE
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


# ============================================================
# GENERAL SETTINGS
# ============================================================

DPI = 300


def save_figure(filename):

    path = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# FIGURE 1
# FINAL INDEPENDENT METHOD DISTRIBUTION
# ============================================================

method_order = [
    "LSQ",
    "LSQ_PLUS_ML_ALL",
    "DIRECT_NONLINEAR_REFERENCE",
]

method_labels = [
    "Deterministic LSQ",
    "Locked Hybrid",
    "Direct Numerical\nReference",
]


data = []


for method in method_order:

    values = (
        final_df[
            final_df[
                "method"
            ]
            ==
            method
        ][
            "final_quality"
        ]
        .to_numpy(
            dtype=float
        )
    )

    data.append(
        values
    )


plt.figure(
    figsize=(7.5, 5.0)
)

plt.boxplot(
    data,
    tick_labels=method_labels,
    showfliers=False,
)

plt.ylabel(
    "Final assembly quality score"
)

plt.title(
    "Final Independent Validation: Quality Distribution"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

save_figure(
    "v33_final_quality_distribution.png"
)


# ============================================================
# FIGURE 2
# PAIRED LSQ VS HYBRID
# ============================================================

lsq = paired_df[
    "lsq_final_quality"
].to_numpy(
    dtype=float
)


hybrid = paired_df[
    "hybrid_final_quality"
].to_numpy(
    dtype=float
)


minimum = float(
    min(
        np.min(
            lsq
        ),
        np.min(
            hybrid
        ),
    )
)


maximum = float(
    max(
        np.max(
            lsq
        ),
        np.max(
            hybrid
        ),
    )
)


plt.figure(
    figsize=(6.2, 6.0)
)

plt.scatter(
    lsq,
    hybrid,
    s=20,
    alpha=0.65,
)

plt.plot(
    [
        minimum,
        maximum,
    ],
    [
        minimum,
        maximum,
    ],
    linestyle="--",
)

plt.xlabel(
    "Deterministic LSQ final quality"
)

plt.ylabel(
    "Locked hybrid final quality"
)

plt.title(
    "Assembly-Level Paired Comparison"
)

plt.grid(
    alpha=0.25,
)

save_figure(
    "v33_paired_lsq_vs_hybrid.png"
)


# ============================================================
# FIGURE 3
# PAIRED IMPROVEMENT DISTRIBUTION
# ============================================================

differences = paired_df[
    "hybrid_minus_lsq"
].to_numpy(
    dtype=float
)


plt.figure(
    figsize=(7.5, 5.0)
)

plt.hist(
    differences,
    bins=25,
    edgecolor="black",
)

plt.axvline(
    0.0,
    linestyle="--",
)

plt.axvline(
    np.mean(
        differences
    ),
    linestyle="-",
)

plt.xlabel(
    "Hybrid - LSQ final quality"
)

plt.ylabel(
    "Number of assemblies"
)

plt.title(
    "Distribution of Paired Hybrid Improvement"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

save_figure(
    "v33_paired_improvement_distribution.png"
)


# ============================================================
# FIGURE 4
# STAGE-WISE SEQUENTIAL QUALITY
# ============================================================

stages = stage_df[
    "component_index"
].to_numpy(
    dtype=int
)


lsq_stage = stage_df[
    "mean_lsq_quality"
].to_numpy(
    dtype=float
)


hybrid_stage = stage_df[
    "mean_hybrid_quality"
].to_numpy(
    dtype=float
)


plt.figure(
    figsize=(8.0, 5.2)
)

plt.plot(
    stages,
    lsq_stage,
    marker="o",
    label="Deterministic LSQ",
)

plt.plot(
    stages,
    hybrid_stage,
    marker="o",
    label="Locked Hybrid",
)

plt.xticks(
    stages
)

plt.xlabel(
    "Sequential component index"
)

plt.ylabel(
    "Mean assembly quality score"
)

plt.title(
    "Stage-Wise Quality Evolution for K = 10"
)

plt.legend()

plt.grid(
    alpha=0.25,
)

save_figure(
    "v33_stagewise_quality_K10.png"
)


# ============================================================
# FIGURE 5
# ROBUSTNESS REGIME HEATMAP
# ============================================================

alpha_levels = sorted(
    robustness_df[
        "nonlinearity_strength"
    ]
    .unique()
)


noise_levels = sorted(
    robustness_df[
        "measurement_noise_mm"
    ]
    .unique()
)


heatmap = np.zeros(
    (
        len(
            alpha_levels
        ),
        len(
            noise_levels
        ),
    )
)


for i, alpha in enumerate(
    alpha_levels
):

    for j, noise in enumerate(
        noise_levels
    ):

        row = robustness_df[
            (
                robustness_df[
                    "nonlinearity_strength"
                ]
                ==
                alpha
            )
            &
            (
                robustness_df[
                    "measurement_noise_mm"
                ]
                ==
                noise
            )
        ]

        heatmap[
            i,
            j
        ] = float(
            row[
                "relative_hybrid_improvement_percent"
            ]
            .iloc[0]
        )


plt.figure(
    figsize=(7.2, 5.2)
)

image = plt.imshow(
    heatmap,
    aspect="auto",
)


plt.colorbar(
    image,
    label="Hybrid improvement over LSQ (%)",
)


plt.xticks(
    range(
        len(
            noise_levels
        )
    ),
    [
        f"{value:.2f}"
        for value in noise_levels
    ],
)


plt.yticks(
    range(
        len(
            alpha_levels
        )
    ),
    [
        f"{value:.1f}"
        for value in alpha_levels
    ],
)


for i in range(
    len(
        alpha_levels
    )
):

    for j in range(
        len(
            noise_levels
        )
    ):

        plt.text(
            j,
            i,
            f"{heatmap[i, j]:.2f}%",
            ha="center",
            va="center",
        )


plt.xlabel(
    "Measurement noise standard deviation (mm)"
)

plt.ylabel(
    "Nonlinearity strength"
)

plt.title(
    "Locked-Hybrid Robustness Regime Map"
)

save_figure(
    "v33_robustness_regime_heatmap.png"
)


# ============================================================
# FIGURE 6
# SEQUENCE-LENGTH SCALING
# ============================================================

scaling_df = scaling_df.sort_values(
    "sequence_length"
)


sequence_lengths = scaling_df[
    "sequence_length"
].to_numpy(
    dtype=int
)


lsq_scaling = scaling_df[
    "mean_lsq_quality"
].to_numpy(
    dtype=float
)


hybrid_scaling = scaling_df[
    "mean_hybrid_quality"
].to_numpy(
    dtype=float
)


x = np.arange(
    len(
        sequence_lengths
    )
)


width = 0.35


plt.figure(
    figsize=(7.2, 5.0)
)


plt.bar(
    x
    -
    width
    /
    2,
    lsq_scaling,
    width,
    label="Deterministic LSQ",
)


plt.bar(
    x
    +
    width
    /
    2,
    hybrid_scaling,
    width,
    label="Locked Hybrid",
)


plt.xticks(
    x,
    [
        f"K = {value}"
        for value in sequence_lengths
    ],
)


plt.ylabel(
    "Mean final assembly quality"
)

plt.title(
    "Sequence-Length Scaling"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25,
)

save_figure(
    "v33_sequence_length_scaling.png"
)


# ============================================================
# FIGURE 7
# DIRECT REFERENCE STABILITY
# ============================================================

budget_order = [
    "LOW_8x5",
    "MEDIUM_18x7",
    "HIGH_30x10",
]


budget_labels = [
    "Low",
    "Medium",
    "High",
]


budget_values = []


for budget in budget_order:

    row = reference_df[
        reference_df[
            "budget"
        ]
        ==
        budget
    ]


    if row.empty:

        raise RuntimeError(
            f"Reference budget missing: {budget}"
        )


    budget_values.append(
        float(
            row[
                "mean_final_quality"
            ]
            .iloc[0]
        )
    )


plt.figure(
    figsize=(7.0, 5.0)
)


plt.plot(
    budget_labels,
    budget_values,
    marker="o",
)


for label, value in zip(
    budget_labels,
    budget_values,
):

    plt.annotate(
        f"{value:.3f}",
        (
            label,
            value,
        ),
        xytext=(
            0,
            8,
        ),
        textcoords="offset points",
        ha="center",
    )


plt.ylabel(
    "Mean final assembly quality"
)

plt.xlabel(
    "Numerical-reference search budget"
)

plt.title(
    "Direct Nonlinear Reference Stability"
)

plt.grid(
    alpha=0.25,
)

save_figure(
    "v33_direct_reference_stability.png"
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    "============================================================"
)

print(
    "V3.3 FINAL THESIS FIGURE GENERATION COMPLETED"
)

print(
    "============================================================"
)

print(
    f"\nFigure folder:\n{FIGURE_DIR}"
)

print(
    "\nGenerated figures:"
)

for path in sorted(
    FIGURE_DIR.glob(
        "*.png"
    )
):

    print(
        f"  {path.name}"
    )