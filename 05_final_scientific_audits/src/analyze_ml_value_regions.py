"""
Analyze where the ML-supported Structured-20 recommendation performs
better or worse than the deterministic least-squares engineering baseline.

This is a post-freeze scientific audit only.
No models are retrained and no controller parameters are changed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_ROOT = PROJECT_ROOT / "05_final_scientific_audits"
RESULTS_ROOT = AUDIT_ROOT / "results"

OUTPUT_DIR = RESULTS_ROOT / "ml_value_regions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Locate existing validated result files
# ---------------------------------------------------------------------

def find_file(filename: str) -> Path:
    matches = list(RESULTS_ROOT.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} anywhere under:\n{RESULTS_ROOT}"
        )

    if len(matches) > 1:
        print(f"Warning: multiple matches found for {filename}.")
        print(f"Using: {matches[0]}")

    return matches[0]


assembly_file = find_file("deterministic_assembly_results.csv")
component_file = find_file("deterministic_component_results.csv")
stage_file = find_file("deterministic_by_stage.csv")
direct_file = find_file("direct_simulator_assembly_results.csv")


print("=" * 78)
print("ML VALUE REGION ANALYSIS")
print("=" * 78)

print("\nInput files:")
print(f"Assembly comparison : {assembly_file}")
print(f"Component comparison: {component_file}")
print(f"Stage comparison    : {stage_file}")
print(f"Direct reference    : {direct_file}")


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

assembly = pd.read_csv(assembly_file)
component = pd.read_csv(component_file)
stage = pd.read_csv(stage_file)
direct = pd.read_csv(direct_file)


# ---------------------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------------------

required_assembly = {
    "assembly_id",
    "zero_final_quality",
    "deterministic_final_quality",
    "structured_final_quality",
}

required_component = {
    "assembly_id",
    "component_index",
    "zero_quality",
    "deterministic_quality",
    "structured_quality",
    "deterministic_utilization",
}

required_direct = {
    "assembly_id",
    "zero_final_quality",
    "direct_final_quality",
    "deterministic_final_quality",
    "structured_final_quality",
}


def check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}"
        )


check_columns(assembly, required_assembly, "deterministic_assembly_results.csv")
check_columns(component, required_component, "deterministic_component_results.csv")
check_columns(direct, required_direct, "direct_simulator_assembly_results.csv")


# ---------------------------------------------------------------------
# Assembly-level comparison
# ---------------------------------------------------------------------

assembly = assembly.copy()

assembly["structured_advantage"] = (
    assembly["deterministic_final_quality"]
    - assembly["structured_final_quality"]
)

assembly["absolute_quality_difference"] = (
    assembly["structured_final_quality"]
    - assembly["deterministic_final_quality"]
)

TIE_TOL = 1e-12

assembly["winner"] = np.select(
    [
        assembly["structured_advantage"] > TIE_TOL,
        assembly["structured_advantage"] < -TIE_TOL,
    ],
    [
        "STRUCTURED20",
        "DETERMINISTIC_LSQ",
    ],
    default="TIE",
)

n_assemblies = len(assembly)

structured_wins = int((assembly["winner"] == "STRUCTURED20").sum())
lsq_wins = int((assembly["winner"] == "DETERMINISTIC_LSQ").sum())
ties = int((assembly["winner"] == "TIE").sum())

structured_win_rate = 100.0 * structured_wins / n_assemblies
lsq_win_rate = 100.0 * lsq_wins / n_assemblies
tie_rate = 100.0 * ties / n_assemblies


# ---------------------------------------------------------------------
# Paired bootstrap confidence interval
# ---------------------------------------------------------------------

def paired_bootstrap_mean_ci(
    differences: np.ndarray,
    n_bootstrap: int = 20000,
    seed: int = 20260830,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = differences[indices].mean()

    mean_difference = float(differences.mean())
    lower = float(np.percentile(bootstrap_means, 2.5))
    upper = float(np.percentile(bootstrap_means, 97.5))

    return mean_difference, lower, upper


mean_advantage, ci_low, ci_high = paired_bootstrap_mean_ci(
    assembly["structured_advantage"].to_numpy()
)


# ---------------------------------------------------------------------
# Initial-quality regions
# ---------------------------------------------------------------------

assembly["initial_quality_quartile"] = pd.qcut(
    assembly["zero_final_quality"],
    q=4,
    labels=[
        "Q1_lowest_initial_error",
        "Q2",
        "Q3",
        "Q4_highest_initial_error",
    ],
    duplicates="drop",
)


def summarize_regions(
    df: pd.DataFrame,
    group_column: str,
    zero_column: str,
    structured_column: str,
    deterministic_column: str,
) -> pd.DataFrame:

    rows = []

    for group_name, group in df.groupby(group_column, observed=False):
        n = len(group)

        if n == 0:
            continue

        structured_better = (
            group[structured_column] < group[deterministic_column]
        ).mean()

        deterministic_better = (
            group[deterministic_column] < group[structured_column]
        ).mean()

        mean_structured = group[structured_column].mean()
        mean_deterministic = group[deterministic_column].mean()
        mean_zero = group[zero_column].mean()

        structured_improvement = (
            100.0 * (mean_zero - mean_structured) / mean_zero
            if mean_zero != 0
            else np.nan
        )

        deterministic_improvement = (
            100.0 * (mean_zero - mean_deterministic) / mean_zero
            if mean_zero != 0
            else np.nan
        )

        rows.append(
            {
                group_column: str(group_name),
                "samples": n,
                "mean_zero_quality": mean_zero,
                "mean_structured_quality": mean_structured,
                "mean_deterministic_quality": mean_deterministic,
                "structured_win_percent": 100.0 * structured_better,
                "deterministic_win_percent": 100.0 * deterministic_better,
                "structured_improvement_vs_zero_percent":
                    structured_improvement,
                "deterministic_improvement_vs_zero_percent":
                    deterministic_improvement,
                "mean_structured_advantage":
                    mean_deterministic - mean_structured,
            }
        )

    return pd.DataFrame(rows)


initial_quality_regions = summarize_regions(
    assembly,
    "initial_quality_quartile",
    "zero_final_quality",
    "structured_final_quality",
    "deterministic_final_quality",
)


# ---------------------------------------------------------------------
# Component / stage analysis
# ---------------------------------------------------------------------

component = component.copy()

component["structured_advantage"] = (
    component["deterministic_quality"]
    - component["structured_quality"]
)

component["winner"] = np.select(
    [
        component["structured_advantage"] > TIE_TOL,
        component["structured_advantage"] < -TIE_TOL,
    ],
    [
        "STRUCTURED20",
        "DETERMINISTIC_LSQ",
    ],
    default="TIE",
)

stage_regions = summarize_regions(
    component,
    "component_index",
    "zero_quality",
    "structured_quality",
    "deterministic_quality",
)


# ---------------------------------------------------------------------
# Deterministic-utilization regions
# ---------------------------------------------------------------------

try:
    component["utilization_quartile"] = pd.qcut(
        component["deterministic_utilization"],
        q=4,
        labels=[
            "U1_lowest",
            "U2",
            "U3",
            "U4_highest",
        ],
        duplicates="drop",
    )

    utilization_regions = summarize_regions(
        component,
        "utilization_quartile",
        "zero_quality",
        "structured_quality",
        "deterministic_quality",
    )

except ValueError:
    print(
        "\nWarning: utilization quartiles could not be formed because "
        "too many values were identical."
    )
    utilization_regions = pd.DataFrame()


# ---------------------------------------------------------------------
# Component initial-quality regions
# ---------------------------------------------------------------------

component["component_initial_quality_quartile"] = pd.qcut(
    component["zero_quality"],
    q=4,
    labels=[
        "Q1_lowest",
        "Q2",
        "Q3",
        "Q4_highest",
    ],
    duplicates="drop",
)

component_initial_quality_regions = summarize_regions(
    component,
    "component_initial_quality_quartile",
    "zero_quality",
    "structured_quality",
    "deterministic_quality",
)


# ---------------------------------------------------------------------
# Direct-simulator reference comparison
# ---------------------------------------------------------------------

direct = direct.copy()

direct["structured_regret_vs_direct"] = (
    direct["structured_final_quality"]
    - direct["direct_final_quality"]
)

direct["deterministic_regret_vs_direct"] = (
    direct["deterministic_final_quality"]
    - direct["direct_final_quality"]
)

direct["structured_closer_to_direct"] = (
    direct["structured_regret_vs_direct"].abs()
    <
    direct["deterministic_regret_vs_direct"].abs()
)

structured_closer_percent = (
    100.0 * direct["structured_closer_to_direct"].mean()
)


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

assembly_initial_correlation = assembly[
    ["zero_final_quality", "structured_advantage"]
].corr().iloc[0, 1]

component_initial_correlation = component[
    ["zero_quality", "structured_advantage"]
].corr().iloc[0, 1]

utilization_correlation = component[
    ["deterministic_utilization", "structured_advantage"]
].corr().iloc[0, 1]


# ---------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------

assembly.to_csv(
    OUTPUT_DIR / "ml_value_assembly_comparison.csv",
    index=False,
)

component.to_csv(
    OUTPUT_DIR / "ml_value_component_comparison.csv",
    index=False,
)

initial_quality_regions.to_csv(
    OUTPUT_DIR / "ml_value_by_initial_quality.csv",
    index=False,
)

stage_regions.to_csv(
    OUTPUT_DIR / "ml_value_by_stage.csv",
    index=False,
)

component_initial_quality_regions.to_csv(
    OUTPUT_DIR / "ml_value_by_component_initial_quality.csv",
    index=False,
)

if not utilization_regions.empty:
    utilization_regions.to_csv(
        OUTPUT_DIR / "ml_value_by_lsq_utilization.csv",
        index=False,
    )

direct.to_csv(
    OUTPUT_DIR / "ml_value_vs_direct_reference.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

summary_rows = [
    ("assemblies", n_assemblies),
    ("structured20_wins", structured_wins),
    ("deterministic_lsq_wins", lsq_wins),
    ("ties", ties),
    ("structured20_win_percent", structured_win_rate),
    ("deterministic_lsq_win_percent", lsq_win_rate),
    ("tie_percent", tie_rate),
    (
        "mean_zero_final_quality",
        assembly["zero_final_quality"].mean(),
    ),
    (
        "mean_structured_final_quality",
        assembly["structured_final_quality"].mean(),
    ),
    (
        "mean_deterministic_final_quality",
        assembly["deterministic_final_quality"].mean(),
    ),
    (
        "mean_structured_advantage",
        mean_advantage,
    ),
    (
        "bootstrap_95ci_lower",
        ci_low,
    ),
    (
        "bootstrap_95ci_upper",
        ci_high,
    ),
    (
        "structured_closer_to_direct_percent",
        structured_closer_percent,
    ),
    (
        "correlation_initial_quality_vs_structured_advantage",
        assembly_initial_correlation,
    ),
    (
        "correlation_component_quality_vs_structured_advantage",
        component_initial_correlation,
    ),
    (
        "correlation_lsq_utilization_vs_structured_advantage",
        utilization_correlation,
    ),
]

summary = pd.DataFrame(
    summary_rows,
    columns=["metric", "value"],
)

summary.to_csv(
    OUTPUT_DIR / "ml_value_summary.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    assembly["zero_final_quality"],
    assembly["structured_advantage"],
    alpha=0.65,
)

plt.axhline(0.0, linewidth=1.0)

plt.xlabel("Zero-correction final quality")
plt.ylabel("Structured-20 advantage (Q_LSQ - Q_Structured)")
plt.title("Structured-20 Relative Value vs Initial Assembly Difficulty")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ml_value_vs_initial_quality.png",
    dpi=220,
)
plt.close()


plt.figure(figsize=(8, 5))

plt.scatter(
    component["deterministic_utilization"],
    component["structured_advantage"],
    alpha=0.45,
)

plt.axhline(0.0, linewidth=1.0)

plt.xlabel("Deterministic LSQ capability utilization")
plt.ylabel("Structured-20 advantage (Q_LSQ - Q_Structured)")
plt.title("Structured-20 Relative Value vs LSQ Correction Effort")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ml_value_vs_lsq_utilization.png",
    dpi=220,
)
plt.close()


plt.figure(figsize=(8, 5))

stage_plot = stage_regions.copy()
stage_plot["component_index"] = pd.to_numeric(
    stage_plot["component_index"]
)

plt.plot(
    stage_plot["component_index"],
    stage_plot["structured_win_percent"],
    marker="o",
    label="Structured-20",
)

plt.plot(
    stage_plot["component_index"],
    stage_plot["deterministic_win_percent"],
    marker="o",
    label="Deterministic LSQ",
)

plt.xlabel("Sequential component index")
plt.ylabel("Win rate (%)")
plt.title("Controller Win Rate Across Sequential Assembly Stages")
plt.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ml_value_by_stage.png",
    dpi=220,
)
plt.close()


plt.figure(figsize=(7, 5))

plot_data = [
    assembly["structured_final_quality"],
    assembly["deterministic_final_quality"],
    direct["direct_final_quality"],
]

plt.boxplot(
    plot_data,
    tick_labels=[
        "Structured-20",
        "Deterministic LSQ",
        "Direct simulator",
    ],
)

plt.ylabel("Final quality")
plt.title("Final Quality Relative to Simulator-Direct Reference")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "ml_value_controller_reference_boxplot.png",
    dpi=220,
)
plt.close()


# ---------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("ASSEMBLY-LEVEL RESULTS")
print("=" * 78)

print(f"Assemblies                : {n_assemblies}")
print(
    f"Structured-20 wins        : {structured_wins} "
    f"({structured_win_rate:.2f}%)"
)
print(
    f"Deterministic LSQ wins    : {lsq_wins} "
    f"({lsq_win_rate:.2f}%)"
)
print(
    f"Ties                      : {ties} "
    f"({tie_rate:.2f}%)"
)

print(
    f"\nMean zero quality         : "
    f"{assembly['zero_final_quality'].mean():.4f}"
)
print(
    f"Mean Structured-20 quality: "
    f"{assembly['structured_final_quality'].mean():.4f}"
)
print(
    f"Mean deterministic quality: "
    f"{assembly['deterministic_final_quality'].mean():.4f}"
)

print("\nPaired Structured-20 advantage:")
print(
    f"Mean Q_LSQ - Q_Structured : {mean_advantage:.6f}"
)
print(
    f"95% bootstrap CI          : "
    f"[{ci_low:.6f}, {ci_high:.6f}]"
)

if ci_low > 0:
    statistical_interpretation = (
        "Structured-20 has a positive mean advantage over LSQ."
    )
elif ci_high < 0:
    statistical_interpretation = (
        "Deterministic LSQ has a positive mean advantage over Structured-20."
    )
else:
    statistical_interpretation = (
        "The mean difference includes zero within the 95% bootstrap CI."
    )

print(f"Interpretation            : {statistical_interpretation}")


print("\n" + "=" * 78)
print("VALUE-REGION SIGNALS")
print("=" * 78)

print(
    "Correlation: initial assembly quality vs Structured advantage "
    f"= {assembly_initial_correlation:.4f}"
)

print(
    "Correlation: component initial quality vs Structured advantage "
    f"= {component_initial_correlation:.4f}"
)

print(
    "Correlation: LSQ utilization vs Structured advantage "
    f"= {utilization_correlation:.4f}"
)

print(
    f"Structured-20 closer to direct reference in "
    f"{structured_closer_percent:.2f}% of assemblies"
)

print("\nInitial-quality regions:")
print(initial_quality_regions.to_string(index=False))

print("\nStage regions:")
print(stage_regions.to_string(index=False))

if not utilization_regions.empty:
    print("\nLSQ-utilization regions:")
    print(utilization_regions.to_string(index=False))


print("\n" + "=" * 78)
print("SCIENTIFIC VERDICT")
print("=" * 78)

if ci_high < 0:
    print(
        "RESULT A: Deterministic LSQ retains a statistically supported "
        "mean advantage over Structured-20 on the evaluated assemblies."
    )
    print(
        "The ML-supported controller should therefore not be claimed "
        "as globally superior under the present additive geometric model."
    )

elif ci_low > 0:
    print(
        "RESULT B: Structured-20 shows a statistically supported mean "
        "advantage over deterministic LSQ."
    )

else:
    print(
        "RESULT C: No statistically clear mean advantage is established "
        "between Structured-20 and deterministic LSQ."
    )

print(
    "\nThe region tables quantify whether Structured-20 becomes relatively "
    "more competitive under specific levels of state difficulty, "
    "sequential stage, or correction effort."
)

print(f"\nResults saved to:\n{OUTPUT_DIR}")

print("\nML VALUE REGION AUDIT COMPLETED")