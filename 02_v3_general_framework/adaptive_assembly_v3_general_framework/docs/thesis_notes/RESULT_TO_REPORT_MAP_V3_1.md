# Result-to-Report Map V3.1

## Purpose

This file maps the frozen Version 3.1 technical outputs to the future thesis report structure.

It is not the final report. It is a preparation note to make report writing clear, consistent, and defensible.

## Frozen technical version

Version 3.1:

ML-Based Deviation- and Capability-Aware Adaptive Assembly Recommendation System

## Core thesis idea

The project investigates whether assembly parameters can be adaptively recommended based on part-specific geometric deviations.

The final prototype evaluates a deviation-aware and capability-aware recommendation framework using controlled synthetic part-variation scenarios.

The system does not only predict quality. It also:

- recommends assembly correction parameters
- compares fixed, batch-level, and individual adaptation
- classifies recommendation feasibility
- analyzes failure modes
- identifies correction-capability limits

---

# 1. Introduction / Motivation

## Main argument

Fixed assembly parameters are limited when nominally identical parts show individual geometric deviations.

Adaptive parameter recommendation can improve assembly quality by using part-specific deviation information.

## Supporting project result

Individual adaptive recommendation strongly outperformed fixed settings.

Key numbers:

- Nominal/global pass rate: about 53.18%
- Individual adaptive pass rate: about 95.45%
- Pass-rate improvement: about 42.27 percentage points
- Mean quality improvement: about 65.82%

## Supporting files

Tables:

- results/tables/adaptation_level_comparison.csv
- results/tables/recommendation_status_summary.csv

Plots:

- results/plots/v3/adaptation_strategy_pass_rate_comparison.png
- results/plots/v3/adaptation_strategy_quality_comparison.png

---

# 2. Methodology: Synthetic Deviation Dataset

## Main argument

Because real industrial measurement data was not available, representative synthetic deviation profiles were used to create a controlled and reproducible validation environment.

## Important limitation to state

The synthetic deviation profiles do not prove industrial production readiness. They are used to evaluate the framework logic under controlled conditions.

## Supporting files

Scripts:

- scripts/run_05_generate_batch_dataset.py
- src/batch_deviation_generator.py

Tables:

- results/tables/batch_aware_dataset.csv
- results/tables/batch_aware_dataset_summary.csv

## Scenarios included

- offset_dominant
- tilt_dominant
- bend_dominant
- waviness_dominant
- twist_dominant
- fixture_drift
- mixed
- disturbed_batch

---

# 3. Methodology: Quality Metrics and Correction Parameters

## Main argument

The framework evaluates assembly quality using interpretable geometric quality metrics.

The current prototype uses a limited but physically interpretable correction parameter set.

## Quality metrics

- mean_gap
- max_gap
- parallelism_error
- quality_score

## Correction parameters

- z_adj
- theta_adj
- locator_offset

## Important limitation to state

The correction parameters represent a simplified assembly correction model. The prototype is not a full industrial digital twin or FEM-based assembly simulation.

---

# 4. Adaptation Strategy Comparison

## Main argument

The project compares different levels of adaptation:

- nominal universal baseline
- global best fixed setting
- batch-adaptive setting
- individual-adaptive setting

This shows whether adaptation is useful and whether individual adaptation is better than batch-level or global correction.

## Key result

Individual adaptive recommendation achieved the strongest improvement.

Key values:

- Nominal/global pass rate: about 53.18%
- Batch-adaptive pass rate: about 56.36%
- Individual-adaptive pass rate: about 95.45%
- Individual pass-rate improvement vs nominal: about 42.27 percentage points

## Interpretation

Batch adaptation gives only limited improvement under heterogeneous deviations.

Individual adaptation is much more effective because each part may require its own correction setting.

## Supporting files

Script:

- scripts/run_06_adaptation_level_evaluation.py

Table:

- results/tables/adaptation_level_comparison.csv

Plots:

- results/plots/v3/adaptation_strategy_pass_rate_comparison.png
- results/plots/v3/adaptation_strategy_quality_comparison.png

---

# 5. ML Surrogate Model Selection

## Main argument

The Random Forest surrogate was not chosen blindly. Several models were tested and compared.

## Models compared

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression

## Key result

Random Forest ranked best overall.

Approximate ranking:

1. Random Forest
2. Gradient Boosting
3. Support Vector Regression
4. Linear Regression
5. Ridge Regression

## Interpretation

Linear models performed poorly because the relationship between deviations, correction parameters, and quality is nonlinear.

Random Forest was selected because it achieved the best predictive performance for the current nonlinear tabular dataset.

## Important limitation to state

The high Random Forest performance is based on controlled synthetic data. It should not be interpreted as proof of industrial prediction accuracy on real production data.

## Supporting files

Script:

- scripts/run_10_model_screening.py

Tables:

- results/tables/model_screening_results.csv
- results/tables/model_screening_summary.csv

Plots:

- results/plots/v3/model_screening_rmse_comparison.png
- results/plots/v3/model_screening_r2_comparison.png

---

# 6. Bayesian Optimization Justification

## Main argument

Bayesian Optimization was compared against Random Search. It was not selected only because it sounds advanced.

## Key result

Bayesian Optimization achieved better final mean best quality than Random Search.

Key value:

- BO improvement vs Random Search: about 17.87%

## Interpretation

Because the prototype parameter space is low-dimensional, Random Search remains competitive.

However, Bayesian Optimization shows better sample efficiency and becomes more valuable when evaluations are expensive, such as simulation-based, measurement-based, or real assembly-trial-based evaluations.

## Supporting files

Script:

- scripts/run_09_check_optimization_efficiency_results.py

Tables:

- results/tables/optimization_efficiency_summary.csv
- results/tables/optimization_efficiency_checkpoints.csv
- results/tables/optimization_efficiency_curve.csv

Plot:

- results/plots/v3/optimization_efficiency_curve.png

---

# 7. Recommendation-Status Classification

## Main argument

The framework should not only output one optimized parameter set. It should also classify whether the recommendation is feasible.

## Status classes

- PASS_RECOMMENDED
- MANUAL_REVIEW
- REWORK_REQUIRED
- OUT_OF_CORRECTION_RANGE

## Key result

Individual adaptive setting:

- PASS_RECOMMENDED: about 95.45%
- REWORK_REQUIRED: about 4.55%

Nominal/global setting:

- PASS_RECOMMENDED: about 53.18%
- significant shares of rework/out-of-correction-range cases

## Interpretation

The system behaves as a decision-support framework, not as a perfect guarantee system.

Remaining non-pass cases are routed to rework/review instead of being falsely accepted.

## Supporting files

Script:

- scripts/run_07_recommendation_status_analysis.py

Tables:

- results/tables/recommendation_status_cases.csv
- results/tables/recommendation_status_summary.csv
- results/tables/recommendation_status_by_scenario.csv

Plot:

- results/plots/v3/recommendation_status_distribution.png

---

# 8. Failure Analysis

## Main argument

The project does not hide failed cases. It analyzes why remaining failures occur.

## Key result

For individual adaptive recommendation, remaining failed cases are dominated by:

- parallelism_error

Key value:

- individual adaptive remaining failure metric share: 100% parallelism_error-related

## Interpretation

The current correction model can reduce gap-related errors effectively, but some cases remain limited by angular/end-to-end mismatch.

This indicates a correction-capability limitation, not only an optimizer failure.

## Supporting files

Script:

- scripts/run_08_failure_analysis.py

Tables:

- results/tables/failure_analysis_by_strategy.csv
- results/tables/failure_analysis_by_scenario.csv
- results/tables/failure_analysis_by_metric.csv
- results/tables/individual_adaptive_failed_cases.csv

Plot:

- results/plots/v3/failure_metric_comparison.png

---

# 9. Capability-Aware Correction Analysis

## Main argument

The final framework goes beyond parameter optimization by identifying which deviation scenarios remain difficult after individual adaptive correction.

## Key result

The capability analysis ranks scenarios by remaining difficult-case share.

Most difficult scenario:

- fixture_drift

Easiest/correctable scenario:

- offset_dominant

Dominant remaining limitation:

- parallelism_error

Suggested future capability:

- stronger angular correction
- multi-point locator correction capability

## Interpretation

This is one of the strongest industrially relevant parts of the project.

The framework does not only say what parameter should be used. It also indicates when the available correction capability is insufficient.

## Supporting files

Script:

- scripts/run_12_capability_map_analysis.py
- scripts/run_13_plot_capability_map.py
- scripts/run_14_plot_correction_difficulty_ranking.py

Tables:

- results/tables/correction_capability_map.csv
- results/tables/correction_capability_summary.csv

Plots:

- results/plots/v3/correction_capability_map.png
- results/plots/v3/correction_difficulty_ranking.png

---

# 10. Limitations

## Required limitation points

The report must clearly state:

- real industrial measurement data was not available
- synthetic data was used for controlled validation
- the assembly correction model is simplified
- the prototype is not a full physical assembly simulation
- only three correction parameters are currently used
- real production validation is future work
- numerical results should not be directly transferred to industry without real-data validation

## Correct limitation framing

These limitations do not invalidate the project.

They define the scope of the current project thesis and motivate future work.

---

# 11. Future Work

## Strong future work directions

- validation using real CMM, 3D scan, or inline measurement data
- integration with industrial digital twins or process simulations
- extension to multi-point locator adjustment
- inclusion of clamping and joining-sequence parameters
- closed-loop adaptive assembly control
- uncertainty-aware recommendation
- transfer from synthetic to real data
- master-thesis extension using real industrial case study

---

# 12. Final Contribution Statement

The final contribution of this project is:

A controlled prototype of a deviation- and capability-aware adaptive assembly recommendation framework that combines part-specific quality prediction, assembly parameter optimization, recommendation-status classification, failure analysis, and correction-capability evaluation.

The work demonstrates that individualized parameter recommendation can significantly improve simulated assembly quality compared with fixed or batch-level settings, while also identifying cases where the available correction capability is insufficient.

---

# Final rule for report writing

Do not overclaim industrial readiness.

Use this framing:

Controlled technical feasibility demonstration under synthetic part-variation scenarios.

Do not use this framing:

Fully validated industrial production system.
