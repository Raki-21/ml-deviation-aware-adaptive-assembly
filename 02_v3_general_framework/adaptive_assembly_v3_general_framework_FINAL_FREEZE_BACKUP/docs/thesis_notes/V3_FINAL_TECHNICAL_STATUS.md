# Version 3 Final Technical Status

## 1. Current project direction

The current project direction is:

ML-Based Deviation-Aware Adaptive Assembly Parameter Recommendation System

The system is based on the D+E+F concept:

- D: individualized part-specific assembly parameter recommendation
- E: ML surrogate modelling for quality prediction
- F: Bayesian Optimization for intelligent parameter recommendation

The project remains aligned with the thesis topic:

Machine Learning-Based Optimization of Assembly Parameters Considering Part Variations

---

## 2. Completed technical components

### Version 2 locked baseline

Completed and preserved:

- synthetic deviation profile generation
- assembly correction model
- gap and parallelism quality metrics
- dataset generation
- Linear Regression baseline
- Random Forest surrogate model
- Bayesian Optimization
- Random Search comparison
- fixed baseline comparison
- robustness evaluation

Version 2 is locked and should not be overwritten.

---

### Version 3 batch-aware framework

Completed:

- batch-aware dataset generation
- part-level metadata
- scenario-level metadata
- severity levels
- disturbed batch condition

Current dataset:

- 8800 rows
- 22 columns
- 1100 unique physical part cases

Implemented scenarios:

- offset_dominant
- tilt_dominant
- bend_dominant
- waviness_dominant
- mixed
- twist_dominant
- fixture_drift
- disturbed_batch

---

## 3. Adaptation strategy evaluation

Completed strategies:

- nominal universal baseline
- global-best fixed setting
- batch-adaptive setting
- individual adaptive setting

Main results:

- Nominal/global pass rate: approximately 53.18%
- Batch-adaptive pass rate: approximately 56.36%
- Individual adaptive pass rate: approximately 95.45%
- Nominal mean quality score: approximately 1.9185
- Individual adaptive mean quality score: approximately 0.6557
- Individual adaptive improvement versus nominal: approximately 65.82%

Interpretation:

Individual part-specific adaptation clearly outperforms universal, global fixed, and batch-level adaptation strategies.

---

## 4. Recommendation-status / feasibility classification

Completed statuses:

- PASS_RECOMMENDED
- MANUAL_REVIEW
- REWORK_REQUIRED
- OUT_OF_CORRECTION_RANGE

Main results:

- Individual adaptive setting: 1050 / 1100 PASS_RECOMMENDED
- Individual adaptive setting: 50 / 1100 REWORK_REQUIRED
- Individual adaptive setting: 0 / 1100 OUT_OF_CORRECTION_RANGE

Interpretation:

The framework does not blindly release every optimized result. It supports industrial decision-making by identifying directly releasable, review, rework, and non-correctable cases.

---

## 5. Failure analysis

Completed failure analysis outputs:

- failure_analysis_by_strategy.csv
- failure_analysis_by_scenario.csv
- failure_analysis_by_metric.csv
- individual_adaptive_failed_cases.csv

Main finding:

The remaining individual-adaptive failures are caused by parallelism_error, not by mean_gap or max_gap.

Interpretation:

The current three-parameter correction model is effective for gap reduction, but some difficult cases remain limited by parallelism correction capability.

---

## 6. Optimization efficiency comparison

Completed result files:

- optimization_efficiency_summary.csv
- optimization_efficiency_checkpoints.csv
- optimization_efficiency_curve.csv

Main result:

- Bayesian Optimization final mean best quality: approximately 1.2779
- Random Search final mean best quality: approximately 1.5560
- Bayesian Optimization improvement versus Random Search: approximately 17.87%

Interpretation:

Bayesian Optimization is justified because it guides the search toward better assembly parameters more efficiently than uninformed random sampling.

Note:

Some optimization-efficiency script files became messy during cleanup. Broken versions were backed up. The valid result CSV files exist and are usable. Scenario metadata in the curve file contains some incomplete labels, but the numeric BO versus Random Search comparison is usable.

---

## 7. ML model screening

Completed model comparison:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression

Main results:

- Random Forest mean MAE: approximately 0.2977
- Random Forest mean RMSE: approximately 0.4140
- Random Forest mean R²: approximately 0.9881
- Gradient Boosting was second-best
- SVR was good but slower
- Linear Regression and Ridge Regression performed poorly with mean R² approximately 0.032

Interpretation:

Random Forest is justified as the primary surrogate model because it provides the best balance of accuracy, robustness, computational practicality, and interpretability.

---

## 8. Generated result plots

Completed plots in results/plots/v3:

- adaptation_strategy_quality_comparison.png
- adaptation_strategy_pass_rate_comparison.png
- recommendation_status_distribution.png
- failure_metric_comparison.png
- optimization_efficiency_curve.png
- model_screening_rmse_comparison.png
- model_screening_r2_comparison.png

These plots are suitable as first report-ready visual results.

---

## 9. Main thesis evidence now available

The current Version 3 evidence supports:

1. Fixed universal assembly settings are insufficient for heterogeneous part deviations.
2. Batch-level adaptation helps only slightly.
3. Individual part-specific adaptation gives the strongest quality improvement.
4. Feasibility classification improves industrial relevance.
5. Remaining failures are explainable and mainly parallelism-related.
6. Bayesian Optimization is more efficient than Random Search.
7. Random Forest is justified as the final surrogate model.

---

## 10. Remaining work

### Technical cleanup

- Clean or archive broken optimization-efficiency scripts.
- Keep only the validated result-check script or rebuild BO efficiency cleanly later.
- Check all generated plots visually.
- Possibly improve plot formatting if labels overlap.

### Thesis/report work

- Convert V3_RESULTS_INTERPRETATION.md into thesis Results and Discussion sections.
- Write Methodology chapter based on the implemented pipeline.
- Write clear limitations and future work.
- Add figures and result tables into the report.
- Prepare supervisor update / meeting explanation.

### Optional but useful

- Create one final results table summarizing all major findings.
- Create one architecture diagram of the full Version 3 framework.
- Create one flowchart from deviation input to recommendation status.