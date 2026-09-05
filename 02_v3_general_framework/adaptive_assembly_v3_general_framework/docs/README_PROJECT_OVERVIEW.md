# ML-Based Deviation- and Capability-Aware Adaptive Assembly Recommendation System

## Project title

Machine Learning-Based Optimization of Assembly Parameters Considering Part Variations

## Project identity

This project develops a deviation-aware and capability-aware adaptive assembly recommendation framework.

The system is designed to support assembly-quality improvement when nominally similar parts show individual geometric deviations.

Instead of applying one fixed assembly parameter setting to all parts, the framework uses part-specific deviation information to recommend suitable assembly parameters and evaluate whether the current correction capability is sufficient.

## Core concept

The project follows the D+E+F concept:

- D: individualized part-specific assembly parameter recommendation
- E: ML surrogate modelling for assembly quality prediction
- F: Bayesian Optimization for intelligent parameter recommendation

Version 3.1 extends this by adding a capability-aware decision layer.

The system does not only ask:

- Which assembly parameters are best?

It also asks:

- Can this part be corrected using the current assembly correction capability?
- Which deviation scenarios are difficult?
- Which quality metric limits the correction?
- What future correction capability would be needed?

## Implemented framework

The current prototype includes:

- synthetic deviation profile generation
- batch-aware dataset generation
- multiple deviation scenarios
- quality metric calculation
- ML surrogate model screening
- Random Forest surrogate model
- Bayesian Optimization
- Random Search comparison
- nominal/global/batch/individual strategy comparison
- recommendation-status classification
- failure analysis
- correction capability analysis
- result visualization

## Implemented deviation scenarios

The implemented scenarios are:

- offset_dominant
- tilt_dominant
- bend_dominant
- waviness_dominant
- mixed
- twist_dominant
- fixture_drift
- disturbed_batch

These scenarios are synthetic but controlled and interpretable. They are not claimed to be direct real industrial measurement data.

## Correction parameters

The current prototype uses three physically interpretable correction parameters:

- z_adj
- theta_adj
- locator_offset

These represent global vertical correction, angular correction, and local locator-related correction.

## Main dataset

Current Version 3.1 dataset:

- 8800 rows
- 22 columns
- 1100 unique physical part cases

Main dataset file:

- results/tables/batch_aware_dataset.csv

## Main results

### Adaptation strategy comparison

Main result:

- Nominal/global pass rate: approximately 53.18%
- Batch-adaptive pass rate: approximately 56.36%
- Individual adaptive pass rate: approximately 95.45%

The individual adaptive strategy achieved the strongest improvement.

### Recommendation-status classification

For the individual adaptive setting:

- PASS_RECOMMENDED: 1050 / 1100
- REWORK_REQUIRED: 50 / 1100
- OUT_OF_CORRECTION_RANGE: 0 / 1100
- MANUAL_REVIEW: 0 / 1100

This shows that the framework does not blindly release every optimized result. It identifies remaining difficult cases for rework.

### Failure analysis

The remaining individual-adaptive difficult cases are mainly caused by:

- parallelism_error

This means the current correction model is effective for gap reduction but limited for some angular or end-to-end mismatch cases.

### ML model screening

Compared models:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression

Best model:

- Random Forest

Random Forest results:

- Mean MAE: approximately 0.2977
- Mean RMSE: approximately 0.4140
- Mean R²: approximately 0.9881

Random Forest was selected because it provided the best balance of prediction accuracy, robustness, computational practicality, and interpretability.

### Optimization efficiency

Bayesian Optimization was compared with Random Search.

Main result:

- Bayesian Optimization final mean best quality: approximately 1.2779
- Random Search final mean best quality: approximately 1.5560
- Bayesian Optimization improvement versus Random Search: approximately 17.87%

This supports Bayesian Optimization as an efficient parameter recommendation method.

### Capability-aware analysis

The correction capability analysis ranks scenarios by remaining difficult-case percentage after individual adaptive recommendation.

Main results:

- fixture_drift: approximately 90.00% pass, 10.00% difficult
- disturbed_batch: approximately 94.00% pass, 6.00% difficult
- waviness_dominant: approximately 94.67% pass, 5.33% difficult
- mixed: approximately 95.33% pass, 4.67% difficult
- tilt_dominant: approximately 95.33% pass, 4.67% difficult
- bend_dominant: approximately 97.33% pass, 2.67% difficult
- twist_dominant: approximately 97.33% pass, 2.67% difficult
- offset_dominant: approximately 98.67% pass, 1.33% difficult

Dominant remaining limitation:

- parallelism_error

Suggested future correction capability:

- stronger angular or multi-point locator correction capability

## Generated key result files

Important tables:

- results/tables/batch_aware_dataset.csv
- results/tables/adaptation_level_comparison.csv
- results/tables/recommendation_status_cases.csv
- results/tables/recommendation_status_summary.csv
- results/tables/failure_analysis_by_metric.csv
- results/tables/model_screening_summary.csv
- results/tables/optimization_efficiency_summary.csv
- results/tables/correction_capability_map.csv
- results/tables/correction_capability_summary.csv

Important plots:

- results/plots/v3/adaptation_strategy_quality_comparison.png
- results/plots/v3/adaptation_strategy_pass_rate_comparison.png
- results/plots/v3/recommendation_status_distribution.png
- results/plots/v3/failure_metric_comparison.png
- results/plots/v3/optimization_efficiency_curve.png
- results/plots/v3/model_screening_rmse_comparison.png
- results/plots/v3/model_screening_r2_comparison.png
- results/plots/v3/correction_capability_map.png
- results/plots/v3/correction_difficulty_ranking.png

## Script execution order

Recommended script order:

```bash
python scripts/run_05_generate_batch_dataset.py
python scripts/run_06_adaptation_level_evaluation.py
python scripts/run_07_recommendation_status_analysis.py
python scripts/run_08_failure_analysis.py
python scripts/run_09_check_optimization_efficiency_results.py
python scripts/run_10_model_screening.py
python scripts/run_11_generate_result_plots.py
python scripts/run_12_capability_map_analysis.py
python scripts/run_13_plot_capability_map.py
python scripts/run_14_plot_correction_difficulty_ranking.py