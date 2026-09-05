# Supervisor Update - Version 3.1

## Project title

Machine Learning-Based Optimization of Assembly Parameters Considering Part Variations

## Current framework direction

The current implementation has been developed as a deviation-aware adaptive assembly parameter recommendation framework.

The main idea is to avoid using one fixed assembly parameter setting for all parts. Instead, the system uses part-specific deviation information to recommend suitable assembly parameters and evaluate the expected assembly quality.

The current framework combines:

- synthetic deviation profile generation
- batch-aware dataset generation
- ML-based surrogate quality prediction
- Bayesian Optimization for parameter recommendation
- comparison of adaptation strategies
- recommendation-status classification
- failure analysis
- capability-aware correction analysis

---

## Implemented Version 3.1 components

### 1. Batch-aware dataset generation

A batch-aware dataset was generated with scenario and severity metadata.

Current dataset:

- 8800 rows
- 22 columns
- 1100 unique physical part cases

Implemented deviation/circumstance scenarios:

- offset_dominant
- tilt_dominant
- bend_dominant
- waviness_dominant
- mixed
- twist_dominant
- fixture_drift
- disturbed_batch

---

### 2. Adaptation strategy comparison

The following strategies were compared:

- nominal universal baseline
- global-best fixed setting
- batch-adaptive setting
- individual adaptive setting

Main result:

- nominal/global pass rate: approximately 53.18%
- batch-adaptive pass rate: approximately 56.36%
- individual adaptive pass rate: approximately 95.45%

The individual adaptive strategy showed the strongest improvement.

---

### 3. Recommendation-status classification

A recommendation-status layer was added to classify optimized cases into:

- PASS_RECOMMENDED
- MANUAL_REVIEW
- REWORK_REQUIRED
- OUT_OF_CORRECTION_RANGE

For the individual adaptive setting:

- 1050 / 1100 cases were PASS_RECOMMENDED
- 50 / 1100 cases were REWORK_REQUIRED
- 0 / 1100 cases were OUT_OF_CORRECTION_RANGE
- 0 / 1100 cases were MANUAL_REVIEW

This makes the framework more realistic because non-passing optimized cases are not blindly treated as acceptable.

---

### 4. Failure analysis

Failure analysis was performed for the remaining difficult individual-adaptive cases.

Main finding:

- remaining difficult cases were mainly caused by parallelism_error
- mean_gap and max_gap were not the dominant remaining limitations

Interpretation:

The current correction model is effective for gap-related correction, but some remaining cases are limited by parallelism correction capability.

---

### 5. ML model screening

The following surrogate models were compared:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression

Main result:

- Random Forest achieved the best overall performance
- Random Forest mean MAE: approximately 0.2977
- Random Forest mean RMSE: approximately 0.4140
- Random Forest mean R²: approximately 0.9881

Interpretation:

Random Forest was selected as the primary surrogate model because it provided the best balance of prediction accuracy, robustness, computational practicality, and interpretability.

---

### 6. Bayesian Optimization comparison

Bayesian Optimization was compared with Random Search.

Main result:

- Bayesian Optimization final mean best quality: approximately 1.2779
- Random Search final mean best quality: approximately 1.5560
- Bayesian Optimization improvement versus Random Search: approximately 17.87%

Interpretation:

Bayesian Optimization is useful because it finds better parameter settings more efficiently than uninformed random sampling.

---

### 7. Capability-aware correction analysis

A capability-aware analysis was added to evaluate scenario-wise correction difficulty after individual adaptive recommendation.

Main result:

- fixture_drift was the most difficult scenario:
  - approximately 90.00% pass recommendation rate
  - approximately 10.00% difficult/rework-required cases

- offset_dominant was the most correctable scenario:
  - approximately 98.67% pass recommendation rate
  - approximately 1.33% difficult cases

Dominant remaining limitation:

- parallelism_error

Interpretation:

The framework now identifies not only recommended assembly parameters, but also where the current correction capability reaches its limit.

---

## Generated key plots

The current result plots are saved in:

```text
results/plots/v3