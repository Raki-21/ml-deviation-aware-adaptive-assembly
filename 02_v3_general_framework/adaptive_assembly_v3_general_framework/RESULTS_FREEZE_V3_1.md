# Results Freeze - Version 3.1

## Version identity

Version 3.1: Deviation- and Capability-Aware Adaptive Assembly Recommendation Framework

## Purpose

This version extends Version 3 by adding a capability-aware assembly decision layer. The system does not only recommend adaptive assembly parameters, but also evaluates scenario-wise correction difficulty and identifies the correction-capability boundary of the current assembly model.

## Dataset

- Rows: 8800
- Columns: 22
- Unique physical part cases: 1100

## Scenarios

- offset_dominant
- tilt_dominant
- bend_dominant
- waviness_dominant
- mixed
- twist_dominant
- fixture_drift
- disturbed_batch

## Main adaptation results

- Nominal/global pass rate: approximately 53.18%
- Batch-adaptive pass rate: approximately 56.36%
- Individual adaptive pass rate: approximately 95.45%
- Individual adaptive mean quality score: approximately 0.6557
- Individual adaptive improvement versus nominal: approximately 65.82%

## ML model screening

Best model: Random Forest

- Mean MAE: approximately 0.2977
- Mean RMSE: approximately 0.4140
- Mean R²: approximately 0.9881

## Optimization efficiency

- Bayesian Optimization final mean best quality: approximately 1.2779
- Random Search final mean best quality: approximately 1.5560
- Bayesian Optimization improvement versus Random Search: approximately 17.87%

## Recommendation status

Individual adaptive setting:

- PASS_RECOMMENDED: 1050 / 1100
- REWORK_REQUIRED: 50 / 1100
- OUT_OF_CORRECTION_RANGE: 0 / 1100
- MANUAL_REVIEW: 0 / 1100

## Capability-aware analysis

Main results:

- fixture_drift: approximately 90.00% pass, 10.00% difficult/rework-required
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

## Generated Phase 4 outputs

Tables:

- correction_capability_map.csv
- correction_capability_summary.csv

Plots:

- correction_capability_map.png
- correction_difficulty_ranking.png

## Version 3.1 conclusion

Version 3.1 strengthens the thesis by adding a capability-aware decision layer. The framework now demonstrates not only adaptive parameter recommendation, but also scenario-wise correction difficulty identification, correction-limit interpretation, and future process-capability guidance.

This supports the long-term direction of an industry-relevant adaptive assembly decision-support system.