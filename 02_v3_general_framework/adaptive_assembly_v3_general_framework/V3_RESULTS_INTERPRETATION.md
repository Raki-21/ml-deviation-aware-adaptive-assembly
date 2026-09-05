# Version 3 Results Interpretation

## 1. Purpose of Version 3

Version 3 extends the earlier single-case adaptive assembly prototype into a more general deviation-aware adaptive assembly recommendation framework. The objective is to evaluate whether individualized assembly parameter recommendation can improve assembly quality under part-specific, batch-specific, and scenario-specific deviations.

The framework remains aligned with the project thesis topic: Machine Learning-Based Optimization of Assembly Parameters Considering Part Variations.

The core idea is:

- Part deviations are represented as measurable/synthetic deviation inputs.
- Assembly correction parameters are adjusted to reduce quality errors.
- Quality is evaluated using gap- and parallelism-based metrics.
- Different adaptation strategies are compared.
- Feasibility/recommendation status is assigned to support industrial decision-making.

---

## 2. Dataset and deviation scenarios

The Version 3 batch-aware dataset contains:

- 8800 sampled rows
- 1100 unique physical part cases
- 22 columns
- batch-level metadata
- part-level metadata
- deviation inputs
- assembly correction parameters
- quality metrics

Implemented deviation/circumstance scenarios:

- offset_dominant
- tilt_dominant
- bend_dominant
- waviness_dominant
- mixed
- twist_dominant
- fixture_drift
- disturbed_batch

The two additional scenarios, twist_dominant and fixture_drift, were added as controlled synthetic extensions. They increase the difficulty and depth of the validation without falsely claiming availability of real industrial measurement data.

The implemented deviation modes are not intended to represent every possible industrial deviation. Instead, they form a controlled and interpretable set of representative deviation families for testing the adaptive recommendation framework.

---

## 3. Adaptation strategy comparison

Four adaptation strategies were compared:

1. Nominal universal baseline
2. Global-best fixed setting
3. Batch-adaptive setting
4. Individual adaptive setting

Main result after adding twist_dominant and fixture_drift:

- Nominal/global pass rate: approximately 53.18%
- Batch-adaptive pass rate: approximately 56.36%
- Individual adaptive pass rate: approximately 95.45%

Mean quality score:

- Nominal universal baseline: approximately 1.9185
- Individual adaptive setting: approximately 0.6557

The individual adaptive setting achieved approximately 65.82% improvement versus the nominal baseline.

Interpretation:

The nominal baseline and global-best fixed setting are unable to compensate for heterogeneous part deviations. Batch-adaptive correction provides only a moderate improvement because individual parts within the same batch can still show different deviation behavior. The individual adaptive setting provides the strongest improvement because the assembly parameters are selected according to each specific part deviation case.

This confirms the main thesis idea: part-specific adaptive assembly parameter recommendation is more effective than universal or batch-level parameter selection under heterogeneous part variations.

---

## 4. Recommendation-status and feasibility classification

The recommendation-status layer converts numerical optimization output into an industrial-style decision-support result.

The statuses are:

- PASS_RECOMMENDED
- MANUAL_REVIEW
- REWORK_REQUIRED
- OUT_OF_CORRECTION_RANGE

Main result for the individual adaptive setting:

- 1050 of 1100 cases were classified as PASS_RECOMMENDED
- 50 of 1100 cases were classified as REWORK_REQUIRED
- 0 cases were classified as OUT_OF_CORRECTION_RANGE

Interpretation:

The system does not blindly claim that every optimized case is acceptable. Instead, it separates directly releasable recommendations from cases that still require additional action. This makes the framework more realistic for industrial quality decision support.

The result shows that individual adaptive recommendation strongly increases the direct release/pass rate while still identifying remaining difficult cases for rework.

---

## 5. Failure analysis

Failure analysis was performed to understand why the individual adaptive strategy did not reach 100% pass rate.

Main result:

- The remaining 50 individual-adaptive difficult cases were all classified as REWORK_REQUIRED.
- No individual-adaptive case was OUT_OF_CORRECTION_RANGE.
- The remaining failures were caused by parallelism_error.
- The remaining failures were not caused by mean_gap or max_gap.

Interpretation:

The current correction model is effective at reducing mean gap and maximum gap errors. However, some remaining cases still violate the parallelism requirement. This indicates a physical correction limitation of the current three-parameter correction model rather than a general failure of the adaptive recommendation approach.

The current correction parameters are:

- z_adj
- theta_adj
- locator_offset

These are intentionally kept physically interpretable. Additional correction actions such as multi-point locator adjustment, additional angular correction, or clamping-force optimization may be considered as future work, but they are not added in the current thesis to avoid fake physical complexity.

---

## 6. Optimization efficiency comparison

Optimization efficiency was evaluated by comparing Bayesian Optimization with Random Search.

Main result:

- Bayesian Optimization final mean best quality score: approximately 1.2779
- Random Search final mean best quality score: approximately 1.5560
- Bayesian Optimization improvement versus Random Search: approximately 17.87%

Checkpoint analysis showed that after the initial exploration phase, Bayesian Optimization consistently achieved lower mean best quality scores than Random Search.

Interpretation:

Bayesian Optimization is justified because it uses previous evaluations to guide the search toward better parameter settings. This is more efficient than uninformed random sampling, especially when evaluations represent physical tests, simulation runs, measurement effort, or production trial cost.

---

## 7. ML model screening

The following surrogate models were compared:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression

Main results:

- Random Forest achieved the best overall performance.
- Random Forest mean MAE: approximately 0.2977
- Random Forest mean RMSE: approximately 0.4140
- Random Forest mean R²: approximately 0.9881
- Gradient Boosting was second-best.
- Support Vector Regression also performed well but required longer training time.
- Linear Regression and Ridge Regression performed poorly, with mean R² approximately 0.032.

Interpretation:

The poor performance of Linear Regression and Ridge Regression confirms that the relationship between part deviations, assembly parameters, and quality score is strongly nonlinear. Random Forest was selected as the primary surrogate model because it provided the best balance of prediction accuracy, robustness, computational practicality, and interpretability.

---

## 8. Overall technical conclusion

Version 3 demonstrates that deviation-aware individualized assembly parameter recommendation can significantly improve assembly quality compared with fixed or batch-level parameter strategies.

The key technical findings are:

- Fixed universal settings are insufficient for heterogeneous deviations.
- Batch-level adaptation provides only limited improvement.
- Individual part-specific adaptation gives the strongest improvement.
- Recommendation-status classification makes the system more realistic for industrial decision support.
- Remaining failures are mainly caused by parallelism error.
- Bayesian Optimization improves search efficiency compared with Random Search.
- Random Forest is justified as the main surrogate model through model screening.

Overall, the Version 3 framework supports the thesis direction: ML-based deviation-aware adaptive assembly parameter recommendation under part variations.

---

## 9. Current limitations

The current work has the following honest limitations:

- The dataset is synthetic because no real industrial measurement data are available.
- The profile model is simplified and represents a controlled validation environment.
- The correction model uses three physically interpretable parameters only.
- The current implementation does not include full 3D scan data, CAD/FEM-based deformation simulation, clamping-force optimization, or joining-sequence optimization.
- Optimization-efficiency metadata still requires later cleanup for scenario labels, although the numeric BO versus Random Search comparison is usable.

These limitations do not invalidate the work. They define the scope of the current project thesis and motivate future research.

---

## 10. Future work

Possible future extensions include:

- validation using real CMM, 3D scan, or inline measurement data
- multi-part assembly extension
- multi-point locator adjustment
- clamping-force optimization
- uncertainty-aware recommendation
- online process drift detection
- closed-loop quality feedback
- industrial dashboard or decision-support interface
- extension toward a master thesis using real production or simulation-derived data

The current thesis provides a strong foundation for these future extensions.

## 6. Capability-aware correction analysis

A correction capability analysis was added to evaluate not only whether the individual adaptive recommendation improves quality, but also which deviation scenarios remain difficult for the current correction model.

The analysis evaluates the individual adaptive recommendation strategy scenario by scenario and identifies:

- pass-recommended share
- rework-required share
- out-of-correction-range share
- dominant limiting quality metric
- suggested future correction capability

Main result:

- fixture_drift was the most difficult scenario, with approximately 90.00% pass recommendation rate and 10.00% difficult/rework-required cases.
- disturbed_batch reached approximately 94.00% pass recommendation rate.
- waviness_dominant reached approximately 94.67% pass recommendation rate.
- offset_dominant was the most correctable scenario, with approximately 98.67% pass recommendation rate.
- The dominant remaining limitation was parallelism_error.

Interpretation:

This analysis moves the framework beyond simple parameter optimization. The system can identify where the available correction capability is sufficient and where it reaches its limit. The remaining difficult cases are mainly parallelism-limited, suggesting that future process improvements should focus on stronger angular or multi-point locator correction capability.

Industrial relevance:

An industrial decision-support tool should not only recommend the best available assembly parameters. It should also indicate whether the available correction capability is sufficient, whether rework is required, and which future physical correction capability should be improved. This supports the long-term product direction of a capability-aware adaptive assembly decision-support system.