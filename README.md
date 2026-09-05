# Machine Learning-Based Optimization of Assembly Parameters Considering Part Variations

## Project Overview

This repository contains the computational implementation developed for the
FAU project thesis:

**Machine Learning-Based Optimization of Assembly Parameters Considering
Part Variations**

The project investigates deviation-aware adaptive assembly parameter
recommendation for sequential multi-component assembly.

The final framework combines:

- synthetic geometric part-variation modelling
- sequential assembly-state propagation
- machine-learning surrogate modelling
- part-specific correction recommendation
- correction-capability assessment
- Bayesian Optimization investigation
- robustness and sensitivity evaluation
- pre-decision assembly-difficulty estimation
- advisory decision-support analysis

The current implementation is a controlled computational feasibility study
based on synthetic one-dimensional geometric profiles. It is not an
industrially validated production system.


## Final Validated Version

The final validated technical baseline is Version 3.2.

The main implementation is located in:

`03_v3_2_sequential_multicomponent`


The sequential assembly state is conceptually represented as:

`S_k = S_(k-1) + D_k + F_(b,k) + C_k`

where:

- `S_(k-1)` = accumulated assembly state before the current operation
- `D_k` = incoming component deviation
- `F_(b,k)` = batch/process disturbance
- `C_k` = applied assembly correction


## Assembly Corrections

Three effective correction parameters are considered:

- `z_adj`
- `theta_adj`
- `locator_offset`

The parameters represent effective assembly-adjustment capabilities within
the simplified computational model.


## Quality Evaluation

Assembly quality is evaluated using geometric indicators including:

- mean gap
- maximum gap
- parallelism error
- RMS deviation

A weighted quality score is used to compare candidate corrections.


## Machine-Learning Surrogate

Several regression approaches were investigated during model development.

Random Forest was selected as the main quality surrogate.

The final surrogate uses both scalar geometric descriptors and information
derived from the assembly profile.

The surrogate is used primarily for quality prediction and candidate
ranking.


## Structured-20 Recommendation

The final operational recommendation method is the Structured-20 controller.

For each assembly decision:

1. the current assembly state is evaluated;
2. the incoming component and process information are considered;
3. a structured set of feasible correction candidates is generated;
4. the Random Forest surrogate predicts candidate quality;
5. the best-ranked candidate is selected;
6. correction-capability utilization is evaluated.

Structured-20 is the primary correction recommendation method in the final
validated system.


## Bayesian Optimization

Bayesian Optimization was investigated as part of the project.

Early sequential closed-loop evaluation showed that strong offline surrogate
performance alone did not guarantee successful optimization.

A structured warm-start Bayesian Optimization approach was therefore
investigated.

Final comparisons showed that Bayesian refinement provided only negligible
system-level improvement relative to Structured-20.

Bayesian Optimization is therefore retained as an investigated optional
refinement rather than the mandatory operational controller.


## Independent Validation

The final independent controller validation contains:

- 300 finished assemblies
- 30 independent batch realizations
- 6 batch/process conditions
- 5 sequential components per assembly
- 1500 sequential assembly decisions

Final Structured-20 results include approximately:

- zero-correction mean final quality: `0.9505`
- Structured-20 mean final quality: `0.2386`
- mean improvement relative to zero correction: `74.90%`
- assembly-level win rate relative to zero correction: `99.33%`
- exact-best candidate ranking: `86.27%`
- top-3 candidate ranking: `99.47%`

Additional robustness and sensitivity analyses were performed for:

- independent random seeds
- deviation magnitude
- available correction capability
- quality-score weighting


## Correctability Extension

A separate extension is located inside:

`03_v3_2_sequential_multicomponent/04_correctability_extension`

The extension investigates whether difficult assembly decisions can be
identified before the current correction is applied.

Only pre-decision information is used for the difficulty predictor.

Repeated assembly-wise validation showed that combining:

- incoming component information
- accumulated assembly-state information
- process/batch information

provides meaningful predictive information about future correction
difficulty.

The resulting difficulty signal is interpreted as an advisory
decision-support signal rather than an autonomous production-routing rule.


## Project Structure

The repository contains earlier development stages together with the final
validated implementation.

```text
FAU_Thesis_Adaptive_Assembly/
│
├── README.md
├── requirements.txt
│
├── 01_v2_locked_baseline/
├── 02_v3_general_framework/
├── 03_results_archive/
├── 03_v3_2_sequential_multicomponent/
│   ├── src/
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── freeze/
│   ├── notes/
│   └── 04_correctability_extension/
│       ├── src/
│       └── results/
│
└── 04_documents/

## Final Scientific Audits

After completion of the main V3.2 validation, additional scientific audits
were performed without modifying or retraining the frozen V3.2 controller.

### Group-Held-Out Difficulty Validation

The pre-decision difficulty predictor was evaluated under a stricter
batch/process-signature held-out setting.

Results:

- balanced accuracy: approximately 79.98%
- attention recall: approximately 72.65%
- attention precision: approximately 80.04%
- ROC AUC: approximately 0.871

The result indicates that meaningful difficulty-prediction capability
remains when complete process-context groups are excluded from training.


### Deterministic Engineering Baseline

A bounded least-squares geometric correction baseline was evaluated using
the same three available correction modes.

Mean final quality:

- zero correction: 0.9505
- deterministic least squares: 0.2202
- Structured-20: 0.2386

The deterministic strategy achieved approximately 76.83% improvement
relative to zero correction, compared with approximately 74.90% for
Structured-20.

This result shows that the simplified additive geometric model is highly
correctable using direct model-based compensation.


### Simulator-Direct Achievable-Correction Reference

A simulator-direct numerical optimization was used as an approximate
reference for the achievable greedy correction performance under the
defined correction limits.

Mean final quality:

- direct simulator reference: approximately 0.1702
- deterministic least squares: approximately 0.2202
- Structured-20: approximately 0.2386

Relative to the direct-simulator improvement:

- Structured-20 captured approximately 91.23%
- deterministic least squares captured approximately 93.59%

The direct-simulator result is an approximate numerical reference and is
not interpreted as mathematical proof of a global optimum.


### Observable-Information Audit

Difficulty prediction was evaluated using different information levels.

Mean balanced accuracy:

- privileged synthetic-generator information: approximately 71.15%
- observable geometry/state information: approximately 76.59%
- full combined information: approximately 83.92%

Observable geometric and accumulated-state information therefore retained
meaningful predictive value without requiring the explicit synthetic
deviation decomposition.


### Two-Step Look-Ahead Diagnostic

A two-step receding-horizon diagnostic was compared with greedy
simulator-direct correction on 30 assemblies.

Mean final quality:

- greedy direct: approximately 0.1632
- two-step look-ahead: approximately 0.1638

Relative look-ahead benefit:

- approximately -0.36%

Mean correction utilization:

- greedy: approximately 0.165
- two-step look-ahead: approximately 0.648

No meaningful advantage from two-step look-ahead was observed under the
investigated additive sequential model.

### Audit 6 — ML Value Regions

A dedicated paired comparison was performed between the final Structured-20
ML controller and the deterministic LSQ baseline over the 300 independent
validation assemblies.

Main results:

- Structured-20 wins: 42.0%
- Deterministic LSQ wins: 58.0%
- Mean Structured-20 advantage, defined as
  `Q_LSQ - Q_Structured20`: approximately -0.0184
- 95% paired bootstrap confidence interval:
  approximately [-0.0255, -0.0114]

The confidence interval remains below zero, indicating that the deterministic
LSQ baseline has a statistically supported mean-quality advantage for the
current simplified additive assembly model.

No general trend was observed in which Structured-20 automatically becomes
superior for assemblies with higher initial quality error.

This result is treated as a method-selection finding rather than as a failure
of the ML framework.


### Audit 7 — Correction-Basis Residual Analysis

A correction-basis residual analysis was performed to investigate why the
deterministic LSQ method performs strongly under the current assembly model.

The deterministic baseline was replayed over:

- 300 assemblies
- 1500 sequential component decisions

The reconstructed deterministic results matched the frozen baseline with a
maximum absolute replay difference of approximately:

- 9.7e-17

This confirms numerical consistency with the frozen deterministic evaluation.

Main basis-analysis results:

- mean bounded residual ratio: approximately 0.496
- mean unbounded residual ratio: approximately 0.496
- mean explained profile energy: approximately 65.9%
- capability-related residual penalty: approximately 0

The bounded and unbounded solutions were effectively identical, indicating
that the strong LSQ result is not primarily caused by correction-limit
saturation.

A positive relationship was observed between correction-basis mismatch and
the relative performance of Structured-20.

Correlation between bounded residual ratio and Structured-20 advantage:

- approximately 0.266

The highest correction-basis mismatch quartile showed:

- Structured-20 win rate: approximately 60.5%
- mean Structured-20 advantage: approximately +0.0014

In contrast, Structured-20 won only approximately 21.9% of decisions in the
best correction-basis-fit quartile.

This indicates that the ML-supported controller becomes relatively more
competitive as the analytical correction basis represents the geometric state
less accurately.

The result does not establish universal ML superiority. It instead identifies
a defensible operating region in which the flexibility of the ML-supported
ranking approach becomes more relevant.


### Controller Input Realism

The final Structured-20 surrogate uses a combination of:

- assembled-state geometric features
- sampled state-profile features
- component-profile characteristics
- correction decision variables
- process/context descriptors
- synthetic deviation descriptors

Several features can be obtained directly or derived from measured geometry,
including state-gap metrics, RMS, parallelism, estimated angle, sampled profile
values, and component-profile statistics.

Other variables, including synthetic deviation amplitudes and some
batch/process descriptors, are currently available from the simulation
environment and would require estimation, calibration, preprocessing, or
replacement in a measurement-driven implementation.

Therefore, the present prototype should be interpreted as a validated
simulation-based adaptive recommendation framework rather than as a directly
sensor-deployable industrial controller.

A measurement-driven implementation would require either:

- a preprocessing/estimation pipeline that extracts the required deviation
  descriptors from measured part geometry, or
- retraining of the surrogate using only directly observable and reliably
  derived measurement features.

A standalone measurement-noise experiment was intentionally not added because
a scientifically meaningful noise analysis would first require a defined
measurement-to-feature estimation pipeline.


### Audit 8 — Final Method Trade-Off Assessment

The principal correction strategies were compared using the existing
independent validation and final-audit results.

Mean final quality:

- Zero correction: approximately 0.9505
- Deterministic LSQ: approximately 0.2202
- Structured-20: approximately 0.2386
- Selective Bayesian Optimization: approximately 0.2377
- Direct simulator numerical reference: approximately 0.1702

Improvement relative to zero correction:

- Deterministic LSQ: approximately 76.83%
- Structured-20: approximately 74.90%
- Selective Bayesian Optimization: approximately 75.00%
- Direct simulator numerical reference: approximately 82.10%

Computational indicators:

- Structured-20: 20 surrogate candidate evaluations per decision
- Direct simulator numerical reference: approximately 784.2 simulator
  evaluations per decision

The direct simulator search provides the strongest observed greedy quality
reference but requires substantially greater computational effort.

The deterministic LSQ method provides the strongest combination of
transparency, computational efficiency, and mean quality for the current
analytically tractable additive model.

Structured-20 remains the primary ML-supported recommendation architecture.
Its value lies in surrogate-based candidate ranking without requiring an exact
analytical inverse of the process response.

Selective Bayesian Optimization provides negligible additional mean-quality
improvement over Structured-20 in the independent validation and is therefore
treated as an optional refinement rather than as a mandatory online stage.

The resulting method-selection interpretation is:

- deterministic analytical methods are preferred when the correction response
  is explicitly available and adequately represents the geometry
- ML-supported recommendation becomes more relevant when analytical
  representation becomes weaker, nonlinear, expensive, uncertain, or
  measurement-driven
- direct simulator optimization is valuable as a quality reference but is
  computationally expensive for online use
- no single method is claimed to be universally optimal


### Final Audit Consistency

A final automated consistency check verified the complete scientific-audit
package after all strengthening analyses were completed.

- scientific audit areas: 8
- automated consistency checks: 60
- passed: 60
- failed: 0

Final consistency result:

`FINAL AUDIT CONSISTENCY: PASS`

The final scientific-audit results are stored in:

`05_final_scientific_audits/results`

The corresponding evaluation scripts are stored in:

`05_final_scientific_audits/src`

No additional model retraining, controller tuning, or scientific experiments
are required for the frozen technical package.
# V3.3 Nonlinear Hybrid Upgrade

## Relationship to the Frozen V3.2 Package

The results and scientific audits documented above belong to the frozen V3.2
technical package and remain preserved as the historical baseline.

The statement that no further model retraining, controller tuning, or
scientific experiments were required refers specifically to completion of the
frozen V3.2 package at that stage of the project.

A later V3.3 extension was subsequently developed in a separate folder:

 6_v3_3_nonlinear_hybrid_upgrade

V3.3 does not overwrite or invalidate V3.2. Instead, it addresses a central
limitation identified during the V3.2 scientific audits: deterministic
least-squares correction remained stronger than the ML-supported Structured-20
approach on average when the assembly response was explicitly additive and
analytically well matched to the LSQ correction basis.


## V3.3 Scientific Direction

V3.3 introduces a controlled nonlinear model-mismatch study and a hybrid
analytical-plus-ML correction architecture.

The final locked controller is:

LSQ_PLUS_ML_ALL

with the conceptual form:

u_hybrid = u_LSQ + delta_u_ML

where the residual ML model predicts corrections to:

- z adjustment
- theta adjustment
- locator offset

The deterministic LSQ solution remains the analytical baseline, while the
Random Forest residual model learns correction information not captured by
the analytical decision with respect to the nonlinear/composite-quality
problem.

The residual ML model uses 26 observable or controller-known features and does
not rely on the known synthetic generator amplitudes used to create the
deviation cases.


## V3.3 Development and Architecture Lock

Primary sequence length:

- K = 10 components

Residual-learning data:

- training: 300 assemblies, 3000 decisions
- development: 60 assemblies, 600 decisions
- train/development assembly overlap: 0

The hybrid architecture was selected using development data and formally
locked before final independent validation.

Locked architecture:

LSQ_PLUS_ML_ALL

Development mean final quality:

- LSQ: 0.409564
- LSQ + ML theta: 0.350504
- LSQ + ML z + theta: 0.347510
- LSQ + ML all: 0.346331
- finite-budget direct nonlinear reference: 0.318627

Development improvement of the locked hybrid relative to LSQ:

- 15.44%


## V3.3 Final Independent Validation

The final locked controller was evaluated on:

- 300 independently generated assemblies
- 10 sequential components per assembly
- no retraining after architecture lock
- no post-validation controller tuning
- no architecture reselection after observing final results

Mean final quality:

- zero correction: 2.105948
- deterministic LSQ: 0.451750
- locked hybrid: 0.388839
- low-budget direct nonlinear reference: 0.360594

Locked-hybrid improvement relative to LSQ:

- 13.93%

Assembly-level win rate:

- hybrid beats LSQ: 94.67%
- LSQ beats hybrid: 5.33%

Median final quality:

- LSQ: 0.401697
- hybrid: 0.343759

P95 final quality:

- LSQ: 0.899600
- hybrid: 0.726257

Mean correction utilization:

- LSQ: 0.2336
- hybrid: 0.2148

The hybrid therefore improved mean quality while also using slightly lower
mean correction utilization.


## V3.3 Final Statistical Evidence

Paired assembly-level statistics for locked hybrid minus LSQ:

- mean difference: -0.062910
- median difference: -0.056104
- 95% paired bootstrap confidence interval:
  [-0.068150, -0.057723]
- Wilcoxon signed-rank p-value:
  5.6242071e-49
- paired standardized effect:
  -1.3599

The 95% paired bootstrap interval lies entirely below zero.

Final statistical gate:

PASS

The pre-locked hybrid shows a statistically supported mean improvement over
deterministic LSQ on the final 300-assembly independent validation population.


## V3.3 Stage-Wise Behavior

The hybrid mean quality is lower than LSQ at every sequential stage from
component 1 through component 10.

Final-stage values:

- LSQ: 0.451750
- hybrid: 0.388839
- hybrid win rate: 94.67%

The advantage therefore does not arise only at the final component but is
observed throughout the sequential assembly process.


## V3.3 Robustness

The locked controller was tested without retraining across:

Nonlinearity levels:

- 0.0
- 0.5
- 1.0

Measurement-noise levels:

- 0.00 mm
- 0.01 mm
- 0.03 mm

This produces nine tested robustness regimes.

The hybrid retained a positive mean advantage over LSQ in all nine regimes.

Clean-measurement improvement:

- alpha = 0.0: 10.61%
- alpha = 0.5: 13.02%
- alpha = 1.0: 14.85%

At alpha = 1.0 and 0.03 mm measurement noise:

- hybrid improvement: 14.81%
- hybrid win rate: 96.00%

These results are interpreted as controlled robustness/generalization evidence
within the synthetic model family, not as proof of industrial measurement
robustness.


## V3.3 Sequence-Length Scaling

The same generated assemblies were evaluated using:

- K = 5
- K = 10

K = 5:

- LSQ mean quality: 0.323963
- hybrid mean quality: 0.270972
- relative hybrid improvement: 16.36%
- hybrid win rate: 94.50%

K = 10:

- LSQ mean quality: 0.435294
- hybrid mean quality: 0.373497
- relative hybrid improvement: 14.20%
- hybrid win rate: 91.00%

The first five stages of the K = 5 and K = 10 experiments were verified as
numerically identical up to floating-point precision.

Scaling gate:

PASS

The hybrid advantage therefore persists when the sequential assembly length is
increased from five to ten components.


## V3.3 Direct Numerical-Reference Stability

The direct nonlinear numerical reference was evaluated using increasing
Differential Evolution budgets.

Mean final quality:

- LOW: 0.315634
- MEDIUM: 0.292256
- HIGH: 0.291004

The additional mean improvement from MEDIUM to HIGH was:

- 0.428%

Reference stability interpretation:

PASS

The medium- and high-budget estimates are numerically stable at the mean level
within less than 1%.

The direct nonlinear reference remains a finite-budget numerical reference and
is not claimed to prove the global optimum.

The earlier 69.01% reference-improvement-capture result was calculated using
the lower-budget 300-assembly reference and is therefore not treated as a
definitive final metric without qualification.


## V3.3 Final Consistency

A dedicated V3.3 scientific consistency checker verified the completed
evidence package.

- total checks: 55
- passed: 55
- failed: 0

Final consistency result:

V3.3 FINAL CONSISTENCY: PASS


## V3.3 Final Technical Status

Core technical experimentation:

COMPLETE

Controller architecture:

LOCKED

Final independent validation:

COMPLETE

Statistical validation:

COMPLETE

Robustness analysis:

COMPLETE

Sequence-length scaling:

COMPLETE

Numerical-reference stability:

COMPLETE

Scientific consistency:

55 / 55 PASS

The authoritative V3.3 result summary is stored in:

 6_v3_3_nonlinear_hybrid_upgrade/V3_3_FINAL_RESULT_MAP.md

The corresponding scripts, data, trained model, and results are stored in:

 6_v3_3_nonlinear_hybrid_upgrade

No additional controller tuning, architecture reselection, or model retraining
is required for the completed V3.3 technical package before thesis-report
integration.

