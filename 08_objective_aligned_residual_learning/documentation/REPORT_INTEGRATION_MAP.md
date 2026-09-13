# Phase 8 — Thesis Report Integration Map

## Purpose

This document maps the completed Phase 8 evidence into the thesis
narrative.

Phase 8 must not be presented in the main thesis as a software-version
history.

The report should instead present the scientific progression:

    part variation
        ↓
    individualized correction need
        ↓
    mechanics-informed deterministic baseline
        ↓
    residual machine-learning augmentation
        ↓
    objective-alignment audit
        ↓
    objective-aligned optimization
        ↓
    residual-learning re-test
        ↓
    capability-aware method-selection conclusion

The main contribution remains deviation-aware, part-specific adaptive
assembly decision support.

---

# 1. Chapter 3 — Methodology

## 3.x Mechanics-Informed Baseline Correction

Introduce Least Squares as the deterministic geometry-based correction
baseline.

Explain that LSQ minimizes profile error using the available correction
basis.

Do not describe LSQ as the universally optimal controller.

Purpose in the thesis:

- physically interpretable correction baseline;
- strong deterministic reference;
- basis for evaluating whether ML provides additional information.

---

## 3.x Residual Machine-Learning Correction

Introduce the hybrid principle:

    u_hybrid = u_LSQ + delta_u_ML

Explain that machine learning does not replace the mechanics-informed
controller.

Instead, it predicts a state-dependent residual correction.

Random Forest is the main Version 3.3 residual learner.

Gradient Boosting is later introduced as an independent model-family
check in the objective-aligned residual-learning audit.

---

## 3.x Composite Quality Objective

Define the quality objective:

    Q =
        0.30 * mean_gap
      + 0.30 * max_gap
      + 0.30 * parallelism_error
      + 0.10 * RMS

Explain clearly that minimizing geometric squared error and minimizing Q
are not necessarily equivalent objectives.

This distinction motivates the objective-alignment analysis.

---

## 3.x Objective-Aligned Composite-Q Controller

Introduce Composite-Q as a deterministic benchmark that directly
minimizes the actual project quality objective.

Composite-Q uses multi-start bounded L-BFGS-B.

Starting points:

- zero correction;
- six correction-space face centers;
- LSQ correction.

Important wording:

Composite-Q is a project-specific objective-aligned deterministic
benchmark.

It is not claimed to be a universally optimal assembly algorithm.

---

## 3.x Objective-Aligned Residual Learning

Introduce the final methodological audit:

    u_final = u_CompositeQ + delta_u_ML

Research question:

> After objective mismatch is removed through direct optimization of the
> actual quality objective, does any useful state-dependent residual
> correction remain for machine learning?

Models:

- Random Forest;
- Gradient Boosting.

Residual target:

    delta_u =
        u_best_verified_reference
        -
        u_CompositeQ

The best verified reference is selected between:

- Composite-Q;
- higher-budget finite Differential Evolution.

The higher-budget DE solution is a numerical reference, not a claimed
global optimum.

---

# 2. Chapter 4 — Experimental and Validation Methodology

## Data Separation

Report the Phase 8 populations:

- training: 300 assemblies;
- development: 100 assemblies;
- final: 300 assemblies;
- 10 sequential component decisions per assembly.

Seeds:

- training: 810801;
- development: 810802;
- final: 810803.

The final population must be described as untouched until architecture
lock.

---

## Feature Definition

Report:

- 26 controller-observable features;
- same observable/controller-known feature concept as the previous
  hybrid framework;
- no generator-privileged hidden deviation amplitudes.

---

## Residual Target Generation

Explain:

1. calculate Composite-Q;
2. calculate higher-budget DE reference;
3. retain DE only when it genuinely improves evaluated Q;
4. otherwise retain Composite-Q;
5. construct residual relative to Composite-Q.

This prevents training toward a numerically inferior reference.

---

## Model Evaluation

Report both:

- prediction-level evaluation;
- closed-loop sequential controller evaluation.

Prediction-level metrics alone are insufficient because the residual
targets are highly sparse.

---

## Final Evaluation Protocol

Explain that:

- model architecture was locked after development;
- no tuning was performed after observing final results;
- the final 300 assemblies were evaluated once.

---

## Statistical Method

Three predefined paired comparisons:

H1:
Composite-Q + RF vs Composite-Q

H2:
Composite-Q + GB vs Composite-Q

H3:
Composite-Q + RF vs Composite-Q + GB

Report:

- paired mean difference;
- paired bootstrap 95% confidence interval;
- Wilcoxon signed-rank test;
- paired standardized effect;
- Holm correction.

Emphasize:

statistical significance and practical engineering magnitude are
interpreted separately.

---

# 3. Chapter 5 — Results

The results should be presented as a scientific progression rather than
by software version.

---

## 5.x Mechanics-Informed and Hybrid Baseline Results

Present the previously validated results:

- LSQ mean quality: 0.451750;
- LSQ + Random Forest hybrid: 0.388839;
- hybrid improvement over LSQ: approximately 13.93%;
- hybrid win rate over LSQ: 94.67%.

Interpretation:

The residual learner captured useful state-dependent information beyond
LSQ.

Do not stop the analysis here.

---

## 5.x Objective-Alignment Result

Present:

- Composite-Q mean quality: 0.337318;
- LSQ mean quality: 0.451750;
- hybrid mean quality: 0.388839.

Composite-Q:

- improves over LSQ by approximately 25.33%;
- beats the LSQ + RF hybrid on all tested final assemblies in the
  post-freeze audit.

Interpretation:

A substantial part of the apparent ML advantage relative to LSQ is
associated with limitations of the LSQ objective rather than proof that
ML is intrinsically superior to direct objective-aligned optimization.

---

## 5.x Remaining Residual Opportunity

Report the Phase 8 target-generation result.

Approximately 99% of residual targets are zero.

Training population:

- 3000 decisions;
- higher-budget DE improves Composite-Q on 36 decisions;
- improvement frequency: 1.2%;
- mean gain across all decisions: approximately 3.09e-08;
- mean gain among positive cases: approximately 2.58e-06;
- maximum gain: approximately 3.30e-05.

Interpretation:

Very little exploitable correction opportunity remains after
Composite-Q.

---

## 5.x Residual Model Performance

Report that both Random Forest and Gradient Boosting perform worse than
the zero-residual predictor on all three residual targets during
development.

This should be connected directly to the sparsity and very small
magnitude of the remaining residual.

---

## 5.x Development Closed-Loop Results

Report:

| Method | Mean final quality |
|---|---:|
| Composite-Q | 0.322256637 |
| Composite-Q + RF | 0.322260003 |
| Composite-Q + GB | 0.322257696 |

Interpretation:

Neither ML augmentation improves Composite-Q during development.

---

## 5.x Independent Final Results

Report:

| Method | Mean final quality |
|---|---:|
| Composite-Q | 0.330214218660 |
| Composite-Q + RF | 0.330217423617 |
| Composite-Q + GB | 0.330217106479 |

Win rates against Composite-Q:

- RF: 22.67%;
- GB: 17.33%.

Loss rates:

- RF: 77.33%;
- GB: 82.67%.

---

## 5.x Statistical Results

### H1 — RF vs Composite-Q

Mean difference:

    +3.204956994579e-06

95% bootstrap CI:

    [1.530904240173e-06,
     5.278192707553e-06]

Holm-adjusted p-value:

    2.383219459382e-21

Effect:

    0.189031

Relative mean degradation:

    0.00097057%

---

### H2 — GB vs Composite-Q

Mean difference:

    +2.887819172499e-06

95% bootstrap CI:

    [1.009548433217e-06,
     5.815019810334e-06]

Holm-adjusted p-value:

    1.254681480650e-25

Effect:

    0.130435

Relative mean degradation:

    0.00087453%

---

### H3 — RF vs GB

Mean difference:

    +3.171378220799e-07

95% bootstrap CI:

    [-2.414123385878e-06,
      2.828575529598e-06]

Holm-adjusted p-value:

    0.699717

Interpretation:

No statistically significant difference between RF and GB.

---

# 4. Chapter 6 — Discussion

The central discussion should not be:

> machine learning failed.

Instead:

> the value of machine learning depends strongly on the quality of the
> deterministic baseline and the alignment between the optimization
> objective and the actual engineering objective.

The evidence supports three stages of interpretation.

### Stage 1

LSQ + RF improves LSQ.

Therefore the residual model captures useful individualized,
state-dependent information relative to the LSQ baseline.

### Stage 2

Composite-Q substantially outperforms LSQ and LSQ + RF.

Therefore objective alignment is a dominant methodological factor.

### Stage 3

RF and GB do not improve Composite-Q.

Therefore, once the actual low-dimensional quality objective is directly
and inexpensively optimized, little useful residual opportunity remains
for the tested ML models.

This is the strongest final method-selection conclusion.

---

# 5. Chapter 7 — Limitations

Phase 8 does not prove that machine learning is unnecessary for adaptive
assembly in general.

The conclusion is restricted to the investigated environment:

- synthetic data;
- controlled deviation model;
- three correction parameters;
- low-dimensional bounded correction space;
- explicitly known quality function;
- inexpensive direct quality evaluation;
- no measurement noise;
- no physical contact/compliance model;
- no real production data;
- no high-dimensional actuator system;
- no online computational constraints.

These limitations are essential to the interpretation.

---

# 6. Chapter 8 — Conclusion and Outlook

The final conclusion should state:

> Deviation-aware individualized correction is beneficial, but the most
> appropriate recommendation method depends on the structure of the
> assembly problem. In the investigated controlled environment, direct
> objective-aligned optimization provides the strongest tested solution.
> Machine learning provides measurable value relative to a mechanics-
> informed LSQ baseline, but this additional benefit disappears when the
> actual quality objective becomes directly and efficiently optimizable.

Future ML value should be investigated when:

- physical response evaluation becomes expensive;
- measured data contain noise and uncertainty;
- system behaviour is only partially known;
- correction dimensionality increases;
- high-fidelity simulation is required;
- direct online optimization becomes too slow;
- digital-twin or closed-loop learning is introduced.

This creates the bridge to a future master's thesis on measurement-driven,
uncertainty-aware, mechanics-informed adaptive assembly.
