# Objective-Aligned Residual Learning Experiment Protocol

## 1. Purpose

This experiment evaluates whether machine-learning-based residual correction
can provide additional assembly-quality improvement when applied on top of
an objective-aligned deterministic Composite-Q controller.

The experiment is conducted as a separate extension of the completed and
frozen Version 3.3 framework and the subsequent post-freeze method-validation
work. Existing validated results are not modified.

## 2. Research Questions

### RQ1
Can state-dependent residual machine learning provide additional quality
improvement beyond the Composite-Q controller?

### RQ2
Does the result depend on the selected machine-learning model family when
Random Forest and Gradient Boosting are evaluated under identical conditions?

## 3. Methods

The following controllers are compared:

1. Composite-Q
2. Composite-Q with Random Forest residual correction
3. Composite-Q with Gradient Boosting residual correction

Composite-Q remains the deterministic base controller for both learned
approaches.

## 4. Residual Target

For each training decision, the Composite-Q correction is calculated first.

A higher-budget finite numerical search is then used to obtain an additional
reference candidate. Both candidates are evaluated using the same composite
quality objective.

The better verified candidate is retained as the reference correction.

The residual learning target is defined as:

delta_u = u_reference - u_CompositeQ

If the higher-budget numerical search does not improve upon Composite-Q,
the Composite-Q correction remains the reference.

The finite numerical reference is not interpreted as proof of a global
mathematical optimum.

## 5. Input Information

Both machine-learning models use the same controller-observable feature set.

Hidden deviation-generation parameters or other privileged simulator
information are excluded.

## 6. Data Separation

Separate datasets are used for:

- training;
- development;
- final evaluation.

Training and development data are used for model construction and model
selection.

The final evaluation population is not used for feature selection,
hyperparameter tuning, architecture modification, or target-definition
changes.

## 7. Machine-Learning Models

Two nonlinear tabular regression approaches are evaluated:

- Random Forest;
- Gradient Boosting.

Both methods receive identical input features and residual targets.

The objective is not to establish a universally optimal machine-learning
algorithm, but to assess whether objective-aligned residual learning provides
additional value and whether the result depends substantially on the model
family.

## 8. Correction Constraints

All evaluated controllers use the same correction variables and correction
bounds as the validated Version 3.3 assembly model.

Machine-learning-based controllers are not permitted additional correction
capability.

Predicted residual corrections are constrained such that the final recommended
correction remains within the defined bounds.

## 9. Final Statistical Comparisons

Three primary paired comparisons are defined:

H1: Composite-Q + Random Forest versus Composite-Q.

H2: Composite-Q + Gradient Boosting versus Composite-Q.

H3: Composite-Q + Random Forest versus Composite-Q + Gradient Boosting.

These comparisons form a separate statistical family from the previously
completed post-freeze hypotheses.

For each comparison, the analysis will report:

- mean quality;
- median quality;
- paired mean difference;
- paired median difference;
- assembly-level win rate;
- paired bootstrap 95% confidence interval;
- Wilcoxon signed-rank test;
- paired standardized effect size;
- Holm-adjusted significance result.

## 10. Interpretation

The experiment is considered informative regardless of whether residual
machine learning improves Composite-Q.

A positive result would indicate that assembly-specific residual information
remains exploitable after objective-aligned deterministic optimization.

A neutral result would indicate that Composite-Q already captures most of the
available correction benefit within the investigated system.

A negative result would indicate that adding learned residual correction can
degrade an already strong objective-aligned deterministic solution.

No result will be interpreted as evidence of industrial production readiness
or universal superiority of a particular optimization or machine-learning
method.

## 11. Preservation of Previous Work

The completed Version 3.3 framework and existing post-freeze method-validation
results remain unchanged.

All Phase 8 code, generated data, trained models, results, and documentation
are stored only within the dedicated Phase 8 directory.