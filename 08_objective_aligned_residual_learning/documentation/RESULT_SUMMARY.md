# Phase 8 — Objective-Aligned Residual Learning Result Summary

## 1. Purpose

Phase 8 investigated whether machine-learning-based residual correction
provides additional assembly-quality improvement when applied on top of
the objective-aligned Composite-Q deterministic controller.

The experiment was conducted after completion of the Version 3.3 hybrid
framework and the subsequent post-freeze method-validation work.

The central question was:

> Does a useful state-dependent correction residual remain after the
> assembly parameters are already optimized directly with respect to
> the composite quality objective?

Two machine-learning model families were evaluated:

- Random Forest;
- Gradient Boosting.

The completed Version 3.3 and post-freeze Phase 1-7 results were not
modified.

---

## 2. Experimental Design

Three independent Phase 8 populations were used:

- 300 training assemblies;
- 100 development assemblies;
- 300 final evaluation assemblies.

Each assembly contained 10 sequential component decisions.

The machine-learning models used the same 26 controller-observable
features as the Version 3.3 residual-learning framework.

The final Phase 8 population was not used for model training,
hyperparameter tuning, feature selection, residual-target definition,
or controller redesign.

---

## 3. Residual-Target Construction

For each training and development decision, the Composite-Q correction
was calculated first.

A higher-budget finite Differential Evolution search was then performed
using the same correction bounds and composite quality objective.

The higher-budget numerical candidate was retained only when it produced
a verified quality improvement over Composite-Q.

Otherwise, Composite-Q itself remained the reference solution.

The machine-learning target was therefore defined as:

    delta_u = u_best_reference - u_CompositeQ

This prevented the learning target from being directed toward a
numerically worse correction.

The higher-budget numerical search was treated as a finite numerical
reference and not as proof of a mathematical global optimum.

---

## 4. Remaining Correction Opportunity Beyond Composite-Q

Across the combined training and development target-generation data,
the higher-budget numerical reference improved upon Composite-Q on only
approximately 1% of component decisions.

The generated residual targets were zero for approximately 99% of
decisions.

For the 3000 training decisions specifically:

- Differential Evolution improved Composite-Q on 36 decisions (1.2%);
- mean quality gain across all decisions:
  approximately 3.09e-08;
- mean gain among the improved decisions:
  approximately 2.58e-06;
- maximum observed gain:
  approximately 3.30e-05.

Maximum observed correction differences were approximately:

- delta z: 0.00139 mm;
- delta theta: 3.58e-06 deg;
- delta locator: 0.01472 mm.

These values indicated that only a very small residual correction
opportunity remained after objective-aligned Composite-Q optimization.

---

## 5. Residual Model Performance

Random Forest and Gradient Boosting regressors were trained independently
for:

- delta z;
- delta theta;
- delta locator.

On the independent development dataset, only 6 of 1000 decisions had
non-zero residual targets.

For all three correction targets, both Random Forest and Gradient
Boosting produced larger MAE and RMSE values than the zero-residual
baseline.

All development R-squared values were negative.

Therefore, the remaining residual was not predictably captured by either
tested model family under the investigated feature representation.

Because the target distribution was highly sparse, prediction metrics
alone were not used to determine controller value. Closed-loop sequential
evaluation was performed separately.

---

## 6. Development Closed-Loop Evaluation

Mean final quality on the 100 development assemblies was:

| Method | Mean final quality |
|---|---:|
| Composite-Q | 0.322256637 |
| Composite-Q + Random Forest | 0.322260003 |
| Composite-Q + Gradient Boosting | 0.322257696 |

Both machine-learning-augmented controllers were slightly worse than
plain Composite-Q.

Assembly-level win rates against Composite-Q were:

- Random Forest: 22.00%;
- Gradient Boosting: 22.00%.

No model or controller setting was changed after observing this result.

---

## 7. Independent Final Evaluation

The locked Phase 8 controllers were evaluated once on 300 previously
unused final assemblies using final seed 810803.

Mean final quality was:

| Method | Mean final quality |
|---|---:|
| Composite-Q | 0.330214218660 |
| Composite-Q + Random Forest | 0.330217423617 |
| Composite-Q + Gradient Boosting | 0.330217106479 |

Lower values indicate better quality.

Paired mean differences were:

- Random Forest minus Composite-Q:
  +3.204956994579e-06;
- Gradient Boosting minus Composite-Q:
  +2.887819172499e-06;
- Random Forest minus Gradient Boosting:
  +3.171378220799e-07.

Assembly-level win rates were:

- Random Forest versus Composite-Q: 22.67%;
- Gradient Boosting versus Composite-Q: 17.33%;
- Random Forest versus Gradient Boosting: 51.00%.

Corresponding loss rates against Composite-Q were:

- Random Forest: 77.33%;
- Gradient Boosting: 82.67%.

---

## 8. Statistical Analysis

Three predefined paired comparisons formed a separate Phase 8
statistical family.

### H1 — Composite-Q + Random Forest versus Composite-Q

Mean paired difference:

    +3.204956994579e-06

Bootstrap 95% confidence interval:

    [1.530904240173e-06, 5.278192707553e-06]

Holm-adjusted p-value:

    2.383219459382e-21

Paired standardized effect:

    0.189031

The difference was statistically significant, with the positive
direction indicating that Random Forest was worse than Composite-Q.

The relative mean degradation was approximately 0.00097057%.

---

### H2 — Composite-Q + Gradient Boosting versus Composite-Q

Mean paired difference:

    +2.887819172499e-06

Bootstrap 95% confidence interval:

    [1.009548433217e-06, 5.815019810334e-06]

Holm-adjusted p-value:

    1.254681480650e-25

Paired standardized effect:

    0.130435

The difference was statistically significant, with Gradient Boosting
again performing slightly worse than Composite-Q.

The relative mean degradation was approximately 0.00087453%.

---

### H3 — Composite-Q + Random Forest versus Composite-Q + Gradient Boosting

Mean paired difference:

    +3.171378220799e-07

Bootstrap 95% confidence interval:

    [-2.414123385878e-06, 2.828575529598e-06]

Holm-adjusted p-value:

    0.699717

Paired standardized effect:

    0.013923

No statistically significant difference between Random Forest and
Gradient Boosting was detected.

---

## 9. Interpretation

Phase 8 provides no evidence that residual machine learning improves the
objective-aligned Composite-Q controller under the investigated
conditions.

The stronger numerical target search found almost no remaining correction
opportunity beyond Composite-Q, and both machine-learning model families
performed worse than the zero-residual prediction baseline during
development.

The independent final closed-loop evaluation confirmed this behaviour.

Although the differences between Composite-Q and the learned controllers
were statistically detectable, their practical magnitude was extremely
small.

Therefore, the result should not be interpreted as machine learning
strongly degrading assembly quality.

The supported conclusion is instead:

> Once the investigated low-dimensional assembly problem is optimized
> directly with respect to the actual composite quality objective,
> essentially no useful residual correction remains for the tested
> Random Forest or Gradient Boosting models to exploit.

This result strengthens the evidence that objective alignment is a
dominant methodological factor in the present controlled framework.

---

## 10. Method-Selection Implication

For the investigated problem, direct objective-aligned deterministic
optimization is preferred when:

- the quality objective is explicitly known;
- objective evaluation is computationally inexpensive;
- the correction dimension is small;
- correction bounds are known;
- the system response can be evaluated directly.

Machine-learning assistance remains relevant for future conditions in
which:

- objective evaluation is expensive;
- the physical response is incomplete or partially unknown;
- measured geometry introduces uncertainty and noise;
- correction dimensionality increases;
- high-fidelity simulation replaces the present controlled model;
- online optimization becomes computationally impractical;
- measurement-driven adaptation or digital-twin updating is required.

---

## 11. Integrity Validation

The final Phase 8 integrity validation completed 58 checks.

Results:

- passed: 58;
- failed: 0.

The checks confirmed:

- correct dataset sizes;
- distinct training, development, and final seeds;
- 26 observable features;
- no missing values;
- unique decision identifiers;
- valid trained model files;
- 300 final assemblies for each controller;
- 9000 final component-result rows;
- no correction-bound violations;
- correct H1-H3 statistical family;
- valid Holm-adjusted p-values;
- consistency between final controller means and statistical directions.

Final integrity status:

    PASS

---

## 12. Final Phase 8 Conclusion

Phase 8 closes the objective-aligned residual-learning question for the
present project configuration.

Composite-Q remained the strongest tested controller.

Neither Random Forest nor Gradient Boosting provided additional
controller-level quality improvement after objective alignment.

The negative ML result is scientifically informative because it defines
a clear boundary for the value of machine learning in the present
controlled assembly problem rather than assuming that additional learning
must improve an already strong deterministic solution.
