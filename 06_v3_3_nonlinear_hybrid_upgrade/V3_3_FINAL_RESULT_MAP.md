# V3.3 Final Result Map

## 1. Status

The V3.3 nonlinear hybrid upgrade has been completed and internally verified.

Final consistency status:

- Total checks: 55
- Passed: 55
- Failed: 0
- Final consistency result: PASS

The locked controller architecture is:

LSQ_PLUS_ML_ALL

The controller architecture was locked before final independent validation.

---

# 2. Scientific Purpose of V3.3

V3.3 extends the frozen V3.2 technical package by introducing:

- a controlled nonlinear assembly-response mismatch,
- ten-component sequential assembly as the primary configuration,
- observable-only ML input features,
- a hybrid analytical and ML residual correction architecture,
- independent closed-loop validation,
- measurement-noise and nonlinear-regime robustness analysis,
- sequence-length scaling,
- direct numerical-reference stability analysis.

V3.2 remains preserved as the historical baseline and is not overwritten.

The purpose of V3.3 is to investigate whether residual machine learning can provide additional correction information when the analytical LSQ formulation is no longer perfectly aligned with the effective nonlinear and composite-quality problem.

---

# 3. Hybrid Controller Architecture

The final locked hybrid controller is expressed as:

u_hybrid = u_LSQ + delta_u_ML

The ML residual model predicts corrections for:

- z adjustment,
- theta adjustment,
- locator offset.

Deterministic LSQ remains the analytical baseline and provides the primary structured correction action.

The residual model is implemented using Random Forest regression with:

- 26 observable or controller-known input features,
- 3 residual outputs,
- 
_jobs = 2.

The final operational feature set excludes known synthetic generator amplitudes used to create the simulated deviations.

---

# 4. Residual-Learning Dataset

The primary V3.3 sequence length is:

- K = 10 components.

The full residual-learning dataset contains:

Training set:

- 300 assemblies,
- 3000 component decisions.

Development set:

- 60 assemblies,
- 600 component decisions.

The assembly overlap between training and development sets is zero.

The residual-learning model was trained at:

- nonlinearity strength lpha = 1.0.

The operational ML feature set contains:

- 26 features.

The residual output space contains:

- 3 correction residuals.

---

# 5. Residual ML Learnability

The full residual-learning model produced the following development performance.

## Delta z

- MAE: 0.048412 mm
- R2: 0.1520
- Improvement over zero-residual baseline: 19.20%

Grouped cross-validation:

- Mean R2: 0.1581
- Improvement over zero-residual baseline: 18.64%

## Delta theta

- MAE: 0.004916 deg
- R2: 0.7821
- Improvement over zero-residual baseline: 55.37%

Grouped cross-validation:

- Mean R2: 0.7790
- Improvement over zero-residual baseline: 55.00%

## Delta locator

- MAE: 0.104279 mm
- R2: 0.0509
- Improvement over zero-residual baseline: 4.73%

Grouped cross-validation:

- Mean R2: 0.0385
- Improvement over zero-residual baseline: 4.90%

Residual learnability is therefore strongly output-dependent. The theta residual is predicted substantially more accurately than the z and locator residuals.

---

# 6. Development Architecture Selection

The architecture comparison was performed on a development population of:

- 60 assemblies,
- K = 10.

Mean final quality:

- Deterministic LSQ: 0.409564
- LSQ + ML theta: 0.350504
- LSQ + ML z + theta: 0.347510
- LSQ + ML all: 0.346331
- Finite-budget direct nonlinear reference: 0.318627

The locked hybrid architecture improved mean quality relative to LSQ by:

- 15.44%

Paired development comparisons:

ALL vs Z + THETA:

- Mean difference: -0.001180
- 95% bootstrap CI: [-0.002254, -0.000121]
- Wilcoxon p-value: 0.0378965

Z + THETA vs THETA:

- Mean difference: -0.002994
- 95% bootstrap CI: [-0.005737, -0.000173]
- Wilcoxon p-value: 0.0242812

The final architecture was locked as:

LSQ_PLUS_ML_ALL

The lock was completed before the final independent validation population was evaluated.

---

# 7. Final Independent Validation

The final locked controller was evaluated on:

- 300 independently generated assemblies,
- 10 sequential components per assembly.

No model retraining, architecture reselection, or post-validation controller tuning was performed after the architecture lock.

Mean final quality:

- Zero correction: 2.105948
- Deterministic LSQ: 0.451750
- Locked hybrid: 0.388839
- Low-budget direct nonlinear reference: 0.360594

The locked hybrid reduced mean quality score relative to LSQ by:

- 13.93%

Assembly-level win rate:

- Hybrid beats LSQ: 94.67%
- LSQ beats hybrid: 5.33%

Median final quality:

- LSQ: 0.401697
- Hybrid: 0.343759

P95 final quality:

- LSQ: 0.899600
- Hybrid: 0.726257

Mean correction utilization:

- LSQ: 0.2336
- Hybrid: 0.2148

The improvement in final assembly quality is therefore not associated with a higher mean correction utilization.

---

# 8. Final Independent Statistical Evidence

The paired assembly-level comparison between locked hybrid and LSQ produced:

- Mean paired difference: -0.062910
- Median paired difference: -0.056104
- 95% paired bootstrap CI: [-0.068150, -0.057723]
- Wilcoxon signed-rank p-value: 5.6242071e-49
- Paired standardized effect: -1.3599

The paired 95% bootstrap confidence interval lies entirely below zero.

Final statistical gate:

PASS

The pre-locked V3.3 hybrid therefore shows a statistically supported mean improvement over deterministic LSQ on the 300-assembly independent validation population.

---

# 9. Stage-Wise Final Validation

The hybrid mean quality is lower than LSQ at every sequential stage from 1 through 10.

Stage 1:

- LSQ: 0.167228
- Hybrid: 0.134659
- Hybrid win rate: 93.33%

Stage 2:

- LSQ: 0.217929
- Hybrid: 0.173784
- Hybrid win rate: 96.33%

Stage 3:

- LSQ: 0.266781
- Hybrid: 0.219690
- Hybrid win rate: 93.00%

Stage 4:

- LSQ: 0.292273
- Hybrid: 0.242979
- Hybrid win rate: 94.33%

Stage 5:

- LSQ: 0.331267
- Hybrid: 0.276250
- Hybrid win rate: 93.00%

Stage 6:

- LSQ: 0.358071
- Hybrid: 0.304653
- Hybrid win rate: 94.33%

Stage 7:

- LSQ: 0.379310
- Hybrid: 0.321251
- Hybrid win rate: 93.33%

Stage 8:

- LSQ: 0.413825
- Hybrid: 0.349727
- Hybrid win rate: 95.33%

Stage 9:

- LSQ: 0.430717
- Hybrid: 0.367947
- Hybrid win rate: 95.00%

Stage 10:

- LSQ: 0.451750
- Hybrid: 0.388839
- Hybrid win rate: 94.67%

The hybrid advantage therefore persists throughout the ten-component sequential assembly and is not restricted to the final stage.

---

# 10. Nonlinearity and Measurement-Noise Robustness

The robustness analysis considered:

Nonlinearity strengths:

- 0.0
- 0.5
- 1.0

Measurement-noise standard deviations:

- 0.00 mm
- 0.01 mm
- 0.03 mm

This yields nine tested robustness configurations.

The locked hybrid shows a positive mean improvement relative to LSQ in all nine tested configurations.

Clean-measurement results:

At alpha = 0.0:

- Hybrid improvement: 10.61%
- Hybrid win rate: 77.33%

At alpha = 0.5:

- Hybrid improvement: 13.02%
- Hybrid win rate: 86.00%

At alpha = 1.0:

- Hybrid improvement: 14.85%
- Hybrid win rate: 95.33%

At alpha = 1.0 and 0.03 mm measurement noise:

- Hybrid improvement: 14.81%
- Hybrid win rate: 96.00%

The residual Random Forest was trained at alpha = 1.0 using clean simulated measurements. Performance at other nonlinearities and measurement-noise levels therefore represents robustness and generalization within the defined synthetic model family.

The alpha = 0 result also shows a positive hybrid advantage. This is consistent with the fact that deterministic LSQ minimizes a profile least-squares objective, whereas the reported assembly quality score is a separate composite objective involving mean gap, maximum gap, parallelism error, and RMS deviation.

---

# 11. Sequence-Length Scaling

The sequence-length analysis compares:

- K = 5
- K = 10

using common generated assemblies, with the K = 5 case corresponding to the exact first-five-component prefix of the K = 10 case.

## K = 5

- Zero mean quality: 1.509606
- LSQ mean quality: 0.323963
- Hybrid mean quality: 0.270972
- Relative hybrid improvement: 16.36%
- Hybrid win rate: 94.50%

## K = 10

- Zero mean quality: 2.083897
- LSQ mean quality: 0.435294
- Hybrid mean quality: 0.373497
- Relative hybrid improvement: 14.20%
- Hybrid win rate: 91.00%

Prefix consistency:

- Maximum LSQ stage-5 difference: 0.0
- Maximum hybrid stage-5 difference: approximately 4.4e-15

Scaling result:

PASS

The hybrid advantage persists when the sequential assembly length increases from five to ten components.

The comparison demonstrates persistence of the hybrid advantage and does not imply that increasing sequence length inherently favors machine learning.

---

# 12. Direct Nonlinear Reference Stability

The direct nonlinear numerical reference was evaluated using increasing Differential Evolution search budgets.

LOW:

- maxiter = 8
- popsize = 5
- Mean final quality: 0.315634

MEDIUM:

- maxiter = 18
- popsize = 7
- Mean final quality: 0.292256

HIGH:

- maxiter = 30
- popsize = 10
- Mean final quality: 0.291004

The additional improvement from MEDIUM to HIGH is:

- 0.428%

Reference stability result:

PASS

The low-budget reference is not fully stabilized, whereas medium- and high-budget references provide closely aligned mean estimates.

The direct nonlinear reference is therefore interpreted as a finite-budget numerical quality benchmark rather than proof of a global optimum.

The previously calculated 69.01% reference-improvement-capture value was based on the low-budget 300-assembly reference and is therefore not treated as a definitive final metric.

---

# 13. Final Consistency Check

A dedicated V3.3 consistency checker verified the completed evidence package.

- Total checks: 55
- Passed: 55
- Failed: 0

Final result:

V3.3 FINAL CONSISTENCY: PASS

The validated V3.3 evidence package is internally consistent across:

- architecture lock,
- data sizes,
- feature counts,
- model dimensions,
- independent validation,
- paired statistics,
- stage-wise behavior,
- robustness analysis,
- sequence scaling,
- numerical-reference stability.

---

# 14. Final Technical Interpretation

Within the controlled synthetic nonlinear sequential-assembly model developed in this project, an observable-feature-based residual Random Forest combined with bounded deterministic least-squares correction provides a statistically supported improvement in assembly quality relative to deterministic LSQ alone.

The analytical LSQ controller provides the directly representable correction component, while the residual ML model supplies individualized correction adjustments associated with nonlinear response mismatch and the composite quality objective.

On 300 independent ten-component assembly sequences, the locked hybrid reduced mean final quality score from:

- 0.451750 to 0.388839,

corresponding to:

- 13.93% relative improvement,

with:

- 94.67% assembly-level wins.

The advantage is present across all ten sequential stages, all nine tested nonlinear/noise regimes, and both five- and ten-component sequence lengths.

---

# 15. Scientific Limitations

The V3.3 results remain subject to the following limitations:

- synthetic rather than measured industrial geometry,
- one-dimensional geometric representation,
- phenomenological nonlinear response,
- absence of validated contact, compliance, friction, clamping, welding, thermal, and elastic deformation physics,
- prototype quality-score weighting,
- effective correction parameters that are not yet mapped to specific industrial actuators,
- simulated measurement noise,
- finite-budget numerical reference,
- sequence scaling demonstrated only up to K = 10,
- no industrial real-time deployment validation.

These limitations define the current scope of the controlled comparative study and identify the principal directions for further development.

---

# 16. Relation to V3.2

The frozen V3.2 package established a strong baseline for adaptive assembly recommendation and demonstrated that ML-supported structured candidate ranking can generalize.

The V3.2 scientific audits also showed that deterministic LSQ remained stronger on average in the analytically matched additive model.

V3.3 addresses this limitation explicitly through a hybrid analytical-plus-residual-ML architecture rather than by weakening the deterministic baseline.

The final technical progression is therefore:

V3.2:
- analytically matched additive assembly model,
- deterministic LSQ stronger than Structured-20 on average,
- ML value primarily associated with flexible candidate ranking under weaker analytical representation.

V3.3:
- controlled nonlinear model mismatch,
- observable-only feature design,
- deterministic LSQ retained as the analytical base action,
- residual Random Forest provides individualized correction adjustments,
- locked hybrid independently validated against LSQ.

---

# 17. Technical Status

Core V3.3 technical experimentation:

COMPLETE

Controller architecture:

LOCKED

Final independent validation:

COMPLETE

Final statistical analysis:

COMPLETE

Robustness analysis:

COMPLETE

Sequence-length scaling:

COMPLETE

Direct numerical-reference stability:

COMPLETE

Final consistency:

55 / 55 PASS

The V3.3 technical package is complete and frozen for thesis-report integration and final submission documentation.
