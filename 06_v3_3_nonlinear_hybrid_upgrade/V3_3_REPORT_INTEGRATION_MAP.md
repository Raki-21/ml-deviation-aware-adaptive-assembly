# V3.3 Report Integration Map

## Purpose

This file defines how the completed V3.3 technical package is integrated
into the project thesis.

The frozen V3.2 package remains the historical baseline.

V3.3 is the final nonlinear hybrid extension and provides the primary
technical evidence for the final thesis contribution.

---

# A. Methodology

## A1. Sequential Assembly Model

Use:
- K = 10 components as primary V3.3 configuration
- K = 5 as reference/scaling configuration
- 101-point one-dimensional geometric profile
- 1000 mm profile length
- sequential state propagation

Explain:

S_k = S_(k-1) + D_k + F_(b,k) + C_k

and clearly distinguish:
- incoming component deviation,
- accumulated assembly state,
- fixture/process disturbance,
- correction action.

V3.2 provides the original additive formulation.

V3.3 adds a controlled nonlinear correction-response mismatch.

Relevant code:
- src/nonlinear_assembly_model.py
- src/v33_config.py

---

## A2. Correction Parameters and Capability

Correction variables:

- z adjustment
- theta adjustment
- locator offset

Limits:

- z: ±2.5 mm
- theta: ±1.2 deg
- locator: ±1.0 mm

These parameters are effective adjustment variables and are not claimed
to represent specific production actuators directly.

---

## A3. Quality Function

Quality score consists of:

- mean gap
- maximum gap
- parallelism error
- RMS deviation

Prototype weights:

- mean gap: 0.30
- maximum gap: 0.30
- parallelism: 0.30
- RMS: 0.10

State clearly that these are prototype engineering weights and are not
industrially calibrated quality weights.

---

## A4. Deterministic LSQ Baseline

Deterministic least-squares correction is an important analytical baseline.

The correction basis contains:

- constant z basis
- angular theta basis
- localized locator basis

LSQ solves a bounded linear least-squares correction problem.

Important thesis positioning:

V3.2 scientific audits showed that deterministic LSQ was stronger than the
ML-supported Structured-20 method on average in the analytically matched
additive model.

This result motivated V3.3 rather than being hidden.

---

## A5. Controlled Nonlinear Extension

V3.3 introduces a controlled phenomenological nonlinear correction response.

Important wording:

The nonlinear formulation is used as a controlled model-mismatch experiment.

Do NOT describe it as:
- validated industrial contact physics,
- high-fidelity structural mechanics,
- real compliance/friction behavior.

Nonlinearity strength alpha = 0 reproduces the linear V3.2 correction response.

Primary residual-learning model uses:
alpha = 1.0.

Existing figures:

- figures/v33_lsq_nonlinearity_gate_K5.png
- figures/v33_lsq_nonlinearity_gate_K10.png

---

## A6. Hybrid Residual-Learning Architecture

Final concept:

u_hybrid = u_LSQ + delta_u_ML

Residual outputs:

- delta z
- delta theta
- delta locator

Final ML model:

Random Forest regression

Input features:

26 observable/controller-known features

Important contribution:

The analytical controller provides the directly representable correction.

The ML model learns residual correction information associated with the
nonlinear/composite-quality mismatch.

---

## A7. Observable-Feature Design

The final V3.3 operational feature set excludes known synthetic generator
amplitudes.

Inputs include:

- state quality metrics
- signed geometric measures
- state-profile samples
- component-profile statistics
- deterministic LSQ correction
- correction utilization
- sequential component index

This is more deployment-realistic than the earlier V3.2 surrogate feature set.

---

## A8. Residual-Learning Dataset

Training:

- 300 assemblies
- 3000 decisions

Development:

- 60 assemblies
- 600 decisions

Train/development assembly overlap:

0

Targets are finite-budget nonlinear-reference residual corrections relative
to deterministic LSQ.

---

## A9. Architecture Selection and Lock

Development comparison:

LSQ:
0.409564

LSQ + ML theta:
0.350504

LSQ + ML z + theta:
0.347510

LSQ + ML all:
0.346331

Direct nonlinear reference:
0.318627

Locked architecture:

LSQ_PLUS_ML_ALL

The architecture was locked before final independent validation.

This chronology is scientifically important.

---

## A10. Validation Design

Final independent validation:

- 300 new assemblies
- K = 10
- no retraining
- no post-validation tuning
- no architecture reselection

Primary comparison:

Locked hybrid vs deterministic LSQ.

Statistical evaluation includes:

- paired mean difference
- paired median difference
- 95% paired bootstrap confidence interval
- Wilcoxon signed-rank test
- paired standardized effect
- assembly-level win rate
- P95 quality
- stage-wise behavior

---

# B. Results

## B1. Final Independent Controller Comparison

Primary table:

results/final_thesis_tables/
table_v33_final_independent_method_comparison.csv

Primary figure:

figures/final_thesis_figures/
v33_final_quality_distribution.png

Report:

Zero:
2.105948

LSQ:
0.451750

Locked hybrid:
0.388839

Relative hybrid improvement vs LSQ:

13.93%

Hybrid beats LSQ:

94.67%

Do not call the low-budget direct numerical reference a global optimum.

---

## B2. Paired Independent Statistical Evidence

Primary table:

table_v33_final_paired_statistics.csv

Primary figures:

- v33_paired_lsq_vs_hybrid.png
- v33_paired_improvement_distribution.png

Report:

Mean hybrid - LSQ:
-0.062910

Median:
-0.056104

95% bootstrap CI:
[-0.068150, -0.057723]

Wilcoxon p:
5.6242071e-49

Paired standardized effect:
-1.3599

Interpretation:

The independent evidence supports a systematic hybrid advantage over LSQ
within the evaluated synthetic nonlinear assembly model.

---

## B3. Stage-Wise Sequential Behavior

Primary table:

table_v33_stagewise_K10.csv

Primary figure:

v33_stagewise_quality_K10.png

Main result:

Hybrid mean quality is lower than LSQ at all 10 sequential stages.

Final stage:

LSQ:
0.451750

Hybrid:
0.388839

Hybrid wins:
94.67%

---

## B4. Correction Utilization

Primary table:

table_v33_correction_utilization.csv

Report:

Mean LSQ utilization:
0.2336

Mean hybrid utilization:
0.2148

Interpretation:

Improved hybrid quality is not obtained by simply using larger correction
magnitudes on average.

---

## B5. Robustness Regime

Primary table:

table_v33_robustness_regime.csv

Primary figure:

v33_robustness_regime_heatmap.png

Tested:

alpha:
0.0, 0.5, 1.0

measurement noise:
0.00, 0.01, 0.03 mm

Total:
9 configurations

Hybrid mean improvement is positive in all 9 regimes.

Clean measurements:

alpha 0.0:
10.61%

alpha 0.5:
13.02%

alpha 1.0:
14.85%

At alpha 1.0 and 0.03 mm noise:

14.81% improvement
96.00% win rate

---

## B6. Sequence-Length Scaling

Primary table:

table_v33_sequence_length_scaling.csv

Primary figure:

v33_sequence_length_scaling.png

K = 5:

LSQ:
0.323963

Hybrid:
0.270972

Improvement:
16.36%

Win rate:
94.50%

K = 10:

LSQ:
0.435294

Hybrid:
0.373497

Improvement:
14.20%

Win rate:
91.00%

Interpretation:

The hybrid advantage persists as the modeled sequence length increases from
five to ten components.

Do not claim that longer sequences inherently favor ML.

---

## B7. Direct Numerical Reference Stability

Primary table:

table_v33_direct_reference_stability.csv

Primary figure:

v33_direct_reference_stability.png

LOW:
0.315634

MEDIUM:
0.292256

HIGH:
0.291004

HIGH vs MEDIUM difference:
0.428%

Interpretation:

The low-budget numerical reference was not fully stabilized.

Medium and high budgets provide similar mean estimates.

The reference remains finite-budget and is not a proof of global optimality.

---

## B8. Residual ML Prediction Performance

Primary table:

table_v33_residual_ml_performance.csv

Key development results:

delta z:
R2 = 0.1520

delta theta:
R2 = 0.7821

delta locator:
R2 = 0.0509

Interpretation:

Residual learnability is strongly output-dependent.

The theta residual is substantially easier to predict.

Locator prediction remains weak but provides a small incremental development
benefit within the locked architecture.

Do not claim uniformly strong residual prediction.

---

# C. Discussion

## C1. Why Hybrid ML Is Stronger Than Pure ML Positioning

V3.2 showed that an analytical method can outperform ML when the simulator
matches its assumptions.

This is a strength of the thesis because the analytical baseline was not
weakened to make ML appear superior.

V3.3 therefore uses analytical LSQ as the base controller and ML only for the
residual information not adequately captured by the analytical decision.

---

## C2. Meaning of the 13.93% Improvement

The 13.93% result applies to:

- the synthetic V3.3 nonlinear assembly model,
- the tested deviation distributions,
- the defined correction capability,
- the composite prototype quality score,
- K = 10,
- the locked Random Forest hybrid architecture.

It is not a universal industrial improvement claim.

---

## C3. Robustness Interpretation

The locked model generalizes across the tested nonlinearity and measurement
noise regimes without retraining.

This demonstrates controlled robustness within the simulated model family.

It does not replace validation with measured industrial data.

---

## C4. Computational Interpretation

Deterministic LSQ is analytically efficient.

The hybrid adds one Random Forest residual prediction to the LSQ correction.

Direct nonlinear numerical optimization is more computationally expensive and
is therefore used primarily as a numerical quality reference.

---

# D. Limitations

Include explicitly:

- synthetic geometry
- one-dimensional representation
- phenomenological nonlinearity
- no validated contact/compliance/friction physics
- no clamping/welding/thermal effects
- prototype quality weights
- abstract correction parameters
- simulated measurement noise
- finite-budget numerical reference
- sequence scaling only to K = 10
- no industrial real-time deployment validation
- no measured industrial dataset

These should be presented as limitations, not hidden.

---

# E. Conclusion

Main technical conclusion:

The project demonstrates that part-specific adaptive assembly recommendation
can be formulated as a hybrid analytical/ML decision problem.

In the final V3.3 system, deterministic least-squares correction provides the
analytical base action and a residual Random Forest predicts individualized
correction adjustments using observable geometric and controller-known
features.

On 300 independent K = 10 assembly sequences, the locked hybrid improves mean
quality by 13.93% relative to deterministic LSQ and wins on 94.67% of
assemblies.

The advantage is statistically supported and remains present across all ten
assembly stages, nine tested nonlinear/noise regimes, and both K = 5 and
K = 10 sequence lengths.

---

# F. Future Work

Primary continuation:

- measured industrial part geometry
- high-fidelity contact/compliance simulation
- calibrated quality metrics
- physically mapped actuator/correction variables
- measurement-to-feature pipeline
- uncertainty-aware recommendations
- capability/risk thresholds
- closed-loop digital twin
- larger sequential production-line studies
- online adaptation using real process feedback

This connects naturally to a future master's-thesis direction.

---

# Source-of-Truth Rule

For final V3.3 claims use:

V3_3_FINAL_RESULT_MAP.md

and:

results/final_thesis_tables/

and:

figures/final_thesis_figures/

Do not replace final independent V3.3 numbers with pilot or development
numbers.

Development results may only be used to explain model training or architecture
selection.

V3.2 results remain valid as historical baseline evidence and should not be
silently rewritten as V3.3 results.