# V3.2 Model Notes

## 1. Purpose of Version 3.2

Version 3.2 extends the deviation-aware adaptive assembly framework to a
sequential multi-component assembly problem.

The objective is to investigate whether part-specific adaptive correction
can reduce accumulated geometric quality deviations when several varying
components are assembled sequentially.

The framework is intended as a controlled synthetic feasibility study and
decision-support prototype. It is not an industrially validated production
model.


## 2. Sequential assembly representation

The partially assembled product is represented by a one-dimensional
geometric deviation profile along a normalized assembly interface.

For component k, the state transition is represented as:

S_k(s) = S_(k-1)(s) + D_k(s) + F_(b,k)(s) + C_k(s)

where:

- S_(k-1) is the accumulated assembly state before component k.
- D_k is the individual geometric deviation of the incoming component.
- F_(b,k) is the batch-dependent process disturbance acting during the
  current assembly operation.
- C_k is the applied correction profile.

The model is additive and geometric. It is not a structural mechanics or
contact simulation.


## 3. Interpretation of batch/process disturbance

The batch disturbance is interpreted as a repeated systematic
assembly-process contribution.

The same batch-dependent disturbance realization can therefore act during
each sequential component installation.

This means that systematic process bias can accumulate over the assembly
sequence.

The batch disturbance must not be interpreted as a single static fixture
geometry error applied only once to the complete product.

The implementation contains three geometric process contributions:

- batch offset bias,
- batch angular bias,
- localized fixture-related process contribution.

These are synthetic process mechanisms used to test robustness of the
adaptive recommendation framework.


## 4. Component-deviation mechanisms

Incoming components may contain:

- offset,
- tilt,
- bend,
- waviness,
- twist-like deviation,
- localized bump,
- combinations of multiple modes.

The twist representation is a simplified one-dimensional twist-like
profile and is not a full three-dimensional torsional deformation model.

The localized bump is represented analytically with a Gaussian profile.


## 5. Synthetic severity levels

The component severity scales are:

- Low: 0.60
- Medium: 1.00
- High: 1.50
- Extreme: 2.00

These values are controlled prototype assumptions.

They are not presented as universal industrial tolerance distributions.

The final robustness analysis additionally tested deviation magnitudes at:

- 0.8 times nominal,
- 1.0 times nominal,
- 1.2 times nominal.


## 6. High-variation batch condition

The high-variation condition uses a variation multiplier of 1.50.

In the current dataset generator, this value does not directly multiply
every geometric amplitude by 1.50.

Instead, it shifts the controlled severity distribution toward more severe
component classes.

Therefore the high-variation condition should be described as a
severity-distribution shift rather than a direct geometric amplitude
multiplier.


## 7. Correction parameters

Three effective assembly adjustment parameters are used:

1. z_adj
   Global vertical correction.

2. theta_adj
   Global angular correction.

3. locator_offset
   Localized correction around the centre of the profile.

The correction profile is:

C(s) =
z_adj
+ tan(theta_adj) * L * (s - 0.5)
+ locator_offset * exp(-(s - 0.5)^2 / (2 sigma^2))

The nominal correction bounds are:

- z_adj: +/- 2.5 mm
- theta_adj: +/- 1.2 degrees
- locator_offset: +/- 1.0 mm

These variables represent effective assembly-adjustment capabilities.

They should not be interpreted as guaranteed physical actuators available
in every real assembly process.


## 8. Correction capability

Correction capability is represented by normalized maximum utilization:

U = max(
    abs(z_adj) / z_limit,
    abs(theta_adj) / theta_limit,
    abs(locator_offset) / locator_limit
)

Capability is treated separately from geometric quality.

The final independent validation showed that the Structured-20 controller
used low correction capability on average.

Sensitivity testing additionally reduced available correction capability
to 80% and 60% of the nominal limits.


## 9. Quality metrics

The geometric quality metrics are:

- mean absolute gap,
- maximum absolute gap,
- end-to-end parallelism error,
- RMS geometric deviation.

The base quality score is:

Q =
0.30 * mean_gap
+ 0.30 * max_gap
+ 0.30 * parallelism_error
+ 0.10 * RMS

The weights are prototype design assumptions and are not industrial
standards.

Sensitivity testing was therefore performed with alternative weighting
schemes.

The controller remained strongly superior to zero correction across all
tested weighting definitions.


## 10. ML surrogate model

A profile-aware Random Forest surrogate is used to predict assembly quality
for candidate corrections.

The model contains scalar state descriptors together with sampled values of
the accumulated geometric state profile.

Random Forest is used as a quality predictor and candidate-ranking
surrogate.

It is not interpreted as a failure-diagnosis model.


## 11. Structured-20 controller

The operational V3.2 controller evaluates twenty physically structured
candidate corrections using the Random Forest surrogate.

The candidate set combines:

- zero correction,
- informed correction,
- local perturbations around the informed solution,
- moderate random candidates,
- broader correction-space candidates.

The RF ranks the candidates and the best predicted candidate is applied.

This structured search became the final operational recommendation method
because it showed strong independent generalization and substantially lower
computational cost than repeated Bayesian Optimization.


## 12. Bayesian Optimization

Bayesian Optimization was extensively investigated.

The original unconstrained search showed weak closed-loop behaviour despite
strong offline surrogate metrics.

Subsequent diagnostics identified search-space mismatch and closed-loop
distribution shift as important causes.

A structured warm-start Bayesian Optimization strategy restored strong
performance.

However, independent validation showed only a negligible average
system-level improvement over Structured-20.

Therefore Bayesian Optimization is retained as an optional refinement for
ambiguous decisions rather than an always-on operational optimizer.


## 13. Selective Bayesian Optimization

A pre-decision ambiguity trigger was calibrated before final independent
validation.

The frozen margin threshold was approximately:

0.0280037392

On the independent validation population, Bayesian Optimization was
triggered for approximately 26.13% of component decisions.

Structured-20 and Selective BO produced essentially equivalent final system
quality.

The final interpretation is therefore:

Structured-20 RF:
primary operational recommendation controller.

Selective Bayesian Optimization:
optional refinement for ambiguous candidate-ranking situations.


## 14. Independent validation

The final independent validation population contained:

- 300 finished assemblies,
- 30 independent batch realizations,
- 6 balanced batch/process conditions,
- 5 sequential components per assembly,
- 1500 sequential component decisions.

The validation population used a new random seed and a separate assembly-ID
range.

No RF retraining, controller tuning, trigger recalibration, trust-region
tuning, correction-bound tuning or penalty tuning was performed using the
independent validation population.


## 15. Main independent validation result

Mean final quality:

- Zero correction: approximately 0.9505
- Structured-20: approximately 0.2386
- Selective BO: approximately 0.2377

Structured-20 mean improvement relative to zero correction:

approximately 74.90%

Structured-20 assembly-level win rate relative to zero correction:

approximately 99.33%

Structured candidate ranking:

- exact-best candidate rate: approximately 86.27%
- top-3 candidate rate: approximately 99.47%

Mean Structured-20 correction capability utilization:

approximately 0.115

Decisions at or above 90% capability:

0%


## 16. Robustness and sensitivity

Three additional independent robustness seeds were evaluated.

The minimum assembly-level win rate across those seeds was approximately:

98.67%

Deviation-magnitude sensitivity:

0.8x, 1.0x and 1.2x nominal deviation magnitudes were evaluated.

The minimum win rate remained approximately:

99.11%

Correction-capability sensitivity:

At only 60% of the nominal correction limits:

- win rate remained approximately 98.67%
- mean improvement remained approximately 69.55%

Quality-weight sensitivity:

Alternative quality definitions were evaluated.

The minimum seed-level win rate across tested weighting schemes remained:

98.00%


## 17. Main modelling limitations

The current V3.2 framework does not model:

- full three-dimensional geometry,
- contact mechanics,
- elastic deformation,
- plastic deformation,
- fastening or joining-force effects,
- friction,
- component compliance,
- fixture compliance,
- detailed tolerance-stack mechanics,
- welding distortion,
- thermal distortion,
- measured CMM or 3D scan data.

The framework uses a simplified one-dimensional additive geometric
representation.

Consequently, the results demonstrate feasibility of the adaptive
decision-support architecture under controlled synthetic conditions.

They do not constitute industrial validation.


## 18. Scientific interpretation

The strongest supported conclusion from V3.2 is not that the method
guarantees industrial assembly quality.

The supported conclusion is:

A deviation-aware, profile-informed and capability-aware adaptive
recommendation framework can substantially reduce accumulated geometric
quality deviation in a controlled sequential multi-component assembly
model.

The results further show that:

- adaptation is strongly beneficial relative to zero correction,
- ML candidate ranking can generalize to independent synthetic populations,
- structured physically informed search can outperform uninformed search,
- correction capability can be incorporated explicitly,
- performance remains robust under tested synthetic assumptions,
- Bayesian Optimization does not necessarily need to be executed for every
  assembly decision.


## 19. Future extension

The next research stage should replace or complement the analytical
synthetic geometry with:

- measured component geometry,
- CMM data,
- 3D scanning,
- higher-fidelity geometric simulation,
- FEM or digital-twin information.

Further extensions can include:

- uncertainty-aware recommendations,
- explicit correctability classification,
- adaptive confidence estimation,
- multi-component 3D geometry,
- richer correction degrees of freedom,
- closed-loop post-assembly measurement feedback,
- online model updating,
- digital-twin-based assembly decision support.