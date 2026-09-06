# Phase 6 — Terminology Check and Interpretation Update

## 1. Status

Phase 6 addresses two documentation questions arising after the post-freeze
method-validation experiments:

1. whether the existing repository improperly describes the direct numerical
   reference as an oracle, true optimum, or global optimum; and
2. whether the interpretation of the locked V3.3 Hybrid-versus-LSQ result
   requires refinement in light of the post-freeze Composite-Q and
   mean-residual experiments.

No controller, model, training data, validation result, or file inside the
locked V3.3 scientific package is modified in this phase.

---

## 2. Terminology audit

The repository was reviewed for the terms:

- oracle
- true optimum
- global optimum

The existing technical material already treats the direct numerical method
cautiously.

Examples include statements that:

- the method is deliberately not called oracle optimization;
- the numerical reference is not claimed to be a proven global optimum;
- the direct nonlinear reference is interpreted as a finite-budget numerical
  quality benchmark rather than proof of global optimality.

Consequently, no repository-wide terminology correction is required.

The preferred interpretation remains:

> finite-budget direct numerical benchmark

The terminology audit does not cover the final thesis Word/PDF document.
That document therefore requires a separate search for the same terms before
submission.

---

## 3. Original V3.3 interpretation

The locked V3.3 robustness analysis already observed that the Hybrid retained
a positive advantage over LSQ at zero nonlinearity.

The V3.3 result map noted that this was consistent with a difference between
the two objectives:

- deterministic LSQ minimizes a profile least-squares objective;
- the reported assembly performance is evaluated using a separate composite
  quality objective based on mean gap, maximum gap, parallelism error, and
  RMS deviation.

At the time of the V3.3 freeze, this objective-mismatch explanation was a
plausible interpretation but had not been isolated experimentally.

---

## 4. Post-freeze test of the objective-mismatch hypothesis

Phase 1 directly tested this interpretation using the Bounded Composite-Q
Deterministic Optimizer.

The optimizer targets the same composite quality score used for final
evaluation while respecting the same correction bounds and replaying the
same 300 independent K=10 validation assemblies.

The validated mean final qualities were:

- LSQ: 0.451750
- Locked Hybrid LSQ + RF: 0.388839
- Composite-Q: 0.337318
- Finite-budget direct numerical benchmark: 0.360594

Composite-Q improved on LSQ by 25.33%.

The locked Hybrid improved on LSQ by 13.93%.

Composite-Q also outperformed the locked Hybrid on all 300 independent
validation assemblies.

Formal paired analysis confirmed the Composite-Q-versus-Hybrid difference:

- mean Hybrid-minus-Composite-Q difference: +0.051522
- 95% paired bootstrap confidence interval: [0.047623, 0.055633]
- Holm-adjusted Wilcoxon p-value: 3.042e-50
- paired standardized effect size: 1.4511

The post-freeze evidence therefore demonstrates that optimization-objective
alignment is a major and, within the investigated controlled environment,
dominant factor in the performance difference between the original LSQ
baseline and the learned Hybrid controller.

The locked Hybrid result against LSQ remains valid, but that comparison alone
does not establish that machine learning is necessary for obtaining the best
assembly-quality result.

---

## 5. Interpretation of the residual-ML contribution

Phase 2 tested whether the Hybrid's advantage over LSQ could be reproduced by
simple non-personalized residual corrections.

The resulting mean final qualities were:

- LSQ: 0.451750
- Global mean residual: 0.451740
- Stage-specific mean residual: 0.452038
- Locked Hybrid: 0.388839

The global mean residual produced essentially no improvement over LSQ.

The stage-specific mean residual was also effectively equivalent to LSQ and
slightly worse in mean quality.

The Hybrid remained approximately 13.9% better than both simple residual
baselines, with a 94.33% assembly-level win rate against each.

Formal paired statistics confirmed these differences after Holm correction.

Therefore, the Hybrid's improvement over LSQ is not explained by a trivial
global correction bias or by a simple stage-dependent average correction.
The RF residual model captures state-dependent information that these
non-personalized baselines do not reproduce.

This does not imply that the learned Hybrid provides additional quality
improvement beyond objective-aligned Composite-Q optimization. In the present
controlled environment, Composite-Q is the stronger controller.

---

## 6. Updated scientific interpretation

The combined post-freeze evidence supports the following interpretation.

The locked V3.3 Hybrid provides a substantial and statistically robust
improvement relative to the profile-error LSQ baseline. Simple global and
stage-conditioned residual corrections fail to reproduce this improvement,
which supports the presence of meaningful state-dependent residual structure.

However, the stronger Composite-Q experiment demonstrates that direct
deterministic optimization of the actual evaluation objective produces a
larger improvement and outperforms the Hybrid consistently on the same
independent validation population.

The Hybrid-versus-LSQ advantage must therefore not be interpreted primarily
as evidence that machine learning is required because of nonlinear assembly
behaviour.

Instead, two distinct effects are present:

1. objective alignment provides the largest demonstrated improvement over
   LSQ; and
2. residual machine learning captures individualized state-dependent
   correction structure relative to LSQ that cannot be reproduced by simple
   population-level or stage-level mean residual corrections.

Within the present low-dimensional controlled synthetic environment,
objective-aligned deterministic optimization is the preferred method when the
composite quality function can be evaluated directly at acceptable
computational cost.

Machine learning remains relevant as a supporting method for conditions in
which direct objective evaluation becomes expensive, incomplete, uncertain,
high-dimensional, or unavailable during online decision making.

---

## 7. Preservation of the locked V3.3 record

The files under:

06_v3_3_nonlinear_hybrid_upgrade/

remain unchanged.

The original V3.3 interpretation is retained as the historical interpretation
available at the time the architecture and validation package were frozen.

The post-freeze conclusions are documented separately under:

07_postfreeze_method_validation/

This preserves chronological traceability between:

- the locked V3.3 evidence;
- the later challenge experiments; and
- the resulting refinement of the scientific interpretation.

---

## 8. Remaining documentation actions

The final thesis Word/PDF document should be searched before submission for:

- oracle
- true optimum
- global optimum

Any wording that implies mathematical global optimality for the direct
numerical reference should be replaced by the experimentally supported term:

> finite-budget direct numerical benchmark

A separate final repository documentation pass remains necessary to ensure
that the root README presents V3.3 as the locked validated technical core and
the post-freeze method-validation work as the subsequent scientific challenge
and interpretation layer.