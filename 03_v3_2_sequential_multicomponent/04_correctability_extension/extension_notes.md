# Correctability Extension - Technical Record

## Purpose

This extension investigates whether the validated V3.2 sequential
assembly framework can provide information about correction difficulty
before the current assembly correction is applied.

The frozen V3.2 implementation was not modified.


## 1. Observed correctability analysis

Independent validation results from 300 assemblies and 1500 sequential
component decisions were analysed.

Observed component-level results:

- 94.60% of component decisions improved after Structured-20 correction.
- Mean improvement was approximately 64.54%.
- Median improvement was approximately 71.58%.
- 99.33% of finished assemblies improved relative to zero correction.

Exploratory outcome classes were created:

- HIGHLY_CORRECTABLE
- CORRECTABLE
- DIFFICULT
- NOT_IMPROVED

These classes are analytical descriptors of observed synthetic outcomes.
They are not industrial tolerance or pass/fail definitions.


## 2. Sequential difficulty behaviour

Correctability difficulty increased through the sequential assembly.

The proportion classified as difficult increased from approximately:

- 21.0% at component 1
- 24.3% at component 2
- 33.7% at component 3
- 45.3% at component 4
- 54.7% at component 5

This indicates that accumulated assembly state contributes to correction
difficulty.


## 3. Pre-decision correctability prediction

A binary pre-decision target was defined:

NORMAL:
- HIGHLY_CORRECTABLE
- CORRECTABLE

NEEDS_ATTENTION:
- DIFFICULT
- NOT_IMPROVED

Only information available before the current correction decision was used
as prediction input.

Four information levels were compared:

1. sequential stage only
2. incoming component/process information
3. accumulated assembly-state information
4. combined component/process and assembly-state information


## 4. Initial assembly-wise holdout result

The combined Random Forest predictor achieved approximately:

- balanced accuracy: 79.62%
- needs-attention recall: 73.15%
- normal specificity: 86.09%
- ROC AUC: 0.8877

The combined information improved balanced accuracy by approximately:

- +20.02 percentage points relative to stage only
- +7.64 percentage points relative to component/process only
- +14.39 percentage points relative to state only


## 5. Repeated holdout validation

Ten repeated assembly-wise holdout tests were performed.

Combined predictor:

- mean balanced accuracy: 81.56%
- standard deviation: 2.60 percentage points
- minimum balanced accuracy: 77.61%
- mean needs-attention recall: 77.94%
- minimum needs-attention recall: 71.09%
- mean ROC AUC: approximately 0.8909

Mean balanced accuracy by information level:

- stage only: 56.57%
- component/process: 70.20%
- state only: 65.34%
- combined: 81.56%

The combined pre-decision representation therefore provided substantial
information beyond sequential stage alone.


## 6. Out-of-fold risk analysis

Five-fold grouped cross-validation was used to generate one completely
out-of-fold difficulty probability for every component decision.

No model evaluated an assembly used during its own training.

Overall out-of-fold prediction performance:

- balanced accuracy: 80.78%
- needs-attention recall: 75.73%
- needs-attention precision: 78.92%
- normal specificity: 85.83%
- F1: 0.7729
- ROC AUC: 0.8881
- PR AUC: 0.8706


## 7. Risk/workload analysis

A 30% review workload was selected as the investigated advisory operating
point.

At this workload:

- needs-attention capture: 64.24%
- difficult-case capture: 65.74%
- not-improved capture: 54.32%
- high-regret capture: 44.00%
- high-residual capture: 84.00%
- attention precision: 88.22%
- enrichment relative to population prevalence: 2.14x

The workload/capture trade-off was considered useful but not sufficiently
strong for automatic routing.


## 8. Advisory decision-support result

The final interpretation is therefore advisory rather than autonomous.

Using the selected risk threshold of approximately 0.628:

ROUTINE:
pre-decision difficulty probability below the advisory threshold.

REVIEW_ADVISED:
pre-decision difficulty probability at or above the advisory threshold.

CAPABILITY_WARNING:
recommended correction utilization at or above 90% of available capability.

Observed distribution:

- ROUTINE: 70% of component decisions
- REVIEW_ADVISED: 30%
- CAPABILITY_WARNING: 0% in the evaluated independent population


## 9. Advisory enrichment

Among REVIEW_ADVISED decisions:

- needs-attention rate: 88.22%
- not-improved rate: 9.78%
- high-residual rate: 70.00%

Among ROUTINE decisions:

- needs-attention rate: 21.05%
- not-improved rate: 3.52%
- high-residual rate: 5.71%

The needs-attention prevalence was therefore approximately 4.19 times
higher in the review-advised population than in the routine population.


## 10. Final interpretation

The evidence supports a difficulty-aware advisory decision-support layer.

The supported architecture is:

incoming component geometry
+
current accumulated assembly state
+
batch/process information
        |
        v
pre-decision difficulty estimation
        |
        +------------------------+
        |                        |
     ROUTINE               REVIEW_ADVISED
        |                        |
        +------------+-----------+
                     |
                     v
             Structured-20
         correction recommendation
                     |
                     v
           correction capability
                  check

Structured-20 remains the primary correction recommendation method.

The risk estimator does not automatically trigger Bayesian Optimization.

Bayesian Optimization remains an investigated optional refinement rather
than a mandatory part of every assembly decision.

The advisory predictor must not be described as an industrially validated
failure predictor because all current validation remains synthetic.


## 11. Important limitations

The correctability labels are derived from outcomes of the synthetic V3.2
assembly model.

The advisory threshold is derived from the investigated synthetic
population.

The results do not establish an industrial pass/fail threshold.

Measured geometry and real process data are required before using the
advisory probability for production decisions.

The current result demonstrates technical feasibility and decision-support
potential only.