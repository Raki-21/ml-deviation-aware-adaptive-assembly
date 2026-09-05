# Version 3 Upgrade Plan

This folder is the working copy for all future upgrades.

Version 2 has been locked as the working baseline.

## Version 3 Goal

Upgrade the current single-case adaptive assembly prototype into a more general deviation-aware adaptive assembly recommendation framework.

## Completed Version 3 Work

### Batch-aware dataset generation
- Created batch-aware dataset generator.
- Added scenario types: offset_dominant, tilt_dominant, bend_dominant, waviness_dominant, mixed, disturbed_batch.
- Added severity levels: low, medium, high.
- Added batch_id, part_id, scenario_type, severity_level, and disturbance_flag.
- Generated batch_aware_dataset.csv with 6400 rows and 20 columns.
- Generated batch_aware_dataset_summary.csv.

### Phase 4: Capability-aware assembly decision layer
- Created capability_map.py.
- Created run_12_capability_map_analysis.py.
- Created run_13_plot_capability_map.py.
- Generated correction_capability_map.csv.
- Generated correction_capability_summary.csv.
- Generated correction_capability_map.png.
- Added scenario-level correction capability analysis for the individual adaptive recommendation strategy.
- Main result: fixture_drift was the most difficult scenario, with approximately 90.00% pass recommendation rate and 10.00% rework-required share.
- Offset_dominant was the most correctable scenario, with approximately 98.67% pass recommendation rate.
- The dominant remaining limitation across difficult cases was parallelism_error.
- Interpretation: the framework now identifies not only the best adaptive assembly parameters, but also the correction-capability boundary of the current assembly system.
- Product relevance: this supports a future adaptive assembly decision-support system that can recommend parameters, classify feasibility, detect correction limits, and suggest future process-capability improvements.

### Phase 4 plot extension: correction difficulty ranking
- Created run_14_plot_correction_difficulty_ranking.py.
- Generated correction_difficulty_ranking.png.
- The plot ranks deviation scenarios by remaining difficult-case percentage after individual adaptive recommendation.
- Main finding: fixture_drift is the most difficult scenario with approximately 10.00% difficult cases.
- Offset_dominant is the easiest scenario with approximately 1.33% difficult cases.
- Interpretation: the framework can identify not only the best parameter recommendation but also scenario-wise correction difficulty and capability limits.
- Product relevance: this supports the concept of a capability-aware adaptive assembly decision-support system for industrial use.

### Phase 4: Capability-aware assembly decision layer completed
- Created capability_map.py.
- Created run_12_capability_map_analysis.py.
- Created run_13_plot_capability_map.py.
- Created run_14_plot_correction_difficulty_ranking.py.
- Generated correction_capability_map.csv.
- Generated correction_capability_summary.csv.
- Generated correction_capability_map.png.
- Generated correction_difficulty_ranking.png.
- The capability analysis evaluates scenario-wise correction difficulty after individual adaptive recommendation.
- Main finding: fixture_drift is the most difficult scenario, with approximately 90.00% pass recommendation rate and 10.00% difficult/rework-required cases.
- Offset_dominant is the most correctable scenario, with approximately 98.67% pass recommendation rate and 1.33% difficult cases.
- The dominant remaining limitation across difficult cases is parallelism_error.
- Interpretation: the framework now identifies not only recommended parameters but also the correction-capability boundary of the current assembly system.
- Product relevance: this supports the long-term concept of a capability-aware adaptive assembly decision-support system for industrial use.