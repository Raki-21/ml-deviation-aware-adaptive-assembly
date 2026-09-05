### Recommendation-status and feasibility classification
- Created recommendation_status.py.
- Created run_07_recommendation_status_analysis.py.
- Added industrial recommendation statuses: PASS_RECOMMENDED, MANUAL_REVIEW, REWORK_REQUIRED, and OUT_OF_CORRECTION_RANGE.
- Generated recommendation_status_cases.csv.
- Generated recommendation_status_summary.csv.
- Generated recommendation_status_by_scenario.csv.
- Main result: individual adaptive setting classified 774 of 800 cases as PASS_RECOMMENDED.
- Main result: individual adaptive setting achieved 96.75% direct release/pass recommendation.
- Main result: only 26 of 800 individual adaptive cases were classified as REWORK_REQUIRED.
- Main result: no individual adaptive cases were classified as OUT_OF_CORRECTION_RANGE.
- Interpretation: global-best and batch-adaptive strategies are limited for heterogeneous part deviations, while individual part-specific recommendation provides the strongest quality and feasibility improvement.

### Advanced deviation/circumstance extension
- Added two controlled advanced scenarios: twist_dominant and fixture_drift.
- Kept correction parameters limited to z_adj, theta_adj, and locator_offset to avoid fake physical complexity.
- Regenerated batch_aware_dataset.csv.
- Dataset increased from 6400 rows x 20 columns to 8800 rows x 22 columns.
- Unique physical part cases increased from 800 to 1100.
- Re-ran adaptation-level evaluation after adding the new scenarios.
- Re-ran recommendation-status analysis after adding the new scenarios.
- Main result after extension: individual adaptive setting achieved 95.45% pass/release rate.
- Main result after extension: individual adaptive setting reduced mean quality score from 1.9185 to 0.6557.
- Main result after extension: individual adaptive setting achieved 65.82% improvement versus nominal.
- Main result after extension: individual adaptive setting classified 1050 of 1100 cases as PASS_RECOMMENDED.
- Main result after extension: only 50 of 1100 individual adaptive cases were classified as REWORK_REQUIRED.
- Interpretation: even after adding harder twist and fixture-drift scenarios, individual part-specific adaptation remained strongly superior to nominal, global-best, and batch-adaptive strategies.

### Failure analysis
- Created failure_analysis.py.
- Created run_08_failure_analysis.py.
- Generated failure_analysis_by_strategy.csv.
- Generated failure_analysis_by_scenario.csv.
- Generated failure_analysis_by_metric.csv.
- Generated individual_adaptive_failed_cases.csv.
- Main result: individual adaptive setting achieved 1050 PASS_RECOMMENDED cases out of 1100.
- Main result: the remaining 50 individual-adaptive cases were classified as REWORK_REQUIRED.
- Main result: no individual-adaptive cases were classified as OUT_OF_CORRECTION_RANGE.
- Failure analysis showed that the remaining individual-adaptive failures were caused by parallelism_error, not mean_gap or max_gap.
- Interpretation: the current three-parameter correction model strongly reduces gap-related errors, while the remaining difficult cases indicate parallelism-related correction limits.

### Optimization efficiency comparison
- Checked the previously generated optimization-efficiency result files using run_09_check_optimization_efficiency_results.py.
- Confirmed that optimization_efficiency_summary.csv, optimization_efficiency_checkpoints.csv, and optimization_efficiency_curve.csv exist and are readable.
- Compared Bayesian Optimization against Random Search.
- Main result: Bayesian Optimization achieved a final mean best quality score of approximately 1.2779.
- Main result: Random Search achieved a final mean best quality score of approximately 1.5560.
- Main result: Bayesian Optimization improved final mean best quality by approximately 17.87% compared with Random Search.
- Checkpoint analysis showed that after the initial exploration phase, Bayesian Optimization consistently achieved lower mean best quality scores than Random Search.
- Interpretation: Bayesian Optimization is justified because it guides the search toward better assembly parameter settings more efficiently than uninformed random sampling.
- Note: the optimization-efficiency curve file contains incomplete scenario metadata in some rows, but the numerical comparison between Bayesian Optimization and Random Search is usable for the thesis. Scenario-level metadata cleanup can be handled later if needed.

### ML model screening
- Created model_screening.py.
- Created run_10_model_screening.py.
- Compared Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, and Support Vector Regression.
- Used repeated train/test splits to check model stability.
- Evaluation metrics: MAE, RMSE, R², training time, and prediction time.
- Main result: Random Forest achieved the best overall performance.
- Random Forest mean MAE: approximately 0.2977.
- Random Forest mean RMSE: approximately 0.4140.
- Random Forest mean R²: approximately 0.9881.
- Gradient Boosting was second-best with mean RMSE approximately 0.5428 and mean R² approximately 0.9796.
- SVR also performed well but required significantly longer training time.
- Linear Regression and Ridge Regression performed poorly with mean R² approximately 0.032, confirming that the quality prediction problem is nonlinear.
- Interpretation: Random Forest is justified as the primary surrogate model because it provides the best balance of prediction accuracy, robustness, computational practicality, and interpretability.

### Version 3 result plots
- Created v3_result_plots.py.
- Created run_11_generate_result_plots.py.
- Generated report-ready plots in results/plots/v3.
- Generated adaptation_strategy_quality_comparison.png.
- Generated adaptation_strategy_pass_rate_comparison.png.
- Generated recommendation_status_distribution.png.
- Generated failure_metric_comparison.png.
- Generated optimization_efficiency_curve.png.
- Generated model_screening_rmse_comparison.png.
- Generated model_screening_r2_comparison.png.
- Interpretation: the plots provide visual evidence for the main Version 3 findings, including the superiority of individual adaptive recommendation, the value of feasibility classification, the remaining parallelism-related failure mode, the efficiency benefit of Bayesian Optimization, and the selection of Random Forest as the final surrogate model.