# new_gates_ast checkpoint (paused by coordinator, nothing committed)

Files changed (src, all under src/mlframe): calibration/_independence_check.py, feature_engineering/hurst.py, feature_selection/ace.py,
feature_selection/filters/_hinge_detect_gpu_resident_batch.py, _orthogonal_univariate_fe/_fourier_detect_gpu_resident.py, _orth_dedup.py,
training/composite/ensemble/_calibration.py (Platt Newton centred), training/composite/transforms/linear.py (centred OLS closed + batched),
training/composite/cache.py (_unfingerprintable_key fallback).
New meta tests: tests/test_meta/test_no_{cancellation_prone_moments,unread_function_params,hardcoded_seed_in_library,constant_fallback_cache_key}.py

Done: cancellation_prone_moments 9 hits: 2 real fixed (linear.py, _calibration.py), 7 suppressed with moment-ok reasons.
hardcoded_seed 1 real hit (ace.py:137) fixed via split_seed threaded from ace_select random_state (default identical); 15 benchmark hits excluded by _benchmarks/.
constant_fallback_cache_key 1 hit (cache.py) fixed.

Remaining:
- tests/feature_selection/cv_policy/test_cv_policy_selector_wiring.py:205 lambda must accept split_seed (lambda n, y, rng, cv_policy=None, split_seed=0).
- unread_function_params 33 hits, planned dispositions: remove dead param (no caller passes): _linear_core seed (update 3 callers in _observation.py), error_bias_per_feature seed,
  segments_bar seed, _pred_sample_trace_panel seed (update regression.py:840 call), partition_folds random_state, mixup_batch sample_weight.
  unused-ok with reason: tmc_shapley/data_banzhaf/shapley_model_values n_jobs (documented no-op), calculate_relevance_table n_jobs/random_state, _kendall_p random_state,
  hybrid_mahalanobis/hybrid_sir random_state (deterministic), combine_probs sample_weight (per-row weights cannot change a within-row blend; pinned by test_x_ml_correctness_meta_fixes),
  4 composite transform *_fit sample_weight (registry signature, documented), shap_worst_errors_explanation/render_class_structure_diagnostic seed (uniform dispatch signature),
  13 verbose params (decide per function: stage signature, logging via logger levels).
- Regression tests: large-offset OLS (x=1.7e9+N(0,1)) closed vs batched; Platt translation equivariance; ace split_seed reaches holdout_indices; cache two different failing frames get different keys.
- Run new meta tests, edited modules' tests, the -k meta selection, mypy and black_filtered_apply on edited files; write audits/2026-10-04/new_gates_ast.md.

Next step: edit the cv_policy test lambda, then apply the unread-param dispositions above.
Command pending: python -m pytest tests/test_meta/test_no_unread_function_params.py tests/test_meta/test_no_cancellation_prone_moments.py -n 1 -s --no-cov
## Resumed (session 2)
- NOTE: another session transiently reverted the working tree mid-run; re-verify edits with git diff before committing.
- Done: cv_policy test lambda; unread_function_params now 0 findings (removed dead params: _linear_core seed, partition_folds random_state, mixup_batch sample_weight,
  error_bias_per_feature/segments_bar/_pred_sample_trace_panel seed; the rest suppressed with `# unused-ok:` reasons).
- Next: regression tests, run 4 meta tests + touched modules' tests, mypy, black_filtered_apply, commit with explicit paths, write new_gates_ast_report.md.
- Session 2 done: all gates green, report written; committing owned paths.
