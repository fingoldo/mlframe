# New py-ci-shared AST gates wired into mlframe tests/test_meta

Gates: cancellation_prone_moments, unread_function_params, hardcoded_seed_in_library, constant_fallback_cache_key.
Wiring: tests/test_meta/test_no_{cancellation_prone_moments,unread_function_params,hardcoded_seed_in_library,constant_fallback_cache_key}.py (4 passed).

## cancellation_prone_moments (9 hits)
| Site | Disposition |
|---|---|
| training/composite/transforms/linear.py, closed OLS | FIXED: mean-centred reductions (large-offset base no longer cancels) |
| training/composite/transforms/linear.py, batched OLS | FIXED: same centred form, bit-agreeing with the scalar path |
| training/composite/ensemble/_calibration.py, Platt Newton | FIXED: Hessian solved in the v-weighted-mean-centred basis; start point is now the slope-1 map centred on the weighted mean logit, which made the fit exactly translation-equivariant (the old start diverged for a logit shifted by 3 or more) |
| 7 other sites (calibration/_independence_check.py, feature_engineering/hurst.py, feature_selection/filters hinge/fourier/orth_dedup, ...) | SUPPRESSED with a moment-ok reason: the quantities are centred or bounded by construction |

## hardcoded_seed_in_library (1 real hit, 15 benchmark hits)
| Site | Disposition |
|---|---|
| feature_selection/ace.py `_pfi_split` random_state=0 | FIXED: `split_seed` threaded from `ace_select(random_state)`; default value unchanged, so default behaviour is identical |
| 15 hits under `_benchmarks/` | EXCLUDED: benchmark scripts are not library behaviour |

## constant_fallback_cache_key (1 hit)
| Site | Disposition |
|---|---|
| training/composite/cache.py `_row_order_fingerprint` returned "" on failure | FIXED: `_unfingerprintable_key` digests type, shape and columns, so differently-structured frames never share a key |

## unread_function_params (33 hits)
Removed as dead (no caller passed them): `_linear_core` seed (3 callers updated), `partition_folds` random_state, `mixup_batch` sample_weight,
`error_bias_per_feature` seed, `segments_bar` seed, `_pred_sample_trace_panel` seed (regression.py caller updated).

Kept with `# unused-ok:` and a reason, because callers or tests pin the signature:
- n_jobs documented no-ops: tmc_shapley, data_banzhaf, shapley_model_values, calculate_relevance_table.
- random_state with no effect: `_kendall_p_numeric_continuous` (a test pins that the seed has no effect), hybrid_mahalanobis_density_fe, hybrid_sir_direction_fe.
- combine_probs sample_weight: per-row weights cannot change a within-row blend.
- Registry fit signatures: _causal_anchor_residual_fit, _second_diff_fit, _target_encoding_residual_fit, _theilsen_residual_fit (sample_weight, unweighted by design).
- Uniform dispatch signatures: render_class_structure_diagnostic and shap_worst_errors_explanation (seed).
- 13 verbose params (stage signatures, tests pass verbose= explicitly, output goes through logger levels): retain_usable_pure_forms, retain_usable_raw_columns,
  decide_exhaustive_sweep, cardinality_prescreen, evaluate_candidate, postprocess_candidates, _persist_fitted_estimators, _maybe_apply_posthoc_calibration,
  unigram_rescues_text_features, run_composite_target_discovery, run_composite_post_processing, run_temporal_audit_batch, _apply_row_wise_extensions.
  FUTURE: gating the info-level lines on `verbose` would be a behaviour change across all callers and was not done here.

## Regression tests added
- tests/training/composite/test_centred_moments_large_offset.py: closed and batched OLS at x = 1.7e9 + N(0,1); Platt translation equivariance (shift 40); two failing frames get different cache keys.
- tests/feature_selection/cv_policy/test_ace_split_seed.py: split_seed reaches holdout_indices.
- tests/feature_selection/cv_policy/test_cv_policy_selector_wiring.py: spy lambda accepts split_seed.

## Verification
4 new meta tests pass; 70 calibration/cache tests, 96 mixup/error-analysis/cv_policy/kendall tests, 51 observation-layer/regression-chart tests, 6 final-fix tests pass; mypy clean on 56 edited files;
black_filtered_apply clean on all 45 owned files. The Platt regression test failed first (Newton diverged on a shifted logit), which led to the centred start point.
