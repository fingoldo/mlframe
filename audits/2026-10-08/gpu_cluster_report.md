# GPU cluster report (2026-10-08)

All items done; verified green: test_shared_checks_wired (long_functions, fail_open), test_cuda_kernel_sources_use_explicit_64bit_types, test_no_import_cycles,
test_public_annotations, test_no_file_over_1k_loc, test_function_complexity, test_public_docstrings, test_pydoclint_baseline, test_code_audit_baseline
(last two after final edits: code audit, plus test_pairs_abs_corr_njit, test_binned_numeric_agg_resident, test_fast_host_searchsorted).

1. col0/col now `long long` in the analytic kernel (all later uses already 64-bit safe).
2. Cycle broken: `_DEGENERATE_REL_TOL` moved to leaf `_pairs_common`; `_pairs_abs_corr_gpu` imports it from there.
3. Concrete annotations on device_quantile, resident_analytic_gate, resident_observed_mi_and_bins (+ _fused helper), design_matvec, weighted_gram.
4. best-effort markers where the fallback is bit-identical or only routes between equivalent kernels (_extra_fe_families, _fe_cmi_redundancy_gate, _pairs_analytic_gpu x1 (the other handler already
   returned None), _shap_proxy_gpu_tuning x2, _device_quantile host fallbacks); the device-quantile self-check failure now logs WARNING with exception type.
5. `_win_corr` guard: a separate `_win_has_corr` flag replaces the dead `is None` test; `_safe_abs_corr` still returns 0.0 so failure keeps vetoing as before (behaviour unchanged).
6. ContentMemo: __getstate__/__setstate__ (drops lock and memo), DOC301 docstring merge, round-trip test added.
7. Carves (re-exported, pyflakes-clean freevar check): `_binned_agg_cheap_mi.py`, `_fe_auto_escalation_proposers.py`, `_usability_greedy_steps.py`. `_propose_poly` further split (<150).
8. Helpers: _tuned_noise_gate_backend, _extval_host_buffer, _config_corr (module level), _score_candidates_one_by_one, _discretise_survivor_block, _device_error_classes, _combo_chunk_cols.
   `_function_length_baseline.json` lowered / entries removed; `_fail_open_handlers_baseline.json` entries removed for moved/obsolete scopes (including the dead `_als_solve_gpu`).
   Side effect: moved handlers re-keyed, so they were marked best-effort or their old baseline entries removed (shrink only).

Changed paths (src/mlframe/feature_selection/): filters/{_binned_numeric_agg_fe,_device_quantile,_extra_fe_families,_fast_host_ops,_fe_auto_escalation,_fe_cmi_redundancy_gate,
_usability_aware_selection,_usability_pool_resident}.py, filters/_feature_engineering_pairs/{_pairs_abs_corr_gpu,_pairs_analytic_gpu,_pairs_common,_pairs_dispatch,_pairs_emit,_pairs_score_steps}.py,
filters/_mrmr_fe_step/{_step_score,_step_score_parts2}.py, filters/_orthogonal_univariate_fe/_orth_mi_backends.py, filters/hermite_fe/_als_kernels_gpu.py, shap_proxied_fs/_shap_proxy_gpu_tuning.py.
New: filters/{_binned_agg_cheap_mi,_fe_auto_escalation_proposers,_usability_greedy_steps}.py.
Tests/baselines: tests/feature_selection/fe/test_fast_host_searchsorted.py, tests/test_meta/{_function_length_baseline,_fail_open_handlers_baseline}.json.
