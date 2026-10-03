# CI environment failures (agent_ci_env)

Items 1a-1d were already fixed at HEAD by 127157a28; verified locally: staleness + time_budget_ensemble + scm_beds harness = 18 passed.

1. macOS hybrid tier1 / scm_beds: tests/feature_selection/_fs_hybrid_cell_skip.py turns a cell whose error is a pytest Skipped into a skip. Staleness
   test: os.sep replaced by a backslash check. time_budget_ensemble: virtual clock, deterministic.
2. deep-nightly:
   - gil_release threading: skip when fewer than 4 usable cores (flag test still pins nogil). File: tests/feature_selection/info_theory/test_gil_release_threading_speedup.py.
   - tier1 TestGateStaysAffordable: wall-clock replaced by deterministic n_model_fits bound (measured 8 per arm). File: test_fs_hybrid_tier1.py.
   - shap_proxied 900s timeouts: cProfile of one ShapProxiedFS.fit (n=2000, p=500) = 570s on a saturated box, 246 xgboost fits, 86% in
     xgboost update (C++), 8% one-time numba compile; CI is a 2-vCPU runner with 2 xdist workers, a watchdog dump showed the first of 8 fits still
     running at 600s. Work is real, not waste. Timeouts now perf_time_budget(900) (4x under xdist, repo convention) in 12 files under
     tests/feature_selection/shap_proxied/. Not re-run end to end (too long on this box); effect unverified until CI.
   - faith_interaction XOR: additive recovers both operands on seed 1 (measured), 0 on seeds 0 and 2, so the premise is platform/seed dependent;
     test skips with that reason when unmet instead of failing. Faith recovered 2/2 on seed 0.
   - boruta heavy-tail: 2 seeds meant unanimity; measured noise admitted per seed 0..7 = 3,2,2,2,1,1,2,0; now 5 seeds, majority 3.
3. target_dist_overlay: multilabel y arrives as an object array of per-row label vectors (fuzz 3way, shard 16); np.asarray(float64) raised.
   Fix: _pool_row_vectors in reporting/charts/_error_analysis_shared.py applied at target_dist_overlay entry and in _as_float_1d. Same root cause
   broke training/_model_cache_fingerprint._array_digest (process_model RAISED ValueError); object arrays now hash by per-element repr.
   Tests added: overlay multilabel (tests/reporting/test_charts_error_analysis.py, 40 passed), fingerprint digest (tests/training/test_model_cache_training_fingerprint.py,
   not run to completion after the machine slowdown; the overlay one and its sibling passed).
