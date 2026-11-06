# Wave 3 tracker (2026-11-05)

Two read-only Opus 5.5 (low effort) audits closing the coverage gaps of wave 2: GPU kernel parity (`kernel_parity_rest.md`) and estimator contracts (`api_contract_rest.md`). Each finding was checked against HEAD before a fix.

## Kernel parity (PAR3)

| ID | Sev | Disposition | Commit / note |
|---|---|---|---|
| PAR3-1 | Medium | RESOLVED | `92066453b` float32 default keeps the double kernel for offset-heavy or non-finite inputs; test `test_usability_corr_float32_guard.py` |
| PAR3-2 | Low | RESOLVED | `fff343b13` one dtype for the |corr| table on every backend; test `test_usability_corr_table_dtype_is_backend_independent.py` |
| PAR3-3 | Low | DOC | CPU reassoc/contract fastmath vs GPU without it; matters only for columns at the near-constant cutoff |
| PAR3-4 | Info | REJECTED | RMSE and AUC GPU vs CPU agree (NaN, float64 accumulation, tie ranks) |

Coverage: the audit read 2 of about 60 GPU-kernel files. Not yet read: the MI and CMI kernels, `_gpu_metrics.py` beyond the two above, DTW, the transformer kernels, the honest-null and resident-select kernels. The list is in `kernel_parity_rest.md`. Next round should cover them.

## Estimator contracts (API3)

| ID | Sev | Disposition | Commit / note |
|---|---|---|---|
| API3-1 | Medium | RESOLVED | `251dc4a4c` `ShapProxiedFS` keeps None through get_params and clone; the per-host default resolves at fit |
| API3-2..11, API3-13 | Medium | RESOLVED | `3b1bfe6de` predict/transform before fit raise NotFittedError; test `tests/test_meta/test_estimators_refuse_predict_before_fit.py` |
| API3-12 | Low | DOC | `_Float32CastTransformer`, `_InfToNaNTransformer` accept *args/**kwargs and transform unfitted |
| API3-14 | Low | RESOLVED | `aafac049e` `EarlyStoppingWrapper` with `base_model=None` raises ValueError naming the parameter |
| API3-15 | High | RESOLVED | `73adbdbbd` composite multi-output predict refuses a different input width |
| API3-5 | Medium | RESOLVED | `a403b4b1a` Tabular1DCNN regressor and classifier record the width and refuse a different one at predict |
| API3-7, API3-8 | Low | RESOLVED | `a403b4b1a` `TrunkResidualMLPRegressor`, `MultiTaskAuxiliaryLossRegressor` record the width and refuse a different one at predict |

## Known failures in the shap-proxied test directory (run on 2026-11-06, 749 passed, 6 failed)

| Test | Status |
|---|---|
| `test_mrmr_then_shap_proxied_fs_reuses_artifacts_e2e`: prefilter does not switch to `precomputed_su` | Pre-existing: fails the same way on a clean worktree at HEAD (`8e4aea9c7`). Not caused by API3-1. Open. |
| `test_column_major_speedup_vs_row_major_reference`: ratio 1.23x vs 1.70x | Timing assertion, labelled load-sensitive in the test. Re-run on a quiet machine. |
| `test_biz_value_revalidation_cap_faster_recovery_preserved`: 1.05x vs 1.30x | Timing assertion. Re-run on a quiet machine. |
| `test_treeshap_gpu_matches_numba[True]`, `[False]`: max rel. diff 1.1e-4 | GPU numerical tolerance. Not checked against HEAD yet. Open. |
| `test_shap_proxied_fs_default_cap_uses_module_default` | Caused by API3-1: the test pinned the old contract. Reframed to the effective-width helper. RESOLVED in `251dc4a4c`. |
