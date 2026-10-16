# Open items, 2026-10-16

Found while taking the four MRMR pillars (F2 across distributions, pure noise, xor3 among noise, conjunction with proxies) to default settings.

| # | Item | State |
|---|---|---|
| 1 | Conjunction pillar at default MRMR (50000 rows, 253 columns, ~3800 candidates): honest-null cut search was 78% of the fit (1358 of 1751 s); GPU uploads failed on `cudaErrorAlreadyMapped`; host memory exhausted by float64/int64 copies | cut search presorted (8-12x per call), compact null layouts, pageable uploads, chunked external validation, streamed modular FE. Re-run of the whole fit with the RSS sampler (`conj_rss.py`) still pending |
| 2 | `test_fe_float32_replay_parity.py::test_f32_nameset_matches_f64[orth_univariate]`: float32 fit loses `min(neg(x1),sqrt(x4))` that the float64 fit selects | fails identically on `543c3a02f` (not a regression of this work); root cause open |
| 3 | `operand_uplift_null_bar` (`_pairs_operand_floor.py`) has no caller: the pair joint-prevalence gate runs at the default uplift bar 1.0 instead of the data-driven null bar | open; wiring it in changes FE selection, needs a before/after on the pair-FE biz-value tests |
| 4 | Modules over the 1000-line budget on master: `_pairs_score_steps` 1229, `_gpu_resident_materialise` 1048, `_fit_impl_core` 1020, `batch_mi_noise_gate_gpu` 1077, `_phase_helpers_fit_pipeline` 1007, `_reporting_regression/__init__` 1050; `_mrmr_class` grew 4180 -> 4256 | open (`test_no_file_over_1k_loc` red on master) |
| 5 | KG-3: `test_biz_val_hybrid_redundancy_chain_drops_redundant_bridge` (HybridSelector keeps the redundant bridge b, corr 0.86 < corr_thr 0.92) is xfail | open, fixable framework bug |
| 6 | `family_null_backend` kernel-tuning-cache sweep not populated on this machine | open |
| 7 | Four-pillar bench on CPU and with `MLFRAME_FE_GPU_STRICT=1` | pending item 1 |

Fixed in this pass, each with a test: multi-output `y` raveled into n * k labels in the FE stages (joint class code now); `basis_route_by_moments` on a constant column; string class labels in the raw-protection probe; GPU pinned-staging failure; `eval_rows` for a screen subsample in the honest null.
