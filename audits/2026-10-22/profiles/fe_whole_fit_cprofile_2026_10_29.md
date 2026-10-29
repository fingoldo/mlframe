# Whole-fit cProfile of the all-flags MRMR fit (n=20000, default GPU mode), 2026-10-29

`D:/Temp/fit_top.py` (warm fit on other data, cProfile around the second fit, pstats in `D:/Temp/fit_prof.pstats`). Wall under cProfile 85 s (the same fit runs in about 64 s without it). Attribution walks the callers up to the first mlframe frame; use shares, not exact seconds.

## Self-time leaders (not attributable to one family)
- `threading.Lock.acquire` 12.3 s: the main thread waiting on thread pools (the work is in the workers, invisible to cProfile). Pool waiters by mlframe function: `_orth_auto_scorer_fe._bootstrap_lcb_tables` 6.4 s (real cumulative), `_mi_greedy_cmi_fe.score_candidates_by_cmi` 4.6 s (real cumulative, two calls from `score_grouped_agg_by_cmi_uplift`), `_orthogonal_routing_fe._enumerate_routing_cells`, `_dcd_swap_*`.
- `numpy ufunc.reduce` 4.5 s over 182711 calls: `np.max` on int8 code matrices in `_family_maxt_stream._code_itemsize` was 2.6 s of it. FIXED in commit 2fff6814b (int8 needs no value scan): `_code_itemsize` 2.70 s to 0.02 s over its 7 calls, same results on int8/uint8/int16/int64/empty input.
- `cupy .get()` 4.1 s over 868 calls: device syncs; 2.1 s of it is the deferred host-code fill in `_pairs_chunks._compute_one_fe_chunk` (the host family gate reads codes the device just produced).
- `lightgbm Booster.update` 3.5 s (3000 iterations), `cupy argsort` 3.1 s (the conditional gate rank binner, see the gate profile).

## Biggest mlframe blocks (cumulative)
`_run_fe_stage_cascades` 51.5 s; `fe_decide_on_subsample` 23 s over 25 calls (a wrapper over the closed-form families, not its own cost); `check_prospective_fe_pairs` 10.4 s (pair scoring 9.3 s over 34 pairs, of which the pair family gate 4.1 s, 2.6 s of that was `_code_itemsize`); `build_fe_operand_pool` 6.5 s; `hybrid_orth_mi_auto_scorer_fe` 6.5 s; `_stage_composite_group_agg` 6.0 s; `apply_surrogate_gbm_seeder` 5.7 s; `_stage_conditional_gate` 5.0 s; `_gate_engineered_accuracy` 4.9 s.

## Not changed, with the reason
- `_family_flags` (0.95 s): 0.61 s is the exact permutation test; the rest is dedupe and triage.
- `_materialise_extval_njit` (0.98 s, 19 calls): serial nogil kernel by design (a prange nested in the threading layer deadlocks, see `_fe_use_parallel_kernels`).
- `np.quantile` binners (`_mi_greedy_cmi_fe_binning._quantile_bin` 2.6 s, `_grouped_quantile_fe._bin` 1.7 s): introselect partition is already the cheapest exact path (documented as a rejected rank-binner attempt in the code).
- The conditional gate rank sort: 90 percent of the sorted columns come from `select`-mode specs (666 of 743 specs, 11322 of 12631 columns); an exact rank derived from one operand sort works for mask mode (prototype exact on four tie patterns, 1.6x on the sort) but covers only 10 percent of the columns, so it is not worth wiring in.
