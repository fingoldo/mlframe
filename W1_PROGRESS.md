# W1 correctness worktree progress

Bench-gated MRMR correctness items (N-F6, S-F3, N-F3). Worktree detached off origin/master.

## N-F6 (Chao-Shen null mismatch) — DOC + silent-noop fix — DONE
- Evidence: CS (`chao_shen_mi`) is standalone; grep shows no production null/relevance caller (only tests + micro-bench). `compute_relevance_score` njit has no CS branch (only use_su/use_mm/plug-in).
- `mi_correction='chao_shen'` passes validation but MRMR.fit only wires miller_madow, so chao_shen degrades to plug-in for BOTH observed and null -> matched estimator, NO mismatch. N-F6 = DOC.
- Adjacent real bug fixed (not silently ignored): MRMR.fit now logs a WARNING that chao_shen falls back to plug-in (`_mrmr_class.py:3384`).
- Bench: `_benchmarks/bench_nf6_chao_shen_null_status.py`. Test: `tests/feature_selection/test_mrmr_critique_nf6_chao_shen_null.py` (3 passed, 143s).
- FUTURE (separate, tracked): wire CS into relevance njit path via a joint-count CS kernel + biz-value bench; thread the null in the same change.

## S-F3 (JMIM `**(nexisting+1)` exponent) — DOC (keep exponent) — DONE
- Bench: `_benchmarks/bench_sf3_jmim_exponent_selection.py` (8 seeds, synergy fixture; MRMR redundancy_aggregator='jmim'; exponent vs discount-only correction, subprocess arms).
- Numbers: discount-only correction 0/8 seed wins, identical selection 7/8, regressed seed=1 (recall 0.8->0.6); mean recall exponent=0.800 vs corrected=0.775.
- Decision: exponent is LOAD-BEARING -> KEEP as default, DOC. Correction retained as off-by-default `MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY` (evaluation.py, bench-attempt-rejected note). Default-off is bit-identical (fleuret+jmim parity: 14 passed).
- Guard test: `tests/feature_selection/test_mrmr_critique_sf3_jmim_exponent.py`.

## N-F3 (`_perm_pvalue(full_budget=)` calibration) — DOC (keep) — DONE
- Bench: `_benchmarks/bench_nf3_perm_pvalue_calibration.py` (true null, 200 seeds, per-seed observed I(x1;y|x0) as bootstrapped_gain).
- Numbers: 107/200 early-break; INVARIANT holds (every early break has nfailed>=max_failed -> gain zeroed at fleuret.py:123); on the 90 KEPT candidates full-budget p == truncated p in 100% (nchecked==budget).
- Decision: full_budget is SELECTION-INERT (only changes the p of an already-rejected candidate whose confidence is discarded) -> correctly calibrated, DOC. Kept the deliberate break-position-independence contract and its pinned test.
- Guard test: `tests/feature_selection/test_mrmr_critique_nf3_perm_pvalue_calibration.py`.
