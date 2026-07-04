# W1 correctness worktree progress

Bench-gated MRMR correctness items (N-F6, S-F3, N-F3). Worktree detached off origin/master.

## N-F6 (Chao-Shen null mismatch) — DOC + silent-noop fix — DONE
- Evidence: CS (`chao_shen_mi`) is standalone; grep shows no production null/relevance caller (only tests + micro-bench). `compute_relevance_score` njit has no CS branch (only use_su/use_mm/plug-in).
- `mi_correction='chao_shen'` passes validation but MRMR.fit only wires miller_madow, so chao_shen degrades to plug-in for BOTH observed and null -> matched estimator, NO mismatch. N-F6 = DOC.
- Adjacent real bug fixed (not silently ignored): MRMR.fit now logs a WARNING that chao_shen falls back to plug-in (`_mrmr_class.py:3384`).
- Bench: `_benchmarks/bench_nf6_chao_shen_null_status.py`. Test: `tests/feature_selection/test_mrmr_critique_nf6_chao_shen_null.py` (3 passed, 143s).
- FUTURE (separate, tracked): wire CS into relevance njit path via a joint-count CS kernel + biz-value bench; thread the null in the same change.
