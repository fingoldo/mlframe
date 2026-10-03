# 07 Performance hot paths (read-only audit, 2026-10-04)

## Scope and method
Static review plus micro-measurements. Read audits/full_audit_2026-09-20/performance.md (PRF-01..07, all RESOLVED) and CLAUDE.md perf sections first, then grepped src/mlframe for: np.add.at, iterrows/itertuples, plotly add_annotation/add_shape loops, re.compile in functions, min-columns/rows dispatch thresholds, n_perm/n_boot defaults, non-cache njit.
Time-boxed pass; no cProfile of a full synthetic fit was run (not done, so no wall-share ranking is measured). Checkpoint 1: prior audits and grep sweep done. Checkpoint 2: add.at measured, thresholds read.

## Findings
| ID | file:line | sev | evidence | cost / potential | fix + validation | Disposition |
|---|---|---|---|---|---|---|
| PERF-01 | feature_selection/filters/_cat_pair_fe.py:415-416,428-429; _count_freq_interaction_fe.py:343-344,363-364 | P3 | `np.add.at(cell_sum, codes64[train_mask], y_arr[train_mask])`, per fold | Measured numpy 2.3.5, n=2M, 50 cells: add.at x2 = 18 ms vs bincount x2 = 23 ms. add.at is NOT slower on this numpy; bincount gives no win. Per-fold boolean-mask copies are the only waste (n_folds x mask gathers), est. well under 1% of a fit | Optional: one njit pass per fold or sorted-fold single pass; bit-identity required (same summation order; bincount reorders vs add.at is NOT guaranteed identical). Low value | OPEN |
| PERF-02 | feature_selection/filters/_cat_pair_fe.py:499, _cat_triple_fe.py:336,497; feature_engineering/transformer/apriori_itemsets.py:118 | P3 | `for _, row in keep.iterrows():` | Iterates over top_k kept candidates (small, bounded by top_k) with heavy per-row work inside; iterrows overhead is negligible | Not worth changing | OPEN |
| PERF-03 | training/composite/discovery/_corr_numba.py:62-63 (`_MIN_ROWS=20_000`, `_MIN_COLS=64`) via _ktc_dispatch.py:258 | P3 | `fallback = "numba" if (n_rows >= min_rows and n_cols >= min_cols) else "numpy"` | Thresholds are only the KTC fallback; `_lookup_backend("composite_corr_dispatch", ...)` is consulted first, so this conforms to doctrine. Residual risk: on a host with no tuned entry, inputs with <64 cols or <20k rows use numpy; whether numba wins there is unmeasured (unverified) | Run the KTC auto-tune sweep on dev host and confirm fallback matches measured crossover; selection-neutral (corr values bit-comparable to ~1e-12) | OPEN |
| PERF-04 | reporting/renderers/plotly.py:395,734-739 | P3 | `fig.add_annotation(` / `fig.add_vline(x=_v, **line_kw)` inside a per-reference-line loop (`for _i, _v`) | Loop is over reference lines (few), annotation only for `_i == 0`, but `add_vline` per line re-validates the shapes tuple (O(k^2) in k lines). k is small in practice; heatmap/line/scatter/network renderers already batch (see _plotly_line.py:145) | Batch vlines into one `fig.update_layout(shapes=tuple)` only if k can exceed ~50; pixel-identical output check | OPEN |

## Verified already-fine (not worth optimizing)
- Bootstrap family: fused prange njit bundle exists (evaluation/_bootstrap_fused_binary_bundle.py:48,112); evaluation/bootstrap.py:185 carries a bench-attempt-rejected note.
- Permutation tests early-stop: filters/permutation.py:309-311 and 375 stop after `min_perms` via Wilson CI; no unbounded n_perm without exit seen.
- Heatmap per-cell annotations are batched and capped (_plotly_heatmap.py:270, plotly.py:184 `_HEATMAP_CELL_TEXT_MAX`).
- Regex: the in-function `re.compile` hits (_mrmr_class_transform.py:79, postanalysis.py:32) are one-shot per call, not in loops; _phase_train_one_target_mlp_helpers.py:43 is cached. re's own cache covers the `re.match/search` sites.
- Per-fit logger.info in _screen_predictors.py (lines 349-362, 681, 844) are per-call, not shown to be inside inner loops (844 not inspected in context: unverified).
- No `@njit` without cache found by the grep was a real hot kernel (matches were docstrings/comments or use NUMBA_NJIT_PARAMS; params contents not inspected: unverified).

## Verdict
No P1 or P2 found in this time-boxed pass; the tree has been through many perf rounds. Counts: P1 0, P2 0, P3 4. Not done: end-to-end cProfile of a synthetic fit, per-column loops inside MRMR/training core, pandas/polars conversion redundancy. A follow-up with a profile is needed before declaring the class clean.
