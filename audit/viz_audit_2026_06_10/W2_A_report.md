# W2-A — LTR chart kernels (finish stream)

Owned files: `src/mlframe/reporting/charts/ltr.py`, `src/mlframe/metrics/ranking.py`,
`tests/reporting/test_charts_ltr.py`, NEW `tests/reporting/test_ltr_charts_perf.py`,
bench `audit/viz_audit_2026_06_10/bench_viz_ltr.py`.

The prior agent's on-disk edits (batched kernels + NDCG_BY_QSIZE panel) were verified, NOT
redone. This stream confirmed correctness, added parity/sanity tests, extended the bench to 2M
plus a cProfile pass, and committed.

## Per-finding status

| Finding | What it was | Status | Evidence |
|---|---|---|---|
| PERF-1  | `_ndcg_dist_panel` converted full y_true/y_score to float64 INSIDE the per-query loop (quadratic on int input) | RESOLVED (on disk) | float64 conversion now once in `_iter_group_slices`; 2M finishes in 0.11 s |
| PERF-9  | `_score_by_rel_panel` built grade list via python `set()`-comprehension over `.tolist()` | RESOLVED (on disk) | full-array `np.unique` (~20 ms @2M vs ~0.25 s) |
| PERF-10 | `_lift_panel` per-query python loop + inner per-rank loop | RESOLVED (on disk) | `_lift_curve_kernel` (njit prange, chunked race-free reduce); 0.094 s @2M |
| PERF-11 | `_mrr_dist_panel` per-query python loop reusing per-query logic | RESOLVED (on disk) | `_per_query_mrr_kernel` reuses `_mrr_one_query`; 0.055 s @2M |
| PERF-12 | `_ndcg_k_panel` ran 50 independent full `ndcg_at_k` passes (re-sort per k) | RESOLVED (on disk) | single `_summary_batched_kernel` pass with `eval_ks=1..max_k`; 0.144 s @2M |
| INV (new panel) | small-group NDCG inflation invisible | RESOLVED (on disk) | NDCG_BY_QSIZE panel added + wired into default 6-panel template |

## Correctness confirmation (task 1)

All verified empirically (warm numba, multiple seeds):

- `_per_query_ndcg_kernel` and `_per_query_mrr_kernel` reuse `_ndcg_one_query` /
  `_mrr_one_query`, so they are **bit-identical** to the per-query loop on int AND float
  relevance, with AND without score ties, including size-1 and all-zero-relevance groups.
- `NDCG_K` batched (`_summary_batched_kernel`) is **identical** (maxdiff 0.0) to per-k
  `ndcg_at_k`.
- `_ndcg_dist_panel` output is **identical** to the pre-batch computation (per-query NDCG@10
  over groups with size >= 2, NaN dropped) — pinned by
  `test_ndcg_dist_panel_identical_to_pre_batch_filter`.
- `_lift_curve_kernel`: **bit-identical (1e-16)** to the plain reference on distinct scores.
  On TIED scores it diverges from a default-quicksort reference by up to ~6e-3 — this is a
  pure tie-break ordering difference: the kernel uses `kind="mergesort"` (the same
  deterministic choice `_ndcg_one_query`/`_mrr_one_query` already make), which is the
  cross-run/cross-platform-correct behaviour, not a regression. Parity test uses a mergesort
  reference (exact) plus a distinct-scores test proving the divergence is tie-break-only, never
  logic.
- Edge cases verified: size-1 groups (NDCG/MRR = 1.0 when rel>0), all-zero-relevance groups
  (NaN), empty input (0 groups, placeholder specs), chunked lift reduce deterministic across
  runs.

## Bench: before / after (task 3)

Before (per `performance.md`, pre-batch python loops): NDCG_DIST 0.66 s @100k / **42.1 s @400k**
(quadratic; 2M would not finish).

After (batched kernels, qsize=10), seconds:

| Panel | 100k | 400k | 2,000,000 | target @2M | met |
|---|---|---|---|---|---|
| NDCG_DIST | 0.006 | 0.033 | 0.113 | < 1.0 | yes |
| NDCG_K    | 0.008 | 0.028 | 0.144 | < 0.5 | yes |
| LIFT      | 0.004 | 0.017 | 0.094 | < 0.5 | yes |
| MRR_DIST  | 0.002 | 0.014 | 0.055 | < 0.3 | yes |
| NDCG_BY_QSIZE | 0.006 | 0.025 | 0.129 | — | — |

NDCG_DIST @400k: **42.1 s -> 0.033 s (~1280x)**; the 2M run now completes (~0.11 s) where the
quadratic version could not.

`compose_ltr_figure` (default 6-panel): 0.185 s @1M, 0.282 s @2M.

## cProfile + optimization (task 4)

cProfile `compose_ltr_figure` n=1M (full table: `bench_profile_ltr_compose.txt`): total ~0.16 s.
Top frames: `_ndcg_k_panel` ~0.05 s, shared `_per_query_ndcg10` kernel ~0.03 s,
`_score_by_rel_panel` ~0.02 s, `_lift_panel` ~0.02 s, `_iter_group_slices` (one-shot group
sort + float64 gathers) ~0.016 s. cProfile attributes numba kernel wall to the calling python
frame.

DISCARDED-WORK probe: `_ndcg_k_panel` calls `_summary_batched_kernel` and discards its MAP@k +
MRR outputs. A pruned NDCG-only kernel was microbenched warm, multi-repeat, two sizes:
n=1M 51.0 -> 45.5 ms (1.12x, bit-identical) but n=2M 80.8 -> 81.2 ms (0.99x). The 1M "win"
evaporates at 2M because the per-group sort (shared by all three metrics, and MAP reuses the
same score-sorted order) dominates. **REJECTED** as a size-inconsistent ~1.1x within numba-
parallel noise; not worth a second redundant kernel. Recorded in the bench `_PROFILE_NOTES`.

Verdict: no actionable in-file speedup beyond the batching already shipped; the per-group sort
is the floor (already parallel). Next lever if a huge-N single call ever dominates: radix/
counting group-sort on integer ids in `_iter_group_slices`, gated on id dtype.

## Tests (tasks 2, 5)

- NEW `tests/reporting/test_ltr_charts_perf.py`: **29 tests** — kernel parity (NDCG/MRR/NDCG_K/
  LIFT across distinct/tied x int/float), lift distinct-score bit-identity + determinism, edge
  cases (size-1, all-zero-rel, empty, ndcg_dist pre-batch parity), NDCG_BY_QSIZE content sanity
  (bar type, value range [0,1], log2 bins, per-bin counts sum to n_valid queries, small-group
  inflation > large-group, empty placeholder), default 6-panel compose regression + shared-cache
  reuse parity.
- `tests/reporting/test_charts_ltr.py`: orchestrator's NDCG_BY_QSIZE token re-frame committed
  with this work.
- Sweep `pytest tests/reporting -k "ltr"`: **63 passed**.
- `tests/reporting/test_charts_ltr.py` + `tests/reporting/test_panel_template_fuzz.py`:
  **40 passed** (fuzz auto-covers NDCG_BY_QSIZE via `ALLOWED_LTR_PANEL_TOKENS`).
- ranking-kernel (`test_ranking_summary_kernel_njit_kind_kwarg.py`) + `test_ranking_metrics.py`
  + full `tests/reporting`: **323 passed, 2 skipped** (skips = kaleido not installed,
  third-party, unrelated).

## Commits

<filled below>

## Deviations

- The lift kernel is NOT bit-identical to the *old quicksort* python on tied scores (~6e-3,
  tie-break only). This is an intentional determinism improvement consistent with the existing
  NDCG/MRR kernels; documented + pinned both ways (mergesort-ref exact, distinct-scores
  bit-identical). No opt-in flag — the deterministic path is the default.
