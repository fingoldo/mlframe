# LOOP9 (axis = SPEED): large-n regression hexbin/log-density render path

## Task
Profile `build_regression_panel_spec` / `compose_regression_figure` on the large-n hexbin/log-density pred-vs-actual + residual panels at n = 1M / 5M / 10M; optimize a real hotspot or document already-optimal with numbers.

## Verdict: OPTIMIZED (two bit-identical binning kernels). Not rejected.

Found two real, >10%-of-wall mlframe-side hotspots, both the SAME class of waste: numpy's `histogram2d` / `digitize` use an O(n log bins) `searchsorted` against the bin edges even though the edges are UNIFORM (`np.linspace`), where the bin index is a direct O(n) arithmetic `floor((v-lo)/width)`. Replaced both with an arithmetic indexer that is bit-identical to numpy (incl. ULP-nudged on-edge values, via a single edge-correction pass).

## cProfile breakdown (10M, 3x runs, cumulative)

### BEFORE -- SCATTER hexbin (1.07 s/call)
- `np.histogram2d` -> `histogramdd` -> `searchsorted`: 2.46s/3 = **0.82s (77% of panel)** -- the binning.
- `_finite_pair` (float64 copy + isfinite): 0.066s (6%).

### BEFORE -- full 4-panel (2.96 s/call)
- `_resid_vs_pred_panel`: 1.10s/call -- dominated by `np.digitize` (searchsorted) + per-bin percentile partition.
- `_scatter_panel` (hexbin): 1.07s/call (the histogram2d above).
- `_err_by_decile_panel`: 0.67s/call -- `np.quantile` k-way partition (already documented optimal).
- `_finite_pair` x12 (4 panels x 3 reps): 0.26s -- each panel independently re-derives the finite float64 arrays.
- `_resid_hist_panel`: 0.09s/call.

## The optimization
`_uniform_bin_index(v, edges, nbins)` -- arithmetic scale+floor, then one vectorized compare of each value against its bin's two edges to correct the rare FP slip near an edge. Bit-identical to `clip(searchsorted(edges, v, 'right')-1, 0, nbins-1)` and to `clip(digitize(v, edges[1:-1]), 0, nbins-1)`.
`_hist2d_uniform(...)` -- `_uniform_bin_index` per axis + one weighted `bincount`; drop-in for `np.histogram2d(...)[0]` on uniform edges.

Wired into:
- `_scatter_panel` hexbin branch (replaces `np.histogram2d`).
- `_resid_vs_pred_panel` (replaces `np.clip(np.digitize(...))`).

### Why bit-identical (not just "within rounding")
A bare arithmetic floor diverges from searchsorted on values nudged 1-2 ULP around interior edges (measured: 195/200 hist2d matrices and 5764 digitize elements differ). The edge-correction pass eliminates that: 0 mismatches over 200 ULP-stress seeds (hist2d) and 300 seeds (digitize, incl. ties / on-edge). For a log-density viz a 1-count shift would be invisible, but the kernels are exactly identical, so the existing parity tests and the heatmap pin hold.

## AFTER -- timings (best of 3, ms/call)

| n | SCATTER hexbin before | after | speedup | full 4-panel before | after | speedup |
|---|---|---|---|---|---|---|
| 1M  | 105.5  | 54.7  | 1.93x | 305.7  | 199.4  | 1.53x |
| 5M  | 494.8  | 302.9 | 1.63x | 1429.2 | 1209.2 | 1.18x |
| 10M | 1012.2 | 625.1 | 1.62x | 3037.6 | 2364.2 | 1.28x |

cProfile after confirms `searchsorted` is gone from the hexbin path entirely; remaining full-panel cost is the `np.quantile` / per-bin percentile partitions in ERR_BY_DECILE and RESID_VS_PRED, which are k-way-partition-bound and already documented optimal in-code (a global lexsort was measured ~4x slower).

## Remaining cost (documented as at-floor, not waste)
- `_err_by_decile_panel` `np.quantile`: k-way partial sort, irreducible O(n) partition. Already documented.
- `_resid_vs_pred_panel` per-bin `np.percentile([25,50,75])`: single partition per bin, beats global sort. Already documented.
- `_finite_pair` called once per panel (4x per figure): each is an O(n) float64 copy + isfinite. Could be hoisted to a single shared pass in `compose_regression_figure`, but each panel is a standalone public builder (called independently by integrators), so sharing would change the API contract; left as-is. ~0.06s/call/panel at 10M, < the binning win.

## Visual equivalence
`docs/gallery/regression/regression_hexbin_largen.png` + `regression_full.png` re-rendered: byte-identical to the committed PNGs (deterministic matplotlib + bit-identical binning), so no PNG diff to commit -- positive confirmation of visual equivalence.

## Tests
- `test_uniform_bin_index_matches_searchsorted_and_digitize_incl_ulp_edges` (120 seeds, incl. ULP/ties/on-edge).
- `test_hist2d_uniform_bit_identical_to_numpy_incl_ulp_edges` (60 seeds).
- Existing `test_scatter_density_heatmap_above_threshold`, `test_decile_vectorization_parity_vs_rank_reference`, `test_resid_vs_pred_band_parity_vs_per_call_percentile` still pass.
- Full reporting suite: 16 + 165 passed.

## Bench (committed, re-runnable)
`scripts/bench_regression_largen_render.py --n 1000000 5000000 10000000 --repeats 3 [--profile]`

## Commit
5b610a7a -- perf(reporting): O(n) uniform binning for large-n regression hexbin + resid-vs-pred
