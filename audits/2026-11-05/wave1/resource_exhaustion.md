# Wave 1 (2026-11-05), attack class A4: untrusted input and resource exhaustion

Read-only audit by one Opus 5.5 low-effort agent. Everything was checked by reading the cited lines; nothing was run and no file was edited. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| R4-01 | High | `src/mlframe/reporting/charts/pdp_ice.py:471-490` (via `_pdp_carrier.py:73-95`) | A categorical PDP axis sweeps every category with no cap, so the tiled predict block holds `sample x n_categories` rows, and there is one predict per outer category. |
| R4-02 | High | `src/mlframe/metrics/classification/_weighted_kappa.py:61,74-78` | The class count `n` is taken from the largest label value, then several dense `n x n` float64 matrices are built (`E`, `diff`, `W`, products). |
| R4-03 | Medium | `src/mlframe/metrics/classification/_threshold_optimization.py:219,226` | `optimal_threshold_bootstrap_ci` builds the full `(n_boot, n)` int64 index matrix in one call and sorts it, with no byte ceiling. |
| R4-04 | Medium | `src/mlframe/votenrank/rank_percentile_stacking.py:28,74,81` | With smoothing on, it builds dense `(n_query, n_oof)` float64 matrices with no size guard; only the docstring warns. |
| R4-05 | Low | `src/mlframe/reporting/charts/slice_finder.py:386,392` | All `p(p-1)/2` pairs are built as a Python list before the `max_combos` cap, even when `max_arity=1`; the triples list is built the same way. |
| R4-06 | Info | `src/mlframe/evaluation/blend_source_selection.py:85-99` | It warns above `_PAIRWISE_MATRIX_WARN_N` but still builds two `n x n` bool matrices plus `triu_indices` (two int64 arrays). |

### R4-01: categorical PDP axis is unbounded

`grid0`/`grid1` cover the full category list (`list(carrier[col].cat.categories)`, or the polars Categorical/Enum categories); `grid=20` is ignored for categoricals. With the default `sample=2000` and a pandas categorical of 10,000 categories:
- `g1 = 10,000`, so the tiled frame has `m*g1 = 2e7` rows x `n_cols`: about 16 GB at 100 float64 columns.
- `inner_values` is built as a 2e7-element object array through a Python list comprehension.
- If the outer feature is also that categorical, that is 10,000 predicts over 2e7 rows each, about 2e11 row predictions.

A polars `pl.Categorical` under the global string cache can report an even larger category set.

### R4-02: weighted kappa sized by label value

The labels are only checked to be non-negative; they do not have to be dense. Ratings coded as raw IDs, or a regression target cast to int, with a value of 100,000 give `n = 100,001`: about 80 GB for each n^2 float64 matrix, with four or more alive at once. A label of 1e6 gives about 8 TB each. An explicit `n_classes` does not help: `n_classes=None` is the default and a value smaller than the largest label raises an error.

### R4-03: threshold CI index matrix

With n=1e6 and the default `n_boot=200` the matrix is 1.6 GB of int64. `.astype(np.int64)` on an int64 array makes a copy, so peak is about 3.2 GB, plus a sort of 2e8 elements. With `n_boot=1000` it is about 8 GB, 16 GB at peak. The same class was already fixed in `calibration/policy.py`, which caps at 1 GiB (`DEFAULT_RESAMPLE_MATRIX_MAX_BYTES`), and the reporting charts draw indices in chunks. This function has neither.

### R4-04: smoothed rank percentile

`rank_percentile_transform(oof, test, smoothing=0.3)`: `n_oof = 1e5` gives a 1e10-element matrix, about 80 GB for `z` and the same again for `ndtr(z)`; `n_oof = 1e6` gives about 8 TB. OOF vectors of this length are normal stacking input. There is no chunking and no size check.

### R4-05: slice finder pair list

`p = 20,000` columns gives 2e8 tuples, about 15-20 GB of Python objects, and the cap then keeps only `max_combos`. The triples list has the same build-then-truncate pattern for a large `three_way_top_features` (500 gives about 2e7 tuples). A smaller related point: at `slice_finder.py:163`, `codes` is a full `(n, p)` int64 array.

### R4-06: pairwise score matrices

`n` counts blend members, so the risk is small. A caller who passes per-row scores (n=1e5) still gets about 10 GB of bool matrices plus 40 GB of triu indices; the function warns and then allocates anyway.

## Not covered

- Same class outside the assigned scope: `training/baselines/_dummy_bootstrap.py:352,372` builds a `(n_resamples, n)` gather, and for multilabel a `(n_resamples, n, K)` one. It is reached only when the numba kernel fails. At n=1e6 with the default `n_resamples=1000` that is about 16 GB or more.
- Recursion depth, ReDoS and unbounded caches were only spot-checked by grep: no `lru_cache(maxsize=None)` hits in scope, and the only regexes found are `.*literal.*` patterns, which are linear.
- `feature_engineering/timeseries.py:917`, a `while True` cumsum-window loop that might fail to terminate on zero-valued windows, was not traced.
- Not reviewed in depth: `competition/`, `core/`, `preprocessing/`, `data/`, and the rest of the `feature_engineering/` entry points beyond the cdist/meshgrid grep.
- Clean on reading: `ltr.py` (query cap), the `calibration.py`/`binary.py` chart bootstraps (chunked), `_core_auc_brier.py:448` (chunked), `knn_shapley` (batched), `variance_gated_pairwise_diff` (byte ceiling), `leak_scan` (`max_derived_features` cap).
