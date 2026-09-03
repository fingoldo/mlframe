# core infra A (core/, utils/, estimators/) -- mlframe audit

## Scope

All 35 `.py` files under the assigned cluster were read in full (no file was too large to review completely; the largest, `utils/_param_oracle.py` at 887 LOC, was read end-to-end).

- `src/mlframe/core/arrays.py`
- `src/mlframe/core/binning.py`
- `src/mlframe/core/composite_similarity.py`
- `src/mlframe/core/ewma.py`
- `src/mlframe/core/frame_compat.py`
- `src/mlframe/core/helpers.py`
- `src/mlframe/core/matrix_seriation.py`
- `src/mlframe/core/proportion_stats.py`
- `src/mlframe/core/recency_step_weight.py`
- `src/mlframe/core/recency_weights.py`
- `src/mlframe/core/robust_location.py`
- `src/mlframe/core/set_similarity.py`
- `src/mlframe/core/stats.py`
- `src/mlframe/core/__init__.py`
- `src/mlframe/core/_benchmarks/__init__.py`
- `src/mlframe/core/_benchmarks/bench_recency_step_weight.py`
- `src/mlframe/core/_benchmarks/profile_robust_location.py`
- `src/mlframe/utils/disk_cache.py`
- `src/mlframe/utils/eda.py`
- `src/mlframe/utils/experiments.py`
- `src/mlframe/utils/misc.py`
- `src/mlframe/utils/nan_safe.py`
- `src/mlframe/utils/safe_pickle.py`
- `src/mlframe/utils/text.py`
- `src/mlframe/utils/_param_oracle.py`
- `src/mlframe/utils/_param_oracle_store.py`
- `src/mlframe/utils/__init__.py`
- `src/mlframe/estimators/base.py`
- `src/mlframe/estimators/baselines.py`
- `src/mlframe/estimators/custom.py`
- `src/mlframe/estimators/early_stopping.py`
- `src/mlframe/estimators/early_stopping_monotonic.py`
- `src/mlframe/estimators/pipelines.py`
- `src/mlframe/estimators/__init__.py`
- `src/mlframe/estimators/_benchmarks/bench_cpx39_decorrelator.py`

Total files reviewed: 35. Total LOC reviewed: 5990 (sum of `wc -l` over the above, matching the assigned scope's own line count exactly).

Several findings below were verified empirically (not just by inspection) by running the actual mlframe code in this environment: the `fit_bin_smoother`/`apply_bin_smoother` crash, the `get_model_best_iter` re-raise bug, and the `DiskCache` path-traversal write, each reproduced with a short standalone script against the installed package.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness/edge-case | src/mlframe/core/binning.py:71-95 | `fit_bin_smoother`/`apply_bin_smoother`/`bin_smooth` crash with `IndexError` on any constant-valued (zero-variance) column, even with fully default arguments. |
| F2 | P1 | correctness/edge-case | src/mlframe/core/arrays.py:271-304 | `topk_by_partition` with `axis=None` on a multi-dimensional array indexes the un-flattened array with flat indices, giving wrong values or an `IndexError`. |
| F3 | P1 | security/robustness | src/mlframe/utils/disk_cache.py:293-296 | `DiskCache` does not sanitize `key`; a key containing `..`/path separators writes the pickle (and its `.sha256` sidecar) outside `cache_dir` (path traversal), confirmed by reproduction. |
| F4 | P1 | test-gap / silent-failure | src/mlframe/estimators/pipelines.py:96-119, 156-157 | `optimize_pipeline_by_gridsearch` writes its CV-results dump via plain `joblib.dump` (no `.sha256` sidecar), but `replay_cv_results` in the same module fail-closed-refuses to load any file lacking a sidecar -- the module's own documented round-trip is broken by default. |
| F5 | P1 | silent-failure | src/mlframe/core/helpers.py:96-102 | `get_model_best_iter`'s `except (TypeError, ValueError): return int(val)` re-executes the exact same failing expression instead of falling back, so it re-raises instead of degrading gracefully as the docstring promises. |
| F6 | P2 | correctness/edge-case | src/mlframe/estimators/custom.py:403-426 | `MyDecorrelator.correlated_features_` is keyed by DataFrame column names when `fit` sees a DataFrame but by integer positions when it sees an ndarray; calling `transform` with the *other* container type silently keeps every column instead of dropping the correlated ones. |
| F7 | P2 | correctness/edge-case | src/mlframe/core/set_similarity.py:32-44 | `_counts`'s dispatch to the boolean-mask code path is triggered whenever *either* input is bool-dtype (not both), so passing one genuine boolean mask alongside a same-length non-boolean array silently reinterprets the latter as a mask via `astype(bool)` instead of raising. |
| F8 | P2 | correctness/edge-case | src/mlframe/utils/misc.py:164-168 | `get_pipeline_last_element` references the for-loop variable `elem` after the loop; an empty `Pipeline.named_steps` raises an opaque `UnboundLocalError`/`NameError` instead of a clear error. |
| F9 | P2 | architecture/docs | src/mlframe/utils/misc.py:89-136 | `hygienic_fit`/`_restore_caller_frame_columns` only restore column schema for `pandas.DataFrame` inputs; a bare polars `DataFrame` passed directly (bypassing the MRMR-internal pandas bridge) keeps any columns a wrapped `fit` engineers into it. |
| F10 | P2 | robustness | src/mlframe/core/matrix_seriation.py:56-70 | `spectral_seriation` performs no NaN/Inf validation on `M`; a NaN/Inf entry surfaces as a generic `numpy.linalg.LinAlgError: Eigenvalues did not converge` rather than a clear `ValueError` naming the actual problem. |

### F1 -- `fit_bin_smoother`/`apply_bin_smoother` crash on a constant column (P0)

```python
>>> x = np.array([5.0]*20)
>>> sm = fit_bin_smoother(x, n_bins=10, binning="quantile")  # default n_bins, default binning
>>> apply_bin_smoother(x, sm, strategy="mean")
IndexError: index -1 is out of bounds for axis 0 with size 0
```

Root cause: when every finite value in `x` is identical, `np.quantile` produces `n_bins+1` identical edges; `np.unique(edges)` then collapses them to a single edge, so `n_actual = len(edges) - 1 == 0`. `fit_bin_smoother` happily returns a smoother with zero-length `bin_mean`/`bin_median` arrays and an empty `interior`. `apply_bin_smoother` then computes `n_bins = len(edges) - 1 == 0` and does `np.clip(np.digitize(...), 0, n_bins - 1)` i.e. `np.clip(..., 0, -1)`, which clips every index to `-1`, and `reps[-1]` on a zero-length `reps` array raises. This is not a rare `n_bins=1` corner case -- it fires for the **default** `n_bins=10` on any constant/near-degenerate column, a realistic occurrence (a zero-variance feature, an all-same-category-coded numeric column, a post-filter column that happens to be constant in a subset). `bin_smooth` (the fit+apply convenience wrapper) inherits the same crash. No test in `tests/core/test_binning.py` exercises a constant column. Fix direction: guard `n_actual == 0` in `fit_bin_smoother` (either raise a clear `ValueError("column has < 2 distinct finite values; binning is undefined")`, matching the existing "no finite values" guard, or special-case it to a single degenerate bin whose representative is the constant value) and/or guard `n_bins <= 0` in `apply_bin_smoother` to raise the same clear error instead of silently clipping into an invalid range.

### F2 -- `topk_by_partition` mishandles `axis=None` on 2-D+ arrays (P1)

`topk_by_partition(arr, k, axis=None, ...)` documents `axis` as accepting `int | None`, and the empty-selection branch already special-cases `axis=None` correctly via `np.take(..., axis=None)`. But the main path does:

```python
vals_part = np.take_along_axis(arr, ind, axis=axis) if axis is not None else arr[ind]
...
val = np.take_along_axis(vals_part, ind_part, axis=axis) if axis is not None else vals_part[ind_part]
```

`ind` here comes from `np.argpartition(arr, kth, axis=None)`, which returns **flat** indices into the *flattened* array (range `0..arr.size-1`). But `arr` at this point is still the original (possibly multi-dimensional) array -- it is only sign-negated or `.copy()`-ed, never flattened. `arr[ind]` therefore performs fancy indexing along axis 0 using those flat indices, either raising `IndexError` (index >= arr.shape[0]) or, when it happens not to raise, silently selecting the wrong rows entirely. This only manifests when `arr.ndim >= 2` and the caller relies on the documented `axis=None` default -- no current in-repo caller does this (the only two callers in `tests/test_numeric_bug_sweep.py` use 1-D arrays, where the bug is invisible because flat index == row index). Any future/external caller passing a 2-D score matrix with the documented default `axis=None` will hit it. Fix direction: flatten `arr` (e.g. `arr = arr.ravel()`) at the top of the function when `axis is None`, mirroring what `np.argpartition`/`np.argsort` already do internally for that mode, and add a regression test exercising `axis=None` on a genuinely 2-D array.

### F3 -- `DiskCache` key is not path-traversal-safe (P1)

```python
>>> c = DiskCache(sub_dir)
>>> c.put("../evil", {"x": 1})
>>> os.listdir(parent_of(sub_dir))
['cache', 'evil.pkl', 'evil.pkl.sha256']   # written OUTSIDE cache_dir
```

`_key_path` builds `self.cache_dir / f"{key}.pkl"` with no validation that `key` stays within `cache_dir` (no traversal-character check, no allowlist on the key alphabet). Every internal caller today builds keys via `hash_object`/`compose_key`/`hash_array_summary` (safe hex digests), so this is not exploited in the current call graph, but the class is a shared, widely-reused utility (99 files reference it across the codebase) and its public `get`/`put` contract accepts an arbitrary `str`. A future or external caller composing a key from a less-trusted string (a filename, a user-supplied label) would silently write/read files outside the intended cache directory. Fix direction: validate `key` against a restrictive pattern (e.g. hex/alnum + limited length, matching what `_hasher().hexdigest()` produces) in `_key_path`, or explicitly resolve+`is_relative_to`-check the resulting path before every filesystem operation.

### F4 -- `optimize_pipeline_by_gridsearch` writes a dump `replay_cv_results` refuses to read (P1)

`optimize_pipeline_by_gridsearch` (estimators/pipelines.py:157) persists results via a bare `joblib.dump(cv_results, ...)` -- it never calls `write_sidecar` (unlike every other joblib/pickle writer in the codebase, e.g. `disk_cache.py`, `training/composite/cache_store.py`, `training/feature_handling/cache.py`, `feature_selection/wrappers/rfecv/__init__.py`, all of which pair a dump with `write_sidecar`). `replay_cv_results` (same file, line 110) calls `_verify_sidecar(fname)` and raises `ValueError("sha256 sidecar mismatch ... refusing to load")` when no `.sha256` sidecar exists next to the file, which is exactly pyutilz's documented fail-closed default (`verify_sidecar` returns `False` when the sidecar is missing, unless `MLFRAME_ALLOW_UNVERIFIED_PICKLE` is set). So the dump artifact this module's own grid-search helper produces can never be loaded back by this module's own replay helper without the caller manually generating a sidecar or setting the env var -- neither of which is mentioned anywhere near either function. There is no test for either function (`grep` across `tests/` found zero references to `optimize_pipeline_by_gridsearch`/`replay_cv_results`), so this gap is invisible to CI. Fix direction: call `write_sidecar(path)` right after the `joblib.dump` in `optimize_pipeline_by_gridsearch` (mirroring `DiskCache.put`'s pattern), and add a round-trip test.

### F5 -- `get_model_best_iter`'s except-clause re-raises instead of falling back (P1)

```python
for field in ("best_iteration_", "best_iteration", "best_epoch"):
    val = getattr(real_model, field, None)
    if val is not None:
        try:
            return int(val)
        except (TypeError, ValueError):
            return int(val)          # <-- calls the SAME failing expression again
```

If `int(val)` raises `TypeError`/`ValueError` (e.g. a model exposes `best_iteration_ = "n/a"` or some other non-numeric sentinel -- plausible for a third-party estimator with an unconventional API, or a mocked/stubbed model in a test), the `except` branch calls `int(val)` again, which raises the identical exception, uncaught, propagating out of the function. Reproduced directly: `get_model_best_iter(obj_with_best_iteration_="not-an-int")` raises `ValueError` instead of falling through to the next field / `tree_count_` fallback / `None`, contradicting the function's own docstring ("Returns `None` ... when nothing is exposed"). Contrast with the `tree_count_` fallback a few lines below, which correctly does `except (TypeError, ValueError): pass`. `get_model_best_iter` is called from the live training loop (`training/_training_loop.py`, `training/_training_loop_refit.py`). Fix direction: change the inner `except` to `pass`/`continue` (matching the `tree_count_` pattern) so a bad value on one field falls through to the next.

### F6 -- `MyDecorrelator` fit/transform type mismatch silently keeps all columns (P2)

`fit` always does `X = pd.DataFrame(X)` and stores `correlated_features_` as whatever `X.corr().columns` produced -- real column labels if the caller passed a `DataFrame`, or the default `RangeIndex` integers `0..p-1` if the caller passed an ndarray. `transform` branches on the *transform-time* input's type: `X.drop(labels=self.correlated_features_, ...)` for a `DataFrame`, else `[j for j in range(...) if j not in self.correlated_features_]` for an ndarray. If `fit` saw a `DataFrame` (so `correlated_features_` holds string column names) and `transform` is later called with a plain ndarray (integer keys), `j not in self.correlated_features_` is `True` for every `j` (a string set never contains an int) -- every column is silently kept, defeating the whole point of the transformer, with no error or warning. The class's own docstring ("TODO: TEST PROPERLY") flags it as under-tested. Fix direction: store the fitted feature identity in one canonical form (e.g. always integer positions, resolved against `feature_names_in_` at `fit` time) and validate at `transform` time that the input's column identity matches what was fit on.

### F7 -- `set_similarity._counts` bool-detection is asymmetric (P2)

```python
if a_arr.dtype == bool or (a_arr.dtype != object and b_arr.dtype == bool):
```

This takes the boolean-mask code path whenever *either* array is boolean (as long as the other isn't object-dtype), not only when *both* are. Two same-length arrays where `a` is a genuine boolean mask and `b` is, e.g., an int array of item ids (not a mask) will have `b_arr.astype(bool)` silently applied (nonzero -> True) rather than raising or falling back to set semantics. This only matters when a caller passes mismatched semantics (a mask + a non-mask of equal length), which the docstring doesn't explicitly forbid at the type level. Fix direction: require both arrays to be bool-dtype (`a_arr.dtype == bool and b_arr.dtype == bool`) before taking the mask branch; anything else falls through to set semantics (which already handles arbitrary iterables including numeric arrays via `set(a)`).

### F8 -- `get_pipeline_last_element` on an empty Pipeline (P2)

```python
def get_pipeline_last_element(clf) -> object:
    for elem in clf.named_steps.values():
        pass
    return elem
```

If `clf.named_steps` is empty, the loop body never executes and `elem` is never bound, so `return elem` raises `UnboundLocalError`/`NameError` rather than a clear, actionable error. An empty `sklearn.Pipeline` is unusual but not impossible (a caller building a pipeline with a variable step list that ends up empty after filtering). Fix direction: raise a clear `ValueError("clf.named_steps is empty")` when the pipeline has no steps.

### F9 -- `hygienic_fit` restores pandas column schema only (P2)

`_restore_caller_frame_columns` (used by `hygienic_fit`) is guarded by `isinstance(X, _pd.DataFrame)`; if `X` is a bare polars `DataFrame` (not routed through MRMR's internal Arrow-backed pandas view, e.g. a caller invoking a `@hygienic_fit`-decorated selector directly on polars data), any columns the wrapped `fit` engineers into `X` in place are never dropped, silently leaking columns into the caller's frame -- exactly the failure mode the decorator exists to prevent, just for the other supported frame library. Fix direction: extend `_restore_caller_frame_columns` with a `hasattr(X, "columns") and hasattr(X, "drop")`-style duck-typed branch (or an explicit polars check) so the invariant holds for both frame types the codebase supports.

### F10 -- `spectral_seriation` gives an opaque error on NaN/Inf input (P2)

`spectral_seriation` validates shape (`M` must be square 2-D) but not finiteness. A NaN/Inf entry in `M` propagates into the Laplacian and surfaces as `numpy.linalg.LinAlgError: Eigenvalues did not converge` from deep inside `np.linalg.eigh`, which does not name the actual problem (a non-finite input) and could equally be a genuine numerical-convergence failure on a valid matrix. Fix direction: add an explicit `if not np.all(np.isfinite(M)): raise ValueError(...)` guard alongside the existing shape check.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | test-coverage | tests/core/test_binning.py | Add a constant-column / near-degenerate-quantile regression test for `fit_bin_smoother`/`apply_bin_smoother`/`bin_smooth` (pairs with F1's fix). |
| PR2 | test-coverage | tests/test_numeric_bug_sweep.py | Add a 2-D `axis=0`/`axis=1`/`axis=None` test for `topk_by_partition` (pairs with F2's fix); today only 1-D inputs are exercised, which is exactly the shape that hides F2. |
| PR3 | test-coverage | tests/estimators/ (no such file exists today) | `estimators/pipelines.py` has zero test coverage for `optimize_pipeline_by_gridsearch`/`replay_cv_results`/`compare_cv_metrics`/`agg_pipeline_metric`; add a round-trip test (would have caught F4). |
| PR4 | code-quality | src/mlframe/core/arrays.py:290-293 | `topk_by_partition`'s `part_kth = min(k - 1, n_along_axis - 1) if k == n_along_axis else k` is correct but non-obvious (relies on `np.argpartition`'s guarantee holding for either `kth=k-1` or `kth=k`); a short inline comment explaining why both branches are safe would save the next reader re-deriving it. |
| PR5 | robustness | src/mlframe/utils/disk_cache.py | Document (or enforce) that `DiskCache` keys must be pre-hashed/hex-safe strings; the module docstring already states the design intent ("content-addressable, not path-keyed") but the code doesn't enforce it (pairs with F3). |
| PR6 | ML-practice | src/mlframe/core/composite_similarity.py | `fit_composite_similarity`'s coordinate-descent grid search has no test asserting it actually beats an equal-weight baseline on a synthetic where one block is pure noise and another is fully informative (a `test_biz_val_*`-style test per this repo's convention); current tests were not located under `tests/core/` for this module during this review (see Coverage notes). |
| PR7 | perf | src/mlframe/utils/_param_oracle.py:298-304 | `default_fingerprint`'s per-column cardinality loop (`for j in range(a.shape[1]): ... np.unique(col)`) is O(p) Python-level calls to `np.unique`; for wide frames (large `p`) this could dominate the otherwise-vectorised fingerprint computation. Given fingerprinting runs on every oracle-decorated call, worth a quick cProfile check on a realistic wide-`p` shape (this module explicitly targets MI-scorer/FE-recipe dispatch, which is exactly a wide-`p` regime) to see if it's worth vectorising (e.g. via a single `pandas.DataFrame.nunique(axis=0)`-style pass) or capping the columns sampled. |

## Coverage notes

- I did not locate a dedicated `tests/core/test_composite_similarity.py` (or equivalent) via the file layout browsed during this review; I did not exhaustively enumerate every test file in `tests/` (out of scope for this read-only source audit), so I cannot assert with certainty whether `composite_similarity.py` has zero, partial, or full test coverage -- PR6 above is phrased conditionally for that reason.
- `src/mlframe/utils/_param_oracle.py` and `_param_oracle_store.py` are large (887 + 169 LOC) and interact with `pyutilz.performance.kernel_tuning.cache` and `filelock`, both external to this cluster; I read the full source and traced the logic but did not execute a live concurrent-writer stress test of `_ParquetStore.append`'s file-lock path (would require a multi-process harness, which is a mutation/benchmark activity outside this read-only audit's remit).
- The two `_benchmarks/` scripts and the `estimators/_benchmarks/bench_cpx39_decorrelator.py` script were read for correctness of the benchmark logic itself (not run), since running them would execute git subprocess calls and timing loops outside the read-only mandate.
- `mlframe.core.helpers` imports `mlframe.config` and `pyutilz.polarslib`, both outside this cluster; I only read their referenced symbols' usage sites, not their internals, per the audit's own scoping rule for cross-package symbols.
