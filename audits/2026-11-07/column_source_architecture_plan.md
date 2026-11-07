# Plan v3: memory cuts first, then a subsample-fit mode with a streamed verification pass

Revised after four independent reviews (v1: correctness/concurrency and scale/polars; v2: feasibility/gates and statistics, Opus 5.5 medium). v1 and v2 assumed the fit could run on codes without raw data; the code says otherwise. What survives is smaller and more honest. Statements below were checked against the code by the reviewers (file:line given) unless marked (unverified).

## What the code actually does (limits on any design)

- The fit needs the full frame resident today: a `LazyFrame` is collected at fit entry (`mrmr/_mrmr_class_fit_helpers.py:178`, again in `discretization/__init__.py:141`); the FE cascades run before discretization and append float columns at full n (`_fit_impl_core_parts.py:585-610`); `transform` rebuilds every engineered column from the input X (`_mrmr_validate_transform.py:652-918`), so transform on a terabyte frame is unsolved by any fit-time change.
- `fe_decide_on_subsample` already decides FE families on a 30k-row subsample by default (`mrmr/_mrmr_class.py:1040`) but then rebuilds every chosen column at full n (`_mrmr_fit_impl/_helpers.py:467-477`), skips out-of-fold families (`:370-381`), and returns `fe_append_columns(X, ...)` (a `concat` of all of X) which the caller appends to `X_acc` again: two full copies per family on pandas (`_fe_stage_cascade_early_a.py:56-71`).
- Default binning is MDLP, which depends on the target; its honest null re-bins the whole raw frame per permutation (`_fit_impl_stages/_binning.py:216-247`). Codes are therefore not computable without the target, and the permutation null needs raw floats (or a subsample), not codes.
- Codes today are int32 by default (`_mrmr_class.py:334`), signed, NaN shifted to 0 or to `n_bins` depending on the strategy (`_discretization_dataset.py:539-571`); numpy-level joint codes like `a*nb+b` overflow on uint8.
- Engineered columns come back in mixed dtypes (some cast to float32, `_orth_basis_recipes.py:269`, `_binned_numeric_agg_fe.py:410`).
- `self` holds frames in places: `_partial_fit_X_buffer_` (pickled, `_mrmr_partial_fit.py:235,264`), `_honest_null_codes_fn` closure (`_binning.py:310`), `_pandas_frame_for_target_cleanup` (`_inputs.py:241`).
- Greedy selection is sequential: each round's conditional gain depends on the previous argmax, so scoring k features on full codes on disk costs about 2k passes plus retries, not 3.
- A full-n permutation null is compute-bound (about B*K*n = 999*750*1e9 = 7e14 count updates at 1B rows), not I/O-bound.

## Design that survives

**A. Memory cuts inside the current architecture** (no behaviour change, selection must be identical):
1. `fe_decide_on_subsample` returns only the new columns; callers merge them once into `X_acc`. Drops one full-frame `concat` per family on pandas.
2. Other `pd.DataFrame(dict)` builders get the single-copy builder (done for composite group-agg).
3. Remove the 15 `fe_to_pandas` full conversions (block reads or numeric float32 view; the function stays as a size-guarded fallback).
4. `CodeStore` with TODAY's code values and dtype, filled block by block (no narrowing yet).

**B. Subsample-fit mode for frames beyond RAM** (new, opt-in by size):
1. One streamed read of the target alone, then a hash-partitioned row split by `splitmix64(seed, row_id)` (or by group/entity hash, or time block when group ids or a time column are given): A (choose), D (exact counts for rare levels and class totals), C (certify). Class-aware: keep all minority rows up to a cap, Bernoulli-sample the rest at probability pi_c with weights 1/pi_c and Kish n_eff in the Miller-Madow terms. Whole groups sampled by hash for group-aware and time families; y shifted circularly for time series nulls.
2. Run today's fit unchanged on A (in RAM): MDLP, honest null, FE, dedup, greedy selection.
3. Freeze the selected path and recipes. One streamed pass over the rest builds codes for the selected columns only (edges frozen from A, a reserved code for unseen values) and certifies on C: G-test 2N*CMI ~ chi2(df from occupied cells on C) (`_analytic_mi_null.py`), Holm over k, an effect-size floor (lower confidence bound MI >= delta) at very large n. A failed step truncates the path; the suffix is re-selected on A and re-tested on C.
4. Absolute MI floors calibrated at m rows are rescaled to the n they are applied at (bias is about df/2m).

Not claimed: transform on a terabyte frame (each engineered column is replayed from the input X; that needs block-wise replay, a separate item); streaming sketch edges (edges from a 2M-row subsample already give rank error about 0.0012; a t-digest is worse on tied values and inflates occupied-bin df; if exact edges are needed the radix histograms in `_order_statistics_f32` merge across row blocks in two or three raw passes).

Dropped from v2: code-first engineered columns for out-of-fold families (impossible), uint8 codes before a cast audit, one-pass permutation null, a float arena wired into the stage seam (the existing `ColumnArena` stays unused until profiling shows `pd.concat` still dominates after A.1 to A.3).

## Phases

| # | Step | Gate |
|---|------|------|
| 0 | Safety net. The "old vs new in one process" harness is impossible (two package versions cannot be imported together): extend `tests/feature_selection/golden/capture_baseline.py` and `_capture.py` to capture from a reference commit in a subprocess and compare; add polars input, `transform`, weights, categoricals, non-Range index, NaNs, pinned `MLFRAME_FE_STAGE_CONCURRENCY` and CPU vs GPU; golden dtype table per FE family; pickle/clone gate incl. a model pickled after `partial_fit` (size check); peak-RSS regression gate per scenario; hook counting full-frame `to_pandas`; rollback via the existing per-feature env switches (`_fe_env_switch.py`); profile: tracemalloc deep trace per stage at 100k/300k/1M | snapshots green on HEAD; tables saved in `profiles/` |
| 1a | FIRST SLICE (about a day): `fe_decide_on_subsample` returns new columns only, callers merge once | golden snapshots equal; 300k peak lower |
| 1b | Single-copy builders for the other `pd.DataFrame(dict)` sites | peak lower, frames equal |
| 2 | Remove `fe_to_pandas` full conversions (`_fe_stage_cascade_early_b.py` and the `fe_decide_on_subsample` fallbacks `:414,440,460,474`); `fe_n_rows`/`fe_column_blocks` learn `LazyFrame` and row slices | polars-input fit: zero `to_pandas` calls |
| 3 | Subsample-fit mode (B.1 to B.4), Pass A/C reading through `scan_parquet` row groups | 10M x 80 fits; below the size threshold nothing changes; certify-on-C p-values calibrated on a synthetic null set |
| 4 | `CodeStore` with today's values, filled block by block, `LazyFrame` scanned per block | codes equal after explicit cast; peak lower |
| 5 | Pass C for fixed-edge columns on the full data; prefetch/double buffering for block reads | GPU timeline (`RSS_TICK_S`) shows a fed device |
| 6 | Narrow dtypes (per-column uint8/uint16, one NaN sentinel) behind a cast audit of every numpy-level joint-code site | selection equal, no overflow |
| 7 | Block-wise transform replay for huge frames (separate design) | transform of a larger-than-RAM frame |
| 8 | Scale runs 1M x 100, 5M x 80, parquet larger than RAM | completes inside the RAM guard; timeline saved |

## Risks that stay open

1. Codes cannot be computed without the target under default MDLP; mode B therefore does the target-dependent work on A only.
2. Certification validity rests on C being disjoint from every choice made on A and D; any later use of C to tune breaks it.
3. Uniform sampling fails for groups, time series and rare classes (hash by entity, class-aware weights, shifts).
4. Concurrent stages and GPU make bit-level equality gates flaky; the gates compare selection (names, order, scores within tolerance), with concurrency pinned.
5. `ColumnArena` flaws recorded for the day it is wired in: publish after write, owner by stage order (per-stage arenas, fixed-order merge), no lossy cast, positional index from `X`, local of `fit` never on `self`, `try/finally` cleanup.

## Constraints at every step

pandas input keeps today's results; float64 stays where a kernel contract or a measured selection change requires it; every new path has a CPU twin, a column-block budget, a test against the in-memory path and a log when it takes a slow path; the four cascades must give the same output whatever the thread timing.
