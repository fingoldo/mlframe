# A5 P2 #9 - Polars LazyFrame.cache() integration

## Problem

Audit observation: zero uses of `polars.LazyFrame.cache()` in `src/mlframe/training/core` despite repeated re-traversal of the same `train_df_polars` across:
- `apply_polars_categorical_fixes` (`_main_train_suite.py:690`)
- `_phase_global_outlier_detection`
- `trainset_features_stats`
- `prepared_frames_cache` builders (per-target reuse)

Each currently fully evaluates the same eager frame.

## Why this is not a drop-in fix

Polars `LazyFrame.cache()` only helps when:

1. The plan is genuinely re-executed within a SINGLE lazy graph (multi-branch). `apply_polars_categorical_fixes`, `_phase_global_outlier_detection`, etc. each receive an eager `pl.DataFrame` and call `.lazy()` internally; they do not share a parent lazy node so `.cache()` would not memoize across calls.
2. The cached node's output fits in RAM. mlframe frames can be 100+ GB per CLAUDE.md - blanket promotion would OOM the host.

A correct integration requires:
- restructuring the suite entry to produce ONE `train_lazy = train_df.lazy().cache()` early and threading it through the four call sites (each accepts an eager frame today)
- byte-size gate via `train_df.estimated_size()` < ~2 GB before caching
- a microbench across the synthetic suite to prove the speedup justifies the refactor

## Proposed options

- **Option A (full refactor, deferred to wave 7)**: introduce a `SuiteFrameContext` carrying `train_lazy / val_lazy / test_lazy` produced once at suite entry, gated on `train_df.estimated_size() <= 2 GB`. Plumb through `apply_polars_categorical_fixes` + the three callers. ~6-10 files touched. Bench against a single multi-target run; ship only if wall-time win > 5%.
- **Option B (skip)**: leave eager evaluation as-is; polars's eager engine already memoizes via Arrow buffers when the user holds the reference. The audit's "zero memory cost over the existing eager path" claim is correct only AFTER the SuiteFrameContext shape lands.

## Recommendation

Defer to wave 7. Pre-implementation requirement per CLAUDE.md "Bench first, dispatch second": measure wall-time of the four call sites individually on a 50k / 1M / 10M row synthetic before designing the threading shape. The audit's claim of "zero memory cost" is plausible but not yet measured.

## Status

ARCH-DEFER (wave 7).
