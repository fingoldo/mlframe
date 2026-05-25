# F24: K-target ensemble fit serial loop (architectural deferral)

## Problem (from perf-hotspots-critique.md row 24)

The cross-target ensemble block in `_phase_composite_post.py:683-731` and `:892-905` iterates per-regression-target sequentially:
```
for _orig_tname, _spec_list in _tt_specs.items():
    # ... compute_oof_holdout_predictions(...)  # per-target refit on 1-frac, predict on frac (seconds per target)
    # ... linear_stack / nnls_stack solve (microseconds)
```
Per CLAUDE.md ladder, independent-target ensemble fits are joblib candidates: K targets x (NNLS + linear_stack solve) is embarrassingly parallel.

## Why deferred

Two concerns:

1. **Side-effect mutation of `metadata`, `models`, `_train_pred_cache`** inside the loop body. Parallelising naively triggers dict-write races on `metadata[...]` (shared dict mutated by each target). Refactoring to "per-target compute -> serial merge" is the only safe path and that is an architectural rewrite.

2. **`compute_oof_holdout_predictions` may itself be parallelised** (it calls `clone(model).fit` over (1-frac, frac) splits). Adding outer-level joblib over targets could double-book CPU when inner already uses `n_jobs=-1`. Need to coordinate the joblib level via `joblib.parallel_config(n_jobs=...)` BEFORE wrapping.

## Options

| Option | LOC | Risk | Speed | Notes |
|---|---|---|---|---|
| A. `joblib.Parallel(threading)` over targets with all metadata writes deferred to a post-loop merge step | ~200 LOC | High - 7+ metadata dict writes need refactoring | K-way (1.5-3x at K>=4) | Needs sklearn clone before parallel handoff; thread-safe access to inner model attributes during fit/predict |
| B. `concurrent.futures.ProcessPoolExecutor` with target-scoped results returned via dataclass | ~300 LOC | Higher - pickling trained models has known surface area | K-way | Tradeoff: avoids GIL but pays serialisation cost |
| C. Defer; document the serial bottleneck in the ensemble docstring; suggest user-supplied `joblib.parallel_backend` context if they want concurrency | 0 LOC | None | 0 | Most suites have K<=3 regression targets where the parallel win is small |

## Recommendation

**Option A**, but only after the inner OOF compute is verified single-threaded (no nested joblib). Measure first per `feedback_perf_measure_first`: needs a real prod fixture with K>=4 regression targets and a heavyweight model to see meaningful win.

## Risks

- `_train_pred_cache` is a process-wide dict written inside the loop; threading needs `threading.Lock` or local-then-merge pattern.
- The `dummy_floor_enabled` block at :611-686 reads `metadata["dummy_baselines"]` (cross-target shared). Per-target read of suite-shared metadata is safe; per-target WRITE of `metadata["composite_target_ensemble"]` is where races happen.
- Cloning models for parallel fit may break sklearn's NGBoost / CatBoost custom-eval paths the wrapper already handles (the same RuntimeError/TypeError fallbacks at `_phase_train_one_target_body.py:573`).
