# Performance optimization log

Running notes from the profile-and-optimize loop (random fuzz combos
under cProfile on mid-size frames, then iteratively patch the hottest
removable cost).

## 2026-05-08 session

### Wins committed

| # | hotspot | fix | impact |
|---|---|---|---|
| 1 | PyTorch / Lightning eager import in `get_training_configs` | gate MLP config block on `enabled_models` containing 'mlp' / 'recurrent' | ~14s saved per non-neural suite invocation (cold start) |
| 2 | N×collect() over cat columns in `get_trainset_features_stats_polars` | batch via `implode()` -> 1 collect | ~5-10ms × `len(cat_cols)` saved (1 call per suite) |
| 3 | Numba JIT cold-start for `compute_ece_and_brier_decomposition` / `fast_aucs_per_group_optimized` / `fast_log_loss` / `fast_ice_only` / `format_classification_report` | extend `prewarm_numba_cache` with the (bool, float64) suite-runtime combo | ~3-5s saved on first calibration-report call |
| 4 | `lgb_shim` `eval_set` 2-tuple unpack vs 3-tuple actuality | robust positional unpack | unblocks LGB e2e test |
| 5 | **kaleido oneshot Chromium spawn per plotly PNG export** | `start_sync_server` on first save + `kaleido.write_fig_sync` (persistent server) + `atexit` cleanup | **13s/call -> 0.13s/call (100×)**; on c0114 (32 PNG saves): 432s -> 12s |

### Numbers

- `c0002_7b21cbe5` (lgb regression, 30k rows): **64s -> 5s** (mostly from MLP-skip)
- `c0114_db5cb49a` (lgb+xgb multiclass, 100k rows, plotly[html,png] dispatcher): **618s -> 135s** (4.6×, from kaleido persistent + MLP-skip)
- Reporting test suite: 191/191 still passing after all optimizations.

### Hotspots that survived (deliberately not optimized)

- **`catboost _train` / `xgboost.fit` / `lgb.fit`** — actual model fit work; out of scope.
- **matplotlib `savefig` + `constrained_layout`** — ~70-100s combined on c0114 (multiclass produces K charts × N reports); could be cut by skipping per-class legacy charts when multi-target panel dispatcher active, but that's a behavior change requiring an opt-in. Deferred.
- **`score_ensemble` re-entry through `train_and_evaluate_model`** — by design (each ensemble metric is computed via the same report code path); refactoring is invasive.
- **`loky._count_physical_cores_win32` first-call subprocess (~1.4s)** — joblib upstream behavior on Windows; cached after first call.

### Rejected as non-actionable

- **Lower matplotlib DPI from 100 -> 80**: 18% smaller PNGs, ~270ms saved per call on c0044, but users may rely on the higher-resolution output. Per-call cost is small relative to other overhead.
- **Aggressive prewarm with all dtype combos**: prewarm itself ballooned to ~50s (worse than the runtime saving).

### Tools

- `profiling/profile_one_combo.py` -- single-combo cProfile run, optional `--save-stats` for snakeviz
- `profiling/profile_fuzz_chains.py` -- random N-combo run with per-combo `.prof` dump
- `profiling/aggregate_prof.py` -- cross-`.prof` hotspot aggregator with `--filter` for noise reduction

## Workflow

```
# 1. Generate fresh profile data on 3 random combos
python profiling/profile_fuzz_chains.py --combos 3 --rows-target 100000 \
    --save-dir D:/Temp/profruns/baseline

# 2. Inspect cross-combo aggregate, filtered to mlframe-internal
python profiling/aggregate_prof.py --dir D:/Temp/profruns/baseline \
    --top 30 --filter mlframe

# 3. Identify a hotspot, look at the source, design a fix.

# 4. Apply the fix.

# 5. Re-profile the SAME combo to verify the speedup:
python profiling/profile_one_combo.py --combo cXXXX --rows 100000 \
    --save-stats D:/Temp/profruns/after.prof

# 6. Diff against baseline:
python profiling/aggregate_prof.py D:/Temp/profruns/baseline/cXXXX.prof \
    D:/Temp/profruns/after.prof --top 20
```
