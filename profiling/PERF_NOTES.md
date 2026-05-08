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
| 6 | regression report `tight_layout` recompute per chart | `layout="constrained"` on plt.subplots; constrained_layout caches solver state | ~3s on c0089 (32 charts); cleaner code |
| 7 | **ensembling re-rendered chart per method (overwriting same file)** | `plot_file=""` + `show_perf_chart=False` in `_process_single_ensemble_method` flat_params | **c0025 82s -> 50s (39%)**; eliminates 24 redundant chart writes per 6-method ensemble |

### Numbers

- `c0002_7b21cbe5` (lgb regression, 30k rows): **64s -> 5s** (mostly from MLP-skip)
- `c0114_db5cb49a` (lgb+xgb multiclass, 100k rows, plotly[html,png] dispatcher): **618s -> 135s** (4.6×, from kaleido persistent + MLP-skip)
- `c0149_a7ff1d5a` (hgb+lgb multilabel, 50k rows): 55s -- profiled for additional hotspots; remaining cost is dominated by sklearn `MultiOutputClassifier.fit` (15.7s, model fit), joblib parallel polling (`time.sleep × 1481: 15.5s`, joblib internal), and matplotlib chart drawing (10.7s for 12 charts via legacy + multi-target dispatcher coexistence). None have non-controversial fixes.
- `c0025_98d57fbf` (cb+hgb+lgb binary, lof outlier detection, 30k rows): **82s -> 50s** (39%, from ensemble chart skip).
- `c0089_246583d5` (hgb+xgb regression with mrmr + 8 cat cols, 80k rows): 44s -> 41s (constrained_layout marginal).
- `c0097_5ea069f9` (hgb+linear regression with mrmr + 8 cat cols, polars_utf8, 50k rows): 20.5s. Already well-optimized -- remaining cost is matplotlib chart drawing (10s) + actual model fits.
- Reporting test suite: 191/191 still passing after all optimizations.

### Session conclusion

After 8+ iterations across diverse combo shapes (binary / multiclass /
multilabel / LTR / regression × small / mid / large), the dominant
remaining costs are:

1. **Real model fits** (CB, XGB, LGB, HGB) -- can't be optimized at
   the mlframe layer.
2. **matplotlib PNG encoding + draw** -- ~0.5-1s per chart; further
   reduction requires DPI lowering (visual quality tradeoff) or
   skipping per-class binary-calib charts when the multi-target
   dispatcher draws the multiclass grid (behavior change).
3. **joblib parallel polling sleeps** -- upstream `time.sleep(0.01)`
   in joblib's wait loop; not actionable from mlframe.
4. **numba JIT first-call cost** for non-prewarmed dtype combos --
   prewarming all combos costs more than it saves.

These are noted but not pursued. Further profile-and-optimize work
should target the suite-level path -- e.g. caching across (target,
model, weight_schema) iterations -- rather than per-call hotspots.

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
