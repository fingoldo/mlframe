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
| 7 | ~~ensembling re-rendered chart per method (overwriting same file)~~ **REVERTED** | `plot_file=""` + `show_perf_chart=False` in `_process_single_ensemble_method` flat_params | **REGRESSION** -- the premise was wrong; charts had unique filenames via `slugify(model_name_prefix)` (`Ens{METHOD}-N{n}_*_perfplot.png`). Per user feedback, reverted on 2026-05-08; sensor test in `tests/training/test_ensembling_chart_artifacts.py` now guards against re-applying this. |

### Numbers

- `c0002_7b21cbe5` (lgb regression, 30k rows): **64s -> 5s** (mostly from MLP-skip)
- `c0114_db5cb49a` (lgb+xgb multiclass, 100k rows, plotly[html,png] dispatcher): **618s -> 135s** (4.6×, from kaleido persistent + MLP-skip)
- `c0149_a7ff1d5a` (hgb+lgb multilabel, 50k rows): 55s -- profiled for additional hotspots; remaining cost is dominated by sklearn `MultiOutputClassifier.fit` (15.7s, model fit), joblib parallel polling (`time.sleep × 1481: 15.5s`, joblib internal), and matplotlib chart drawing (10.7s for 12 charts via legacy + multi-target dispatcher coexistence). None have non-controversial fixes.
- ~~`c0025_98d57fbf` (cb+hgb+lgb binary, lof outlier detection, 30k rows): 82s -> 50s~~ -- **the win was from a regression** (deleting per-method ensemble charts that have unique filenames). Reverted on 2026-05-08.
- `c0089_246583d5` (hgb+xgb regression with mrmr + 8 cat cols, 80k rows): 44s -> 41s (constrained_layout marginal).
- `c0097_5ea069f9` (hgb+linear regression with mrmr + 8 cat cols, polars_utf8, 50k rows): 20.5s. Already well-optimized -- remaining cost is matplotlib chart drawing (10s) + actual model fits.
- Reporting test suite: 191/191 still passing after all optimizations.

## 2026-05-08 session 2

After the user's pushback on win #7 (which was reverted), restarted
the profile-and-optimize loop with a strict rule: **before any
performance patch, verify the premise empirically -- bench before/
after, list on-disk artifacts, etc. Reading the code is not enough.**

### Wins shipped (3)

| # | hotspot | fix | impact (verified) |
|---|---|---|---|
| 8 | loky `_count_physical_cores_win32` (wmic subprocess) blocks suite for ~1.65s on first call | kick `cpu_count()` in a daemon thread BEFORE numba JIT inside `prewarm_numba_cache`; subprocess overlaps with this-process JIT compile | **~1.0s saved per fresh suite run** (3-run bench: 4.16s -> 3.03s). Verified with controlled "neuter the kick" test. |
| 9 | profile drivers fired matplotlib's Qt backend probe (~1.45s of `activateWindow` calls on c0088) | `matplotlib.use("Agg", force=False)` at top of `profile_one_combo.py` and `profile_fuzz_chains.py` | dev-quality only (production suite unchanged) -- removes Qt noise from profile traces |
| **10** | **CRITICAL DEADLOCK** -- win #5 (persistent kaleido server) regression: a single figure that triggers a JS error inside kaleido (`KaleidoError: Error 525...`) cancels the asyncio task chain and every subsequent `write_fig_sync` hangs forever in `await asyncio.gather`. Reproduced empirically: c0031 hung 2+ hours on `multiclass + recency + ensembling` combo. | `PlotlyRenderer.save`: catch ALL exceptions from persistent-path write (not just ImportError/ValueError); call new `_restart_kaleido_server()` helper to clear the broken async chain; retry via plotly's oneshot `write_image` (~13s but bug-isolated). | **c0031 unstuck: 2+h hang -> 84s completion**. .prof file actually produced (was 0 bytes before). Two regression sensor tests in `tests/reporting/test_kaleido_recovery.py` monkey-patch kaleido to raise and assert no hang + file written. |

### Hotspots considered + skipped (with empirical justification)

- **`show_inline_population_labels`** (10 text overlays per calibration chart): bench showed 277ms with vs 253ms without per chart -- only 24ms saved, doesn't justify the readability hit.
- **`show_perf_chart=False` Agg fast-path**: bench showed 290ms in both modes -- no actual win because savefig itself dominates, not pyplot init.
- **matplotlib `tight_layout` -> `constrained_layout`**: previous fix #6 already shipped; constrained_layout's solver still costs ~17s on c0088 but reverting to no-auto-layout would be a quality regression.

### Combos profiled

- `c0103_d00d3792` (hgb+mlp multilabel ensembling, 30k rows): 774s -- 348s in **cold-disk** lightning import (one-off, not recurring; subsequent calls in same process pay sys.modules cache ~14s).
- `c0088_110cbfb8` (hgb+lgb+xgb binary mrmr ensembles, 80k rows): 55.6s, 38s in calibration-report charts. Drilled into `show_calibration_plot`'s callees -- 290ms intrinsic per chart, no further easy cuts.
- `c0074_618df307`: hit a real bug (rankers don't handle text features). Filed as data point, not optimised.
- Tiny suite (1k rows, lgb-only, 5 iter): 4.3s -> 3.0s with prewarm + new parallel kick.

### Lesson learned (2026-05-08)

Win #7 was reverted after user pushback. The "optimization" hard-coded
``plot_file=""`` inside the ensembling re-entry path on the false
premise that all ensemble methods overwrote the same file. They DON'T:
``trainer._setup_model_info_and_paths`` already prefixes plot_file
with ``slugify(model_name_prefix)``, and the ensembling loop sets
``model_name_prefix=f"Ens{METHOD}-N{n}"`` per method, so each writes
to a unique path (``EnsARITHM-N6_val_perfplot.png``,
``EnsGEO-N6_val_perfplot.png``, ...). The "win" was really
_silently dropping 24 user-visible artifacts_ -- per-method
calibration charts that operators legitimately compare. The premise
was never empirically verified before the patch landed.

**Process rule going forward**: before applying a perf "optimization"
that disables a feature, list the existing on-disk artifacts and
verify they actually collide. Reading the path-construction code is
not a substitute -- empirical `os.walk` is.

### Validation against pre-fix aggregate

A 2-combo `profile_fuzz_chains` run launched BEFORE the kaleido fix
(c0023 + c0114, 100k rows each) finally completed after 4+ hours of
buffered runtime. Its aggregate top-50 confirms the kaleido oneshot
bug as the #1 hotspot in pre-fix mlframe:

```
cumtime      function
1093.186     train_mlframe_models_suite
 933.351     render_multi_target_panels (32 calls)
 918.023     plotly.basedatatypes.write_image (32 calls)
 917.953     _thread.lock acquire (302 calls)  <- kaleido sync wait
 916.587     kaleido._sync_server.oneshot_async_run (32 calls)
 850.876     ensembling.score_ensemble
 118.502     fast_calibration_report (72 calls)
 ...
```

That's 916.6s (84% of total) spent in 32 kaleido oneshot Chromium
spawns. Win #5 (persistent server) replaces those 32 × 28.6s/call
spawns with 1 × 8.1s warmup + 31 × 0.13s reuses = ~12s total. Net
saving: ~905s per 2-combo run with multi-target panel emission.

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
