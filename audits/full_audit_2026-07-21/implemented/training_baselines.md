# training/baselines -- mlframe audit

## Scope

All 24 `.py` files under `src/mlframe/training/baselines/**` were read in full:

- `__init__.py` (47 LOC)
- `dummy.py` (774 LOC)
- `diagnostics.py` (516 LOC)
- `_dummy_baseline_compute.py` (670 LOC)
- `_dummy_baseline_regression.py` (239 LOC)
- `_dummy_baseline_classification.py` (179 LOC)
- `_dummy_baseline_quantile.py` (109 LOC)
- `_dummy_numba_kernels.py` (271 LOC)
- `_dummy_compute_helpers.py` (315 LOC)
- `_dummy_bootstrap.py` (575 LOC)
- `_dummy_timeseries.py` (251 LOC)
- `_dummy_metrics_pick_plot.py` (539 LOC)
- `_dummy_report_type.py` (241 LOC)
- `_dummy_summary_format.py` (325 LOC)
- `_baseline_diagnostics_quick_model.py` (142 LOC)
- `_baseline_diagnostics_ablation.py` (119 LOC)
- `_baseline_diagnostics_init_score.py` (161 LOC)
- `_baseline_diagnostics_recommend.py` (62 LOC)
- `_smoke_dummy_baselines_e2e.py` (90 LOC)
- `_profile_dummy_baselines.py` (216 LOC)
- `_profile_dummy_baselines_recent.py` (276 LOC)
- `_benchmarks/bench_multiclass_auc_averaging.py` (119 LOC)
- `_benchmarks/bench_bootstrap_ci_n_resamples.py` (180 LOC)
- `_benchmarks/bench_ablation_n_estimators_provisioning.py` (277 LOC)

Total: **24 files, 6693 LOC**, all reviewed in depth (no file was skipped or partially skipped).

Two of the P0/P1 findings below were empirically reproduced in-process (read-only: calling the public `compute_dummy_baselines` entry point with a synthetic fixture, no repo/filesystem mutation) to avoid reporting a hand-waved hypothesis; the reproduction transcripts are quoted in the finding paragraphs.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness/crash | `_dummy_baseline_classification.py:57-58` | A single NaN in `train_y` crashes `compute_dummy_baselines` for binary/multiclass classification targets with an uncaught `ValueError` from `np.bincount`. |
| F2 | P1 | correctness/nan-handling | `_dummy_baseline_regression.py:63-75` | A single NaN in `train_y` silently wipes out every constant-prediction regression baseline (mean/median/quantile_p25/p75) via non-NaN-aware `np.mean`/`np.median`/`np.quantile`, collapsing the whole dummy-baselines verdict to `strongest=None` with a misleading "both splits degenerate" message even though only train has one bad value. |
| F3 | P1 | correctness/key-mismatch | `_dummy_baseline_regression.py:170,174,201,207,209` vs `_dummy_timeseries.py:234-235` | `_resolve_ts_periods` stores ACF-detected periods under the key `"acf_periods"`, but the regression dispatcher reads `ts_diag.get("acf_peaks")` (a key that is never set) -- the `rolling_mean_w7`/`rolling_mean_w30 (ts)` baselines are therefore never emitted and the `"(ts, ACF-detected)"` label never applied, for every TS regression target. |
| F4 | P2 | architecture/dead-code | `_dummy_baseline_compute.py:378-663` | A complete second copy of `compute_dummy_baselines` lives in this "facade" module (contrary to its own docstring, which claims the split moved the function out and left only a re-export); it has already drifted from the canonical `dummy.py` version (missing the `config.overlay_plot` feature entirely) and is importable/callable but has zero production call sites. |
| F5 | P2 | correctness/edge-case | `_dummy_summary_format.py:292,298,303,304` | `raw_val = raw_metric.get("val_RMSE") or raw_metric.get("RMSE")` (and the `comp_val`/`ct_val`/`lift`-gate siblings) use `or`/`and` as a falsy-default fallback; an RMSE of exactly `0.0` (a perfectly-fit or degenerate constant target) is falsy and incorrectly triggers the fallback key lookup or drops the lift computation entirely. |
| F6 | P2 | efficiency/dead-code | `_dummy_metrics_pick_plot.py:78-91` | `pinball_per_a` is built (one `.append()` per alpha, per split, per baseline) inside `_compute_metrics_table`'s quantile branch but never read; `non_boundary_vals` is independently recomputed from the `row` dict two lines later. Wasted allocation/dead local. |
| F7 | P2 | correctness/edge-case | `_baseline_diagnostics_recommend.py:30` | `max(e.delta_pct for e in ablation if math.isfinite(e.delta_pct))` raises `ValueError` (empty-sequence `max()`) when `ablation` is non-empty but every entry's `delta_pct` is non-finite; the exception is caught by `BaselineDiagnostics.fit_and_report`'s outer handler, but the entire diagnostic report degrades to a generic `internal_error` skip instead of a graceful `"unlikely_to_help"` classification. |
| F8 | P2 | correctness/edge-case | `_dummy_compute_helpers.py:284-295` + `_dummy_baseline_classification.py:57` | `_coerce_y`'s object-dtype-only guard (`if arr.dtype == object: arr.astype(np.int64)`) does not cover non-object float classification targets; `_compute_classification_baselines` then does an unguarded `train_y.astype(np.int64)` on whatever arrives, silently truncating fractional/rounding-noise labels (e.g. `2.9 -> 2`) instead of raising or rounding. |
| F9 | P2 | reproducibility | `_dummy_timeseries.py:159` | `_detect_acf_periods`'s stratified-window sampling (fires when `n_train > 50_000`) uses a hardcoded `np.random.default_rng(42)` instead of threading `config.random_state` through; two suite runs configured with different seeds always sample the identical ACF windows. |
| F10 | P2 | reproducibility | `_dummy_metrics_pick_plot.py:382` | `plot_best_dummy_baseline_overlay`'s row-subsampling for the overlay chart uses a hardcoded `np.random.default_rng(0)` instead of `config.random_state`; low practical impact (display-only), but the same unparameterized-seed pattern as F9. |

**F1** (`_dummy_baseline_classification.py:57-58`): Reproduced directly against the public API:

```
train_y = rng.integers(0,2,n).astype(float); train_y[5] = np.nan
compute_dummy_baselines(target_type='binary_classification', ..., train_y=train_y, ...)
# -> CRASHED: ValueError 'list' argument must have no negative elements
```
`np.nan` cast to `int64` becomes an implementation-defined large-magnitude value (observed `-9223372036854775808` on this box), and `np.bincount` on a negative element raises. Nothing between `_coerce_y` (`_dummy_compute_helpers.py:260-295`, which only special-cases `object`-dtype arrays) and `_compute_classification_baselines` filters or asserts finiteness of `train_y`; `compute_dummy_baselines` does not wrap the classification dispatch call in a try/except the way several other steps in the same function are defended (paired-bootstrap, bootstrap-CI, overlay plot all have `try/except Exception` around them -- the dispatch call itself does not). Any classification target with even one missing/NaN training label -- a realistic real-world scenario for partially-labeled data -- crashes the whole dummy-baselines phase for that target. Suggested fix: filter (or explicitly reject with a clear message) non-finite rows from `train_y`/`train_X` once, at the top of `compute_dummy_baselines`, mirroring the `n_val_finite`/`n_test_finite` gate that already exists for val/test.

**F2** (`_dummy_baseline_regression.py:63-75`): Reproduced directly:
```
train_y = rng.normal(size=200); train_y[5] = np.nan   # 1 NaN out of 200
compute_dummy_baselines(target_type='regression', ..., train_y=train_y, ...)
# -> strongest=None; mean/median/quantile_p25/quantile_p75 all val_RMSE=NaN, failed=True
```
`per_group_mean` (via pandas `groupby(...).mean()`, which skips NaN by default) would have survived the same input, so the failure is specific to the constant-baseline code path's use of plain `np.mean`/`np.median`/`np.quantile` instead of their `nan*` counterparts. The resulting report's `strongest=None` line reads "(both splits degenerate; review table manually)" (`_dummy_report_type.py:145`), which is actively misleading here since val and test are both fully finite -- the actual cause is one bad value in train. Suggested fix: use `np.nanmean`/`np.nanmedian`/`np.nanquantile` for the constant baselines (or filter train_y once), and differentiate the "degenerate" message by which split actually failed.

**F3**: with the key mismatch, `ts_diag.get("acf_peaks")` always returns `None`, so `(ts_diag.get("acf_peaks") or [])` is always `[]` regardless of what `_detect_acf_periods` actually found. This means: (a) `rolling_mean_w7 (ts)` / `rolling_mean_w30 (ts)` baselines can never appear in the table for any target, even when ACF genuinely detects a period >= 7 or >= 30, and (b) `seasonal_naive_pP` rows never get the `", ACF-detected"` suffix. No existing test asserts these baselines actually fire (grepped `tests/training/baselines/` for `rolling_mean_w`/`acf_peaks`/`acf_periods` -- only the unrelated `test_acf_returns_empty_when_statsmodels_missing` touches this area). Suggested fix: rename one side to match the other (`"acf_periods"` is also the sibling module's own internal name, so fixing the three read sites in `_dummy_baseline_regression.py` is the smaller change) and add a regression test asserting the baseline appears on an ACF-detectable synthetic series.

**F4**: `grep`-confirmed zero production import sites for `_dummy_baseline_compute.compute_dummy_baselines` (only the split-facade smoke test `tests/training/baselines/test_dummy_baseline_compute_split.py` touches it, and only via `assert callable(fn)` -- it never calls it, so the drift was never caught). The `overlay_plot`/`plot_best_dummy_baseline_overlay` feature added to the canonical `dummy.py:581-591` copy was never mirrored here. This is exactly the "logic duplicated near-identically elsewhere" pattern the repo's own conventions warn against; a future maintainer editing "the" `compute_dummy_baselines` via this facade (its docstring reads as though it's the canonical location) would silently ship a change that no caller ever executes. Suggested fix: delete the duplicate body and replace it with `from .dummy import compute_dummy_baselines` (the way every other symbol in the file is genuinely re-exported), or move the lazy-import indirection the other way so `dummy.py` is the sole owner.

**F5**: e.g. `raw_val = raw_metric.get("val_RMSE") or raw_metric.get("RMSE")` -- if a raw target's best model achieves exactly `RMSE=0.0` (plausible on a synthetic/degenerate-constant target, and this exact function is exercised by `tests/training/baselines/` fixtures that use small synthetic regressions), the `or` falls through to the `"RMSE"` key, which may be absent or stale, silently corrupting the `raw_best` column of the unified verdict table. Same pattern for `comp_val`/`ct_val`, and the `and`-gated `lift` computation additionally drops the lift entirely (`lift=None`, rendered as `"-"`) whenever `raw_val == 0.0`. Suggested fix: `x if x is not None else fallback` instead of `x or fallback`.

**F6**: not a behavioral bug (the correct value is still produced via the independent `row[...]` list comprehension a few lines later), just wasted work and a confusing dead variable that a future reader may assume is load-bearing. Suggested fix: delete `pinball_per_a` and its `.append()` call, or replace the redundant `non_boundary_vals` recomputation with a direct use of it.

**F7**: triggers only when the raw-fit metric is finite (checked earlier in `fit_and_report`) but every one-feature-dropped refit produces a non-finite metric -- an edge case (e.g. a LightGBM refit that degenerates to constant/NaN predictions after dropping a specific feature) but plausible on tiny/degenerate samples. Because the whole `try/except (ValueError, ...)` in `diagnostics.py:402-416` is broad, the caller just sees `skip_reason="internal_error: max() iterable argument is empty"` instead of the more informative `"unlikely_to_help"` verdict the situation actually warrants. Suggested fix: `max(..., default=float("-inf"))` and treat that sentinel as "no dominant feature" in `_build_recommendation`.

**F8**: this is the checklist's flagged "implicit truncating `.astype(int)` on a continuous target" pattern, but scoped narrowly here: it only bites when `train_y` for a `binary_classification`/`multiclass_classification` target arrives as a non-`object` float array carrying genuine fractional values (e.g. upstream label-encoding bug, or a `target_type` misconfiguration routing a continuous score through the classification path). Under mlframe's normal FTE pipeline this is unlikely but not impossible, and unlike F1/F2 it does not raise -- it silently produces wrong class assignments. Suggested fix: assert `np.allclose(arr, np.round(arr))` (or equivalent) before the cast and raise/log loudly on violation, the same way the `object`-dtype branch already does for a failed cast.

**F9/F10**: both are diagnostic/plotting-only sampling, not decision-affecting metrics, so blast radius is low, but they are exactly the class of bug the repo's own review history calls out repeatedly (hardcoded seed instead of the caller-supplied `random_state`/`config.random_state` reaching the call). Suggested fix: thread `config.random_state` (or a per-target derivative via the existing `_per_target_seed` helper) into both call sites.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| P1 | test-coverage | `_dummy_baseline_regression.py`, `_dummy_baseline_classification.py` | No test in `tests/training/baselines/` exercises a `train_y` containing NaN/missing values (grepped `nan`/`NaN` case-insensitively -- zero hits); add regression tests pinning F1/F2's fixed behavior once addressed. |
| P2 | test-coverage | `_dummy_baseline_regression.py:193-212` | No test asserts `rolling_mean_w{7,30} (ts)` baselines actually appear, or that the `"(ts, ACF-detected)"` label is applied, on an ACF-detectable synthetic series -- would have caught F3 directly. |
| P3 | architecture | `_dummy_baseline_compute.py` vs `dummy.py` | Either delete the dead duplicate `compute_dummy_baselines` (F4) or add a behavioral-equivalence test (not just `callable()`) between the two, so a future edit to one is forced to keep the other in sync (or the duplicate is removed outright, which is the cleaner fix). |
| P4 | refactor | `_dummy_bootstrap.py:92` vs `_dummy_metrics_pick_plot.py:300-307` | `_paired_bootstrap_vs_runner_up`'s runner-up pick (`series_excl_strongest.idxmin()/idxmax()`) doesn't use the same explicit alphabetical tie-break `_pick_strongest` applies for the strongest pick; today it happens to be consistent because `baseline_names` is sorted before the table is built, but that's an implicit invariant across two files rather than a shared helper -- extracting one `_tiebreak_pick(series, minimize)` would remove the duplication and the latent fragility. |
| P5 | docs | `_baseline_diagnostics_init_score.py:116-138` | The binary `init_score` probability-scale heuristic (`mn>=0 and mx<=1` implies "already a probability") can misfire on any `[0,1]`-normalized non-probability feature (e.g. min-max-scaled numeric); a docstring caveat (already partially present) plus an explicit override knob would make the heuristic's failure mode discoverable rather than silent. |
| P6 | robustness | `dummy.py:421-455` | `n_classes = max(2, len(unique_classes))` has no upper bound; a misconfigured `target_type` on a genuinely continuous column would silently allocate `(N, n_classes)` probability matrices per baseline for however many distinct floating-point values exist in the column. A cheap `n_classes` ceiling + WARN (e.g. "n_classes=48213 looks like a misrouted continuous target") would fail fast with a clear message instead of a slow/large-memory near-hang. |
| P7 | comment | `_baseline_diagnostics_ablation.py:69-90` | `_one_drop`'s per-drop refits all reuse the exact same `config.random_state` (both for the train/val split and the LightGBM's own seeding) by design, which is what makes the ablation deltas comparable apples-to-apples; this is already correct but relies on the reader not "fixing" it into per-drop seeds. Worth one inline comment stating the invariant explicitly, since the analogous code in `_fit_init_score_baseline` does the same thing without comment either. |

## Coverage notes

- Both P0/P1 NaN-handling findings (F1, F2) were empirically reproduced by calling the public `compute_dummy_baselines` entry point directly (read-only: no files written, no repo state changed) rather than inferred from static reading alone.
- I could not verify from within this cluster's scope whether `src/mlframe/training/core/_phase_dummy_baselines.py` (out of scope for this audit -- lives under `training/core/`, not `training/baselines/`) wraps its `compute_dummy_baselines(...)` call (`_phase_dummy_baselines.py:151-152`) in a try/except; a bare grep shows no such guard immediately around the call, only a `with phase(...)` context manager whose exception-handling behavior I did not trace. This matters for F1's real-world blast radius (does one target's crash abort the whole suite, or just that target?) but tracing it would mean auditing code outside my assigned cluster.
- `mlframe.metrics_registry` (`metric_name_higher_is_better`) is imported and relied upon throughout `_dummy_metrics_pick_plot.py` / `_dummy_summary_format.py` / `_dummy_bootstrap.py` for direction-aware metric comparisons; that registry module itself lives outside `training/baselines/` and was not audited here -- if it mis-registers a metric's direction, several of the call sites that trust it (`_pick_strongest`, `format_suite_end_summary`, `_paired_bootstrap_vs_runner_up`) would silently invert a verdict. Out of scope to verify.
- `mlframe.metrics.core` (`fast_roc_auc`, `fast_root_mean_squared_error`, `fast_mean_absolute_error`, `prewarm_numba_cache`) and `mlframe.metrics.ranking.compute_ranking_summary` are load-bearing dependencies of this cluster's metric computation but live outside `training/baselines/`; their internal correctness was not audited, only their call-site usage here.
- `feature_selection/filters/**` and `feature_selection/shap_proxied_fs/**` were correctly excluded per instructions and not read.
