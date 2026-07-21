# training/core phase machinery A (composite-discovery phases, setup helpers, dummy baselines) -- mlframe audit

## Scope

All 36 files in the assigned cluster were read in full (no file was skipped or partially reviewed):

- src/mlframe/training/core/_misc_helpers.py (958 LOC)
- src/mlframe/training/core/_phase_composite_discovery.py (828 LOC)
- src/mlframe/training/core/_phase_helpers_fit_split.py (804 LOC)
- src/mlframe/training/core/_phase_composite_wrapping.py (742 LOC)
- src/mlframe/training/core/_phase_helpers_fit_pipeline.py (721 LOC)
- src/mlframe/training/core/_predict_main_from_models.py (689 LOC)
- src/mlframe/training/core/_main_train_suite_phases.py (610 LOC)
- src/mlframe/training/core/_phase_train_one_target.py (581 LOC)
- src/mlframe/training/core/_setup_helpers.py (521 LOC)
- src/mlframe/training/core/_achievable_ceiling.py (494 LOC)
- src/mlframe/training/core/_phase_train_one_target_weight_iteration.py (451 LOC)
- src/mlframe/training/core/_phase_dummy_baselines.py (373 LOC)
- src/mlframe/training/core/_phase_polars_fixes.py (373 LOC)
- src/mlframe/training/core/_setup_helpers_pre_pipelines.py (336 LOC)
- src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py (332 LOC)
- src/mlframe/training/core/_phase_composite_post_moe.py (271 LOC)
- src/mlframe/training/core/_phase_train_one_target_post.py (266 LOC)
- src/mlframe/training/core/_training_context.py (254 LOC)
- src/mlframe/training/core/_phase_train_one_target_ensembling.py (230 LOC)
- src/mlframe/training/core/_phase_train_one_target_mlp_helpers.py (227 LOC)
- src/mlframe/training/core/_phase_composite_post_summary.py (172 LOC)
- src/mlframe/training/core/_diversity_recommendations.py (160 LOC)
- src/mlframe/training/core/_phase_temporal_audit.py (159 LOC)
- src/mlframe/training/core/_phase_train_one_target_pre_screen.py (152 LOC)
- src/mlframe/training/core/_phase_drift_snapshot.py (148 LOC)
- src/mlframe/training/core/_ar1_failsafe_veto.py (107 LOC)
- src/mlframe/training/core/_phase_composite_post_lag_predict.py (92 LOC)
- src/mlframe/training/core/__init__.py (75 LOC)
- src/mlframe/training/core/_main_train_suite_defaults.py (63 LOC)
- src/mlframe/training/core/_phase_runners.py (38 LOC)
- src/mlframe/training/core/_predict_main.py (20 LOC)
- src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py (1147 LOC)
- src/mlframe/training/core/_phase_composite_post_xt_ensemble/_post_xt_ensemble_mtr.py (307 LOC)
- src/mlframe/training/core/_phase_composite_post_xt_ensemble/_phase_composite_post_xt_mtr_oof.py (168 LOC)
- src/mlframe/training/core/_phase_composite_post_xt_ensemble/_benchmarks/bench_mtr_oof_slice.py (216 LOC)
- src/mlframe/training/core/_phase_composite_post_xt_ensemble/_benchmarks/bench_mtr_oof_refit_count_irreducible.py (116 LOC)
- src/mlframe/training/core/_phase_composite_post_xt_ensemble/_benchmarks/__init__.py (0 LOC)

Total files reviewed: 36. Total LOC reviewed: 13201 (`wc -l` sum, verified).

Files imported by the above for context only (signature/docstring reading, per the audit's "excluded package" carve-out or "belongs to a sibling cluster" rule) but NOT analyzed for bugs: `._ar_skip`, `._phase_composite_discovery_helpers`, `.utils` (core/utils.py), `._phase_train_one_target_dataset_cache`, `._phase_train_one_target_helpers`, `._phase_train_one_target_body`, `._phase_train_one_target_schema`, `._ensemble_chooser`, `._ood_lag_router`, `._volatility_lag_router`, `._main_train_suite_target_distribution`, `._setup_helpers_pipeline_cache`, `._setup_helpers_outliers`, `._setup_helpers_metadata`, `._phase_config_setup`, `._phase_composite_post`, `._phase_finalize`, `._phase_recurrent`, `._phase_global_outlier_detection`, `.main`, `.predict`, `._predict_main_suite` -- these are siblings under `training/core/` not listed in the assigned 31+xt_ensemble file set, so they belong to a different cluster.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | ml-best-practice / sample-weight-propagation | src/mlframe/training/core/_phase_composite_post_xt_ensemble/_phase_composite_post_xt_mtr_oof.py:50-57 (and call site __init__.py:152-155) | Honest-OOF NNLS weighting for the MULTI_TARGET_REGRESSION cross-target ensemble silently trains every K-fold refit unweighted, even when the suite is a weighted-training run, while the parallel single-target cross-target ensemble path explicitly threads `ctx.sample_weights`. |
| F2 | P2 | silent-failure / config | src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py:68 | `x or DEFAULT` pattern: an explicit `MLFRAME_PANDAS_VIEW_CACHE_MAX_MB=0` (intended to disable the pandas-view cache) silently reverts to the 2048 MB default because `0.0 or _DEFAULT_ABS_BYTES` evaluates to the default. |
| F3 | P2 | correctness / logging | src/mlframe/training/core/_misc_helpers.py:34-38 | `_ensure_logging_visible`'s fast-path returns as soon as it finds the FIRST root handler whose formatter already has `%(asctime)s`, even when other handlers in `root.handlers` still lack a timestamp -- on a mixed handler set (e.g. a Jupyter-added handler appended after a prior call) those other handlers are silently left unfixed. |
| F4 | P2 | architecture | src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py (1147 LOC) | File exceeds the project's own ~900-1000 LOC monolith-split convention (`CLAUDE.md` "New code goes in focused submodules from the start"); the `_build_cross_target_ensemble_for_target` function alone is ~980 lines with deeply nested OOF / gate / calibration / routing logic that could be carved into siblings (e.g. `_oof_build.py`, `_gate_and_routing.py`, `_reporting.py`). |

### F1 -- MTR cross-target ensemble honest-OOF weighting silently drops sample weights

`compute_mtr_oof_nnls_weights` (`_phase_composite_post_xt_mtr_oof.py:50`) has the signature `(components, X_train, y_train, *, kfold=5, random_state=42)` -- there is no `sample_weight` parameter, and its inner K-fold refit loop calls `cl.fit(X_tr, y_tr)` with no weight argument at all. Compare this to the general (non-MTR) cross-target ensemble build in `_phase_composite_post_xt_ensemble/__init__.py` (lines ~426-434, ~601), which explicitly resolves `_sw_for_oof = np.asarray(_sw_raw)[filtered_train_idx]` from `ctx.sample_weights` and threads `sample_weight=_sw_for_oof` into `compute_oof_holdout_predictions`. The MTR branch (lines 124-171 of `__init__.py`) calls `compute_mtr_oof_nnls_weights(_mtr_components, filtered_train_df, _y_arr_mtr, kfold=..., random_state=...)` with no weight at all -- `ctx.sample_weights` is never read on this path. A caller running a MULTI_TARGET_REGRESSION suite with a non-uniform weight schema (recency weighting, class-imbalance weighting, etc.) gets per-column NNLS ensemble weights fit as if every row were equally important, silently diverging from the weighting contract the rest of the suite honors. Suggested fix: add a `sample_weight` kwarg to `compute_mtr_oof_nnls_weights`, slice it the same way `_sw_for_oof` is sliced for the single-target path, and pass it into each fold's `cl.fit(X_tr, y_tr, sample_weight=sw_tr)` when the underlying estimator accepts it (mirroring the existing `getattr(..., "supports_sample_weight", ...)`-style dispatch used elsewhere in the suite).

### F2 -- `MLFRAME_PANDAS_VIEW_CACHE_MAX_MB=0` silently reverts to the 2 GB default

In `resolve_pandas_view_cache_budget_bytes()`:
```python
if _legacy_mb is not None and not ctype and not size_raw:
    try:
        return max(0.0, float(_legacy_mb)) * (1024**2) or _DEFAULT_ABS_BYTES
    except ValueError:
        return _DEFAULT_ABS_BYTES
```
An operator setting the deprecated `MLFRAME_PANDAS_VIEW_CACHE_MAX_MB=0` to disable the polars-to-pandas view cache (a legitimate, explicit intent -- 0 is a valid budget meaning "never reuse") gets `max(0.0, 0.0) * (1024**2)` = `0.0`, and `0.0 or _DEFAULT_ABS_BYTES` evaluates to the 2048 MB default because `0.0` is falsy in Python. The explicit "disable" request is silently overridden with a 2 GB budget instead. `tests/training/test_pandas_view_cache_budget.py` has no test for the `MAX_MB=0` case, so this regression class would not be caught. Suggested fix: use an explicit `is not None`/sentinel check (`budget = max(0.0, float(_legacy_mb)) * (1024**2); return budget if _legacy_mb.strip() else _DEFAULT_ABS_BYTES` or similar) instead of the truthiness `or`.

### F3 -- Logging visibility fast-path can skip fixing a newly-added handler

```python
if root.handlers and (root.level != logging.NOTSET and root.level <= level):
    for h in root.handlers:
        existing = getattr(h.formatter, "_fmt", None) if h.formatter else None
        if existing and "%(asctime)" in existing:
            return
```
The loop returns on the FIRST handler it finds with an asctime-bearing formatter, not after verifying ALL handlers have one. If a second call to `_ensure_logging_visible` happens after some other code path (e.g. a notebook kernel, or a library) appended a new `StreamHandler` without a timestamped formatter, and the ORIGINAL (already-fixed) handler is still present and iterated first, the function exits early and the newly-appended handler never gets its formatter upgraded -- contradicting the docstring's own contract ("Replaces non-timestamped formatters in place"). Concrete scenario: `train_mlframe_models_suite` call #1 installs/upgrades a handler with `%(asctime)s`; between calls, Jupyter or another package pushes its own bare `StreamHandler` onto `root.handlers`; suite call #2 sees the old fixed handler first in the fast-path scan and returns without touching the new bare handler, so some log lines silently lose timestamps. Low real-world severity (cosmetic, log-formatting only) but a genuine logic bug relative to the documented contract. Suggested fix: require ALL handlers to carry `%(asctime)s` before taking the fast-path return (`all(...)` instead of returning inside the loop on the first match).

### F4 -- `_phase_composite_post_xt_ensemble/__init__.py` exceeds the file-size convention

At 1147 LOC (with a single function, `_build_cross_target_ensemble_for_target`, spanning roughly lines 68-1051 -- ~980 lines), this file is well over the project's own documented ~900-1000 LOC monolith-split threshold. It already imports helpers from two carved-out siblings (`_post_xt_ensemble_mtr.py`, `_phase_composite_post_xt_mtr_oof.py`) but the core function itself was never split further even though it has clearly separable phases (OOF matrix construction, dummy-floor gating, residual dedup, stacking-strategy dispatch, AR1-failsafe + OOD/volatility routing, output calibration, reporting). No functional bug found in it, but it is a real maintainability/architecture risk: the file is the single largest in this cluster and any future edit to one phase risks an unrelated merge conflict or accidental interaction with another phase's local variables (all sharing one function scope with dozens of `_`-prefixed locals).

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | test-coverage | src/mlframe/training/core/_phase_train_one_target_weight_iteration.py | `_run_one_weight_iteration` (the innermost per-(model, weight-schema) iteration, 451 LOC file) has no dedicated unit test file (only indirectly exercised via full-suite integration/e2e tests); a focused unit test covering the `break_model_loop` (identity-equivalent pre_pipeline dedup) and `skip` (continue_on_model_failure) return paths would pin the two non-obvious control-flow branches without needing a full suite run. |
| PR2 | test-coverage | src/mlframe/training/core/_main_train_suite_defaults.py | `_build_default_extractor` / `_infer_target_is_classification` are only exercised indirectly (via `tests/training/test_suite_api_ergonomics.py`); no direct unit test pins the int/bool/float/object dtype-inference boundary cases (e.g. a low-cardinality float column, or an all-NaN target column, which would currently raise deep inside `_is_classification_target` rather than with a clear message). |
| PR3 | perf | src/mlframe/training/core/_setup_helpers.py:160-168 (`tune_decision_threshold`) | The threshold sweep recomputes `balanced_accuracy_binary`/`f1_score` from scratch for each of up to 200 evenly-spaced candidates via a Python loop over the full `(y, p)` arrays. On a multi-million-row val/OOF split (this codebase's typical scale) this is O(200 x N) Python-level work; a vectorized formulation (e.g. sort `p` once, compute cumulative TP/FP/TN/FN via `np.cumsum` over the sorted order, then evaluate all 200 candidate metrics from the cumulative counts) would turn 200 O(N) passes into one O(N log N) sort + O(200) vector ops. Not measured in this audit (per CLAUDE.md's "measure before optimizing" rule) -- flagged as a lead, not a verified win. |
| PR4 | ml-best-practice | src/mlframe/training/core/_diversity_recommendations.py:41-63 | `_member_individual_score` only reads `member.metrics["val"]`; on a suite configured with `compute_valset_metrics=False` (test-only reporting) every member's individual score comes back `None` and `compute_diversity_recommendations` silently returns `None` for the whole diagnostic (documented as intentional "OOF-preferred, no WARN fallback" behavior in the module docstring, but the val-vs-OOF-vs-test priority is not configurable) -- consider falling back to OOF-derived individual scores (already required elsewhere in the same function via `oof_preds`/`oof_target`) instead of `metrics["val"]`, so the diagnostic degrades gracefully on val-metrics-disabled suites instead of going fully dark. |

## Coverage notes

- All 36 assigned files were read to completion; none were too large to review in full (largest, `_phase_composite_post_xt_ensemble/__init__.py` at 1147 LOC, was read in two passes covering every line 1-1147).
- Sibling `training/core/` modules referenced by imports (e.g. `._phase_train_one_target_body`, `._phase_composite_post`, `.main`, `.predict`) were deliberately NOT analyzed for bugs -- they are not in the assigned file list and evidently belong to a different parallel-audit cluster (per the task's cluster-boundary instructions: "do NOT recurse into subdirectories... unless listed").
- No git history / blame commands were needed; all findings were confirmed by direct reading of the current file contents plus one live numpy/psutil check (`np.unique(..., axis=0, return_inverse=True)` shape behavior on the installed numpy 2.3.5, and reading `tests/training/test_pandas_view_cache_budget.py` to confirm the `MAX_MB=0` case has no regression test).
- Did not attempt to execute the suite end-to-end (out of scope: read-only audit, no pytest/benchmark runs permitted).
