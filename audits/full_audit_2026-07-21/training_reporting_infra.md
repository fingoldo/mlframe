# reporting infrastructure, non-charts (training/reporting + top-level reporting/ + reporting/renderers) -- mlframe audit

## Scope

All files were read in full this session.

- `src/mlframe/training/reporting/__init__.py` (42 LOC)
- `src/mlframe/training/reporting/_reporting.py` (644 LOC)
- `src/mlframe/training/reporting/_reporting_diagnostics.py` (465 LOC)
- `src/mlframe/training/reporting/_reporting_probabilistic.py` (903 LOC)
- `src/mlframe/training/reporting/_reporting_probabilistic_calib.py` (212 LOC)
- `src/mlframe/training/reporting/_reporting_regression/__init__.py` (744 LOC)
- `src/mlframe/training/reporting/_reporting_regression/_mtr.py` (123 LOC)
- `src/mlframe/training/reporting/_reporting_regression/_sensors.py` (176 LOC)
- `src/mlframe/reporting/renderers/__init__.py` (25 LOC)
- `src/mlframe/reporting/renderers/_kaleido.py` (320 LOC)
- `src/mlframe/reporting/renderers/_plotly_color.py` (73 LOC)
- `src/mlframe/reporting/renderers/_plotly_interactivity.py` (122 LOC)
- `src/mlframe/reporting/renderers/_shared_helpers.py` (33 LOC)
- `src/mlframe/reporting/renderers/_trend.py` (72 LOC)
- `src/mlframe/reporting/renderers/base.py` (58 LOC)
- `src/mlframe/reporting/renderers/matplotlib.py` (776 LOC)
- `src/mlframe/reporting/renderers/plotly.py` (950 LOC)
- `src/mlframe/reporting/renderers/save.py` (288 LOC)
- `src/mlframe/reporting/__init__.py` (59 LOC)
- `src/mlframe/reporting/_diagnostics_dispatch_extra.py` (661 LOC)
- `src/mlframe/reporting/_risk_coverage_diagnostic.py` (62 LOC)
- `src/mlframe/reporting/auto_dispatch.py` (340 LOC)
- `src/mlframe/reporting/catalog.py` (169 LOC)
- `src/mlframe/reporting/colors.py` (140 LOC)
- `src/mlframe/reporting/diagnostics_dispatch.py` (877 LOC)
- `src/mlframe/reporting/output.py` (117 LOC)
- `src/mlframe/reporting/report_html.py` (240 LOC)
- `src/mlframe/reporting/spec.py` (396 LOC)

28 files, 9087 LOC reviewed (matches `wc -l` exactly). No file was skipped or partially read; every file above was opened top-to-bottom this session. `src/mlframe/reporting/charts/**` and `src/mlframe/reporting/charts/_benchmarks/**` are out of my cluster's scope and were not reviewed except where a file in scope imports a symbol from them (import surface only, not internals).

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness | `src/mlframe/training/reporting/_reporting_regression/__init__.py:186-260,350-363,555-588,624-626,744` | `apply_prediction_envelope_clip` output (`preds_arr`) feeds MAE/RMSE/R2/title metrics, but the scatter/residual chart, the residual audit, and the function's own return value all keep using the raw, unclipped `preds` -- so exactly during the catastrophic-extrapolation scenario the clip exists to guard against, the printed metrics and the chart/audit/returned predictions tell two different stories. |
| F2 | P1 | correctness / robustness | `src/mlframe/training/reporting/_reporting_probabilistic.py:345` | `if not classes:` crashes with `ValueError: truth value of an array with more than one element is ambiguous` whenever a caller passes `classes` as a numpy ndarray of length >= 2 -- a case the function's own type hint (`Sequence | np.ndarray | None`) and docstring explicitly document as supported. |
| F3 | P1 | silent-failure / observability | `src/mlframe/reporting/renderers/save.py:254-256` (vs. `233-253`) | The single-backend path of `render_and_save` (the common case: one backend, one-or-more formats) calls `_do_backend` with no `try/except` and never calls `_record_render_failure`, unlike the multi-backend `ThreadPoolExecutor` path a few lines above -- a rendering exception there is neither counted in `get_render_failure_stats()` nor guaranteed to be caught (several call sites elsewhere in the codebase call `render_and_save` with no wrapping `try/except` of their own). |
| F4 | P2 | UI / backend-parity | `src/mlframe/reporting/renderers/plotly.py:735-760` | `PlotlyRenderer._line` sets `showlegend=any(labels)` identically on every series trace instead of per-series (`bool(labels[i])`), so in a multi-series `LinePanelSpec` where only some series carry a label, plotly renders a blank/`None` legend entry for the unlabeled series; the matplotlib renderer does not have this problem (unlabeled artists are excluded automatically by `ax.get_legend_handles_labels()`). |
| F5 | P2 | silent-failure | `src/mlframe/training/reporting/_reporting.py:55-64` | `_reporting_field_default` catches bare `except Exception` with no logging and permanently caches `None` for the field on ANY failure (not just the documented "config unavailable" case) -- a real bug in this lookup (e.g. a pydantic API change) would silently and permanently disable the `panel_emphasis="data_aware"` default-detection feature for the rest of the process with zero trace in the logs. |
| F6 | P2 | concurrency | `src/mlframe/reporting/renderers/_kaleido.py:24-46,131-169` | Module-level mutable state (`_KALEIDO_SERVER_STARTED`, `_KALEIDO_PERSISTENT_FAIL_COUNT`, `_KALEIDO_PERSISTENT_BURNED`, the oneshot counters) is read/written with a check-then-act pattern and no lock; two threads calling `render_and_save` concurrently (e.g. two different figures being saved on separate joblib/thread-pool workers) can race `_ensure_kaleido_server_started` and the failure counters. |
| F7 | P2 | correctness (minor) | `src/mlframe/training/reporting/_reporting_regression/_mtr.py:77-96` | Per-target MTR charts are built from the full `_yt_k`/`_yp_k` columns (including non-finite rows), while the accompanying metrics/audit for the same target use the `_mask_k`-filtered (finite-only) subset -- a target with any NaN/Inf predictions gets a chart that was not built on the same rows as the numbers next to it. |
| F8 | P2 | test-gap | `src/mlframe/training/reporting/_reporting_regression/_sensors.py` + `__init__.py` | `tests/training/reporting/test_reporting_regression.py::test_apply_prediction_envelope_clip_moved_body_runs` only exercises `apply_prediction_envelope_clip` in isolation; no test asserts that `report_regression_model_perf`'s returned `preds`, its residual audit, or its chart actually reflect the clip -- exactly the gap that let F1 ship unnoticed. |
| F9 | P2 | reproducibility (documented tradeoff, flagged for completeness) | `src/mlframe/reporting/renderers/_trend.py:48` | `robust_fit_endpoints` seeds its subsample with a hardcoded `np.random.default_rng(0)` with no caller-supplied seed hook. Alternative reading: this is a pure visual overlay (trend line on a scatter/heatmap), and a fixed seed is probably deliberate so the same chart looks identical across re-renders of the same data -- flagged per the audit's reproducibility checklist item, not asserted as a bug. |

### F1 -- prediction-envelope clip does not reach the chart / audit / return value

`apply_prediction_envelope_clip` (`_sensors.py`) is documented as clipping predictions "to a sigma window around the train (or eval-fallback) target range BEFORE metrics + chart" -- i.e. its own docstring promises both consumers get the sanitized values. In `report_regression_model_perf` (`_reporting_regression/__init__.py`), the clip is applied to a local `preds_arr` (line 187) which then feeds every numeric metric (MAE/RMSE/R2/the extended block/title tokens, lines 219-344) and the collapse sensor (line 255). But:

- The residual audit at line 356 is computed as `_audit_residuals_fn(targets, preds)` -- the ORIGINAL, un-reassigned `preds`, not `preds_arr`.
- The DSL chart path at line 559 calls `_plot_residual_diagnostics(targets, preds, audit=_audit, ...)` -- again raw `preds`.
- The legacy matplotlib path builds its OWN `_preds_arr` at line 577 from `np.asarray(preds, ...)` (raw `preds`, despite the similar variable name) and plots that.
- The function's own return statement at line 744 is `return preds, None` -- the raw, unclipped array.

Concretely: on a group-aware split where a linear model extrapolates catastrophically (the exact scenario this feature targets, per the comments in `_sensors.py` referencing a "prod incident"), the title/log will show a metric computed on sane, sigma-clipped predictions (e.g. a plausible R2), while the scatter plot and residual-audit hypothesis text right next to it are computed on the raw catastrophic predictions -- and the caller of `report_model_perf`/`report_regression_model_perf` (which stores the returned `preds` as `entry.test_preds`, later consumed by `render_model_comparison_from_suite`, `render_split_comparison_from_suite`, `render_prediction_stability_diagnostic`, and every post-fit diagnostic in `_reporting_diagnostics.py::_render_post_fit_diagnostics`, all of which receive this same `preds`) gets the unclipped values throughout the rest of the reporting pipeline. The one place the clip actually lands is the numeric metrics dict / title string.

Suggested-fix direction: after computing `preds_arr` via the clip, either (a) rebind `preds = preds_arr` once (single source of truth for the rest of the function, including the return statement), or (b) if returning the raw model output is intentional (e.g. so downstream ensembling never sees clamped values), thread `preds_arr` explicitly into the audit + both chart paths instead of `preds`, and document in the docstring that the return value is deliberately the raw prediction while metrics/chart differ. Either way the current state -- chart and audit silently diverging from the metrics computed two lines above them -- is not the documented "BEFORE metrics + chart" contract.

### F2 -- `classes` ndarray crashes `report_probabilistic_model_perf`

`report_probabilistic_model_perf`'s signature types `classes: Sequence | np.ndarray | None = None` and the docstring says "Class labels. If None, inferred from model or targets." -- explicitly inviting an ndarray. At line 345, `if not classes:` is used to detect the "caller left it at the default" case. `bool(np.array([...]))` on any ndarray with 2+ elements raises `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`. No caller internal to `mlframe` currently passes `classes=model.classes_` (all internal call sites route `classes=None` and let the function infer it), so this is not hit in-repo today, but it is a documented public API contract violation reachable by any external caller (or a future internal caller) following the type hint literally, e.g. `report_model_perf(..., classes=model.classes_)`.

Suggested-fix direction: `if classes is None or len(classes) == 0:` (or `np.asarray(classes).size == 0` to handle both list and ndarray uniformly).

### F3 -- inconsistent failure handling between single- and multi-backend `render_and_save`

The multi-backend branch (`len(output.backends) > 1`) wraps each `_do_backend` call in a `Future.result(timeout=60)` with `except _FutureTimeout` / `except Exception`, incrementing `_RENDER_FAILURE_COUNT` via `_record_render_failure` either way (lines 236-253). The single-backend branch (the common case -- most call sites in this codebase request one backend, e.g. `"plotly[html,png]"`) is a bare list comprehension with no exception handling at all (line 256). A rendering exception in the single-backend path is neither counted by `get_render_failure_stats()` (so the suite-end "N charts silently dropped" summary undercounts) nor caught -- it propagates to whatever called `render_and_save`. Most in-cluster callers (`diagnostics_dispatch._save_spec`, `auto_dispatch.render_multi_target_panels`, `_reporting_diagnostics.py`) do wrap their own call in `try/except`, but at least two call sites outside this cluster (`training/targets/_target_temporal_plot.py:65`, `training/targets/regression_residual_audit.py:696`) call `render_and_save` directly with no visible wrapping try/except in the shown context, so a single-backend rendering failure there would propagate uncaught.

Suggested-fix direction: route the single-backend path through the same try/except + `_record_render_failure` bookkeeping as the multi-backend path (a plain function call inside a `try:` is cheap; no need for the thread pool).

### F4 -- plotly multi-series line legend shows blank entries for unlabeled series

`PlotlyRenderer._line` builds one `go.Scatter` trace per series and sets `showlegend=any(labels)` on EVERY trace (the same expression, not indexed by `i`). If `labels = ("AUC", None)`, both traces get `showlegend=True`; the second trace's `name=None` renders as an empty/`"undefined"` legend row in the HTML output. The matplotlib backend collects legend handles via `ax.get_legend_handles_labels()`, which by matplotlib convention omits artists with no explicit label, so the same `LinePanelSpec` renders cleanly there -- a real (if cosmetic) cross-backend parity gap.

Suggested-fix direction: `showlegend=bool(labels[i]) if i < len(labels) else False` per trace.

### F5 -- silent, permanent-per-process swallow in `_reporting_field_default`

```python
def _reporting_field_default(field_name: str):
    cache = _reporting_field_default.__dict__
    if field_name not in cache:
        try:
            from ..configs import ReportingConfig
            cache[field_name] = ReportingConfig.model_fields[field_name].default
        except Exception:
            cache[field_name] = None
    return cache[field_name]
```
The comment says this exists to tolerate "config unavailable" (an import-cycle concern), but the `except Exception` also silently absorbs e.g. a `KeyError` from a renamed/removed `ReportingConfig` field, or an `AttributeError` from a pydantic version bump changing `model_fields`'s shape -- with no `logger.debug`/`warning` call anywhere in the function. Because the result is memoized in `cache`, one such failure disables `panel_emphasis="data_aware"`'s "is this still at the default" detection for every subsequent call in the process, invisibly.

Suggested-fix direction: narrow the except to `(ImportError, KeyError)` (the two "config genuinely unavailable" cases) and add a `logger.debug` on the fallback so a real bug is at least discoverable in verbose logs.

### F6 -- unlocked module-global kaleido state

`_KALEIDO_SERVER_STARTED`/`_KALEIDO_PERSISTENT_FAIL_COUNT`/`_KALEIDO_PERSISTENT_BURNED`/the oneshot counters in `_kaleido.py` are plain module globals mutated via classic check-then-act (`if _KALEIDO_SERVER_STARTED: return True` ... later `_KALEIDO_SERVER_STARTED = True`) with no `threading.Lock`. `render_and_save`'s own multi-backend path already proves concurrent rendering happens via `ThreadPoolExecutor`; a caller that kicks off multiple independent `render_and_save` calls concurrently (e.g. per-model or per-split report generation parallelized by the training suite) could race two threads through `_ensure_kaleido_server_started` simultaneously. The likely outcome is a harmless duplicate `kaleido.start_sync_server()` call (kaleido's own `silence_warnings=True` suggests idempotency is expected), but the failure counters (`_KALEIDO_PERSISTENT_FAIL_COUNT`, `_RENDER_FAILURE_COUNT` in `save.py`) can under/over-count under a genuine race, and `_KALEIDO_PERSISTENT_BURNED` could be flipped inconsistently between threads.

Suggested-fix direction: guard the state transitions with a module-level `threading.Lock()`; the cost is negligible relative to the ~13s Chromium spawn this code already amortizes.

### F7 -- MTR per-target chart plots unmasked rows the metrics exclude

In `render_mtr_report` (`_mtr.py`), `_mask_k = np.isfinite(_yt_k) & np.isfinite(_yp_k)` filters the rows used for `audit_residuals`/MAE/RMSE/R2 (lines 86-89), but `build_regression_panel_spec` at line 91 is called with the UNMASKED `_yt_k, _yp_k`. Any NaN/Inf point in a target column reaches the scatter/histogram panel even though it was excluded from every number printed alongside it.

Suggested-fix direction: pass `_yt_k[_mask_k], _yp_k[_mask_k]` to `build_regression_panel_spec` as well.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | test-coverage | `tests/training/reporting/test_reporting_regression.py` | Add an end-to-end test asserting that when the envelope clip actually fires (extreme out-of-range `preds`), `report_regression_model_perf`'s returned `preds`, the chart's plotted values, and the residual-audit input are ALL consistent with each other and with the printed MAE/RMSE/R2 -- would have caught F1. |
| PR2 | test-coverage | new test for `report_probabilistic_model_perf` | Add a regression test calling `report_probabilistic_model_perf(..., classes=np.array([0, 1, 2]))` (an ndarray with >=2 elements) to pin the documented `Sequence \| np.ndarray \| None` contract and catch F2's `ValueError`. |
| PR3 | test-coverage | `tests/reporting` (renderer-level) | No test in the visible reporting test surface renders a `LinePanelSpec` with a mixed labeled/unlabeled series tuple and asserts on the resulting plotly legend trace count/labels -- would have caught F4. |
| PR4 | refactor / DRY | `matplotlib.py` vs `plotly.py` `_per_series_flags` | The identical `_per_series_flags` helper is duplicated verbatim in both `matplotlib.py` (lines 66-73) and `plotly.py` (lines 111-118). Moving it to `_shared_helpers.py` (which already exists for exactly this purpose -- see `_thin_tick_positions`/`_finite_range`) would remove the duplication and the drift risk it creates (a fix applied to one copy and not the other). |
| PR5 | perf / minor | `save.py::render_and_save` | For the single-backend, single-format case (the majority of calls in this codebase, e.g. most `_save_spec` diagnostic calls request one backend + one format) `multi_output` is computed via `len(output.backends) > 1 or any(len(fmts) > 1 ...)` on every call; this is O(1) already so not a real cost, but folding it together with the F3 fix (adding the try/except) is a natural single change. |
| PR6 | docs | `report_html.py::_render_entry_body` | `plotly_html_fragment` is embedded verbatim into the combined-report HTML with no escaping (documented as intentional -- "we do not re-render or sanitise"). This is safe today because fragments are only ever produced by mlframe's own plotly renderer, never from user-controlled text, but the module docstring could say so explicitly (a future caller passing an externally-sourced HTML fragment here would have an XSS-shaped footgun). |

## Coverage notes

- `src/mlframe/reporting/charts/**` (the domain-specific `compose_*_figure`/`build_*_spec` builders that `diagnostics_dispatch.py`, `auto_dispatch.py`, and the `_reporting_*` modules call into) is out of this cluster's assigned scope and was deliberately not read beyond the public function signatures needed to understand call sites in-scope. Some of the findings above (e.g. F7's `build_regression_panel_spec`) only concern how in-scope code CALLS that layer, not the layer's own internals.
- `mlframe/training/_prediction_envelope_clip.py` (the `TrainEnvelopeStats`/`clip_predictions_to_train_envelope` implementation that `_sensors.py` calls) lives directly under `mlframe/training/`, not under `mlframe/training/reporting/`, so it is outside this cluster's directory scope; F1's finding is about how the reporting layer USES the clip's output, not about the clip's own arithmetic, which was not audited.
- `mlframe/reporting/renderers/_kaleido.py`'s claimed timing/perf numbers in comments (e.g. "~13s per call", "~0.13s warm") were taken on faith from the code comments -- verifying them would require running the suite with kaleido installed, which this read-only audit did not do.
