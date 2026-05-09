# Visual audit harness for mlframe charts

Rendering every chart type x both backends (matplotlib + plotly) for
manual inspection. Not part of the automated `pytest` suite — visual
inspection cannot be automated reliably (font metrics differ across
matplotlib versions, plotly serialises layout differently across
releases, kaleido auto-positions axes, etc.).

Run **before shipping** any change to:

- `mlframe.reporting.charts.*` — chart spec builders
- `mlframe.reporting.renderers.*` — matplotlib / plotly backends
- `mlframe.reporting.spec.*` — FigureSpec / PanelSpec dataclasses
- `ReportingConfig.plot_outputs` DSL or any `*_panels` template

## Run

```bash
python -m mlframe.tests.reporting.visual_audit.render_all_charts
# or
python -m mlframe.tests.reporting.visual_audit.render_all_charts --out /tmp/audit
```

Output dir defaults to `D:/Temp/chart_audit/` on Windows.

## What gets rendered (one matplotlib + one plotly PNG each)

| # | Chart | Panels | Tests |
|---|-------|--------|-------|
| 01 | `calibration_binary` | scatter + histogram (binary classification) | `tests/reporting/test_charts.py::TestCalibrationChart` |
| 02 | `regression` | 3-panel: scatter / residual hist / resid-vs-pred | `tests/reporting/test_charts.py::TestRegressionChart` |
| 03 | `multiclass` | 6-panel grid: confusion / PR_F1 / ROC / calib_grid / prob_dist / top_k_acc | `tests/reporting/test_charts_multiclass.py` |
| 04 | `multilabel` | 5-panel grid: PR_F1 / calib_grid / co-occurrence / cardinality / jaccard_dist | `tests/reporting/test_charts_multilabel.py` |
| 05 | `ltr` | 5-panel grid: NDCG@k / NDCG_dist / lift / MRR_dist / score_by_rel | `tests/reporting/test_charts_ltr.py` |
| 06 | `quantile` | 5-panel grid: reliability / pinball / interval band / width_dist / pit_hist | `tests/reporting/test_charts_quantile.py` |
| 07 | `temporal` | single line: target rate over time | `tests/reporting/test_temporal_audit.py` |

## What to look for

When inspecting the PNGs side-by-side:

| Symptom | Likely cause | Where to look |
|---------|--------------|---------------|
| matplotlib suptitle overlapping per-panel titles | suptitle `y` position too low; constrained_layout doesn't auto-extend with multiline subplot titles | `mlframe/reporting/renderers/matplotlib.py::render` (suptitle call); the inline regression block in `training/evaluation.py` |
| plotly per-panel titles bleed horizontally into adjacent subplots | subplot title font too large for column width OR `\n` not converted to `<br>` | `mlframe/reporting/renderers/plotly.py::render` (subplot_titles + annotation font) |
| plotly multiline titles render as one line | `\n` instead of `<br>` (plotly uses HTML line-breaks) | same |
| Text clipping at figure edge | figsize too small for content, or font scaling didn't kick in | per-chart spec builder |
| Unreadable axis labels on dense grids | overlapping tick labels — increase `cell_height` or rotate xtick labels | per-chart spec builder |
| Empty / black panels | renderer crash (check stderr) | renderer + chart spec for the failed panel |

## Adding a new chart type

1. Add a new `audit_<type>` function in `render_all_charts.py` that
   builds a representative FigureSpec for the chart.
2. Append it to the `for name, fn in [...]` list in `main()`.
3. Add a row in the table above.
4. Re-run the audit and verify both PNGs render without errors.

## Test infrastructure context

This audit complements the structured / behavioural tests under
`tests/reporting/`:

- `test_charts.py` — calibration + regression FigureSpec correctness
- `test_charts_multiclass.py` — per-token panel builder unit tests
- `test_charts_multilabel.py` — same for multilabel tokens
- `test_charts_ltr.py` — same for LTR tokens
- `test_charts_quantile.py` — same for quantile tokens
- `test_kaleido_recovery.py` — kaleido JS-error sensor (deadlock avoidance)
- `test_legacy_chart_dsl_optin.py` — opt-in path for legacy `show_calibration_plot`

Those tests verify the *data shape* and *spec contract*. The visual
audit verifies the *rendering* (which the structured tests cannot
inspect). Both are required.

Last verified outputs: 2026-05-09 against
`mlframe/reporting/{charts,renderers}/` at HEAD.
