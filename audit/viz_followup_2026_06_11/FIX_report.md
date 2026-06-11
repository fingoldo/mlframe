# Gallery visual-quality follow-up fixes

All 6 review findings fixed + 1 optional. Targeted tests green (105 + 104 passed across two batches);
affected gallery PNGs re-rendered (30/30 entries OK, 0 failed).

## Fixes

### 1. Confusion-matrix colormap -> CB-safe sequential viridis
- `src/mlframe/reporting/charts/multiclass.py`: import `HEATMAP_CMAP` (was `CONFUSION` RdBu_r diverging);
  `_confusion_panel` now passes `colormap=HEATMAP_CMAP` (multiclass.py:31, :143). Counts / row-rates are
  unsigned magnitudes, so a sequential map is correct; cell-count annotations + auto text contrast kept.
- Test: `tests/reporting/test_charts_multiclass.py::TestPanelTypes::test_confusion_uses_cb_safe_sequential_cmap_not_diverging`.
- Re-rendered: `docs/gallery/multiclass/multiclass_full.png`.

### 2. Quantile reliability legend overflow
- `src/mlframe/reporting/charts/quantile.py` `_quantile_reliability_panel`: per-tau nominal reference lines no
  longer each emit a legend label (was ~2K entries); one "nominal tau (dotted)" key + K obs-curve labels, obs and
  its nominal share a per-tau color, `legend_outside=True` (quantile.py:404-432).
- `src/mlframe/reporting/spec.py`: added `LinePanelSpec.legend_outside` + `legend_ncol` (spec.py:197-200).
- `src/mlframe/reporting/renderers/matplotlib.py` `_line`: honors `legend_outside` (legend placed center-left at
  bbox (1.02,0.5), fontsize 7) + `legend_ncol` (matplotlib.py:486-493).
- Test: `tests/reporting/test_charts_quantile.py::...::test_quantile_reliability_legend_does_not_emit_per_tau_nominal_entries`.
- Re-rendered: `docs/gallery/quantile/quantile_full.png`.

### 3. Quantile grid density
- `src/mlframe/reporting/charts/quantile.py` `compose_quantile_figure`: per-cell defaults bumped
  `cell_width 6.0->7.5`, `cell_height 4.0->4.8` (quantile.py:653-654). No panels removed.
- Re-rendered: `docs/gallery/quantile/quantile_full.png` (same file as #2).

### 4. Slice-finder ordering (worst-on-top)
- `src/mlframe/reporting/charts/slice_finder.py`: removed the `cats[::-1]` / `vals[::-1]` pre-reverse
  (slice_finder.py:278-285). The table is already worst-first and the matplotlib renderer inverts the y-axis for
  horizontal bars, so the prior code double-reversed and put the worst slice at the BOTTOM. Now worst is on TOP,
  matching the title. Global-reference vline + support/ratio annotations kept.
- Test: `tests/reporting/test_charts_slice_finder.py::test_bar_categories_are_worst_first_to_match_title`.
- Re-rendered: `docs/gallery/slice_finder/slice_finder.png`.

### 5. SHAP beeswarm summary (canonical, not interaction grid)
- `src/mlframe/reporting/charts/shap_panels.py` `shap_summary_and_dependence`: beeswarm now calls
  `shap.summary_plot(shap_mat, vals_sample, plot_type="dot", ...)` on the 2-D per-sample SHAP matrix
  (positive class for classifiers) instead of the raw Explanation, which made shap render a per-class /
  interaction grid; figure widened to 9in so the x-axis label is not clipped (shap_panels.py:301-310).
  Top-K dependence plots unchanged.
- Covered by existing `tests/reporting/test_charts_shap_panels.py` (importorskip shap).
- Re-rendered: `docs/gallery/shap_panels/shap_shap_beeswarm.png` (+ dependence PNGs unchanged in shape).

### 6. Calibration-drift panel spacing
- `src/mlframe/reporting/charts/calibration_drift.py` `build_calibration_drift_spec`: 2-panel figure now
  `constrained_layout=True` + height `figsize[1]*1.7 -> *2.0` (calibration_drift.py:240-247), so the top panel's
  rotated date x-tick labels no longer collide with the bottom panel's "Reliability curve per window" title.
- Covered by existing `tests/reporting/test_charts_calibration_drift.py`.
- Re-rendered: `docs/gallery/calibration_drift/calibration_drift.png`.

### Optional: calibration reliability bubble/histogram color -> sequential viridis
- `src/mlframe/reporting/charts/calibration.py`: bin-population is a sequential quantity; scatter bubble color +
  histogram bar color switched from `CALIBRATION` (RdYlBu diverging) to `HEATMAP_CMAP` (viridis)
  (calibration.py:20, :177, :198).
- Re-rendered: `docs/gallery/binary/calibration_reliability.png`.

## Tests
Batch 1 (multiclass, slice_finder, quantile, calibration_drift, shap, perfplot, renderers, default_panels_e2e):
105 passed. Batch 2 (quantile_w3e, cb_safe_colormaps, spec_vocabulary, renderers_w5_vocabulary,
vectorized_panels): 104 passed. All green.

## Render
`python scripts/render_gallery.py` -> rendered 30, failed 0.
