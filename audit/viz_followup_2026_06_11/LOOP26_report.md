# LOOP26 — 2D calibration-ECE heatmap (ACCURACY / exotic)

## Design
New `src/mlframe/reporting/charts/calibration_heatmap_2d.py`:
- `compute_calibration_heatmap_2d(y_true, y_score, feat_x, feat_y, *, n_bins=5)` — headless metric grid.
- `compose_calibration_heatmap_2d_figure(..., feat_x_name, feat_y_name, n_bins=5)` — RdYlGn_r HeatmapPanelSpec.

Both features are quantile-binned into an `n_bins x n_bins` grid (rows=feat_y low->high, cols=feat_x). Per-cell
ECE = `|mean(y_score) - mean(y_true)|` (cell-conditional calibration gap; with both features quantile-binned this
isolates a localized over/under-confidence pocket the pooled / 1D per-feature views average away). Green=calibrated,
red=miscalibrated. Each cell annotated with ECE + support n. Headline = worst-cell ECE + its `(f_x bin, f_y bin)`
location + traffic-light (<0.05 green, <0.10 amber, else red).

Edge-safe: NaN rows dropped; cells with <30 rows greyed (NaN, masked); a feature with <2 distinct quantile edges is
annotated and the grid skipped (degenerate-axis note), composer falls back to an AnnotationPanelSpec.

Efficiency: cell assignment = two `np.searchsorted` on quantile edges; sums = three weighted `np.bincount` over the
flattened `cy*nx + cx` index — single O(n) pass, no per-cell python loop over data. Inputs above 1M rows are uniformly
subsampled before the pass.

## biz_value (localized-corner numbers)
Synthetic: `p = sigmoid(0.8*f0 + 0.8*f1)`, model overconfident (+0.35 to score) ONLY where f0>median AND f1>median.
- worst-cell ECE = **0.296**, median-cell ECE = **0.013** -> ratio **22.7x** (floor asserted >=3x), traffic-light **red**.
- worst cell lands in the high-f0/high-f1 quadrant (bin (3,3); test asserts wx>=3 and wy>=3). The extreme (4,4) cell's
  base prob is already ~1 so its clipped gap is smaller — the pocket peaks at (3,3), inside the boosted quadrant.
- Uniform control (score == true probability everywhere): worst-cell ECE **<0.05**, traffic-light **green** (flat grid).

## cProfile (n=2M, subsampled to 1M)
Wall **237.6 ms**; single O(n) pass. Top cumulative: the compute fn itself (bincount/searchsorted, tottime 0.140s) +
`np.quantile` edges (0.054s) + `np.partition` (0.048s, inherent to quantile binning). No actionable hotspot — already
one pass; quantile-edge cost is irreducible without changing the binning contract.

## Tests
`tests/reporting/test_calibration_heatmap_2d.py` — 9 tests, all pass (4.2s):
unit (grid shape+labels+support, per-cell ECE == manual gap, low-support cell greyed, degenerate-axis skip, NaN drop,
figure is HeatmapPanelSpec/RdYlGn_r, degenerate -> AnnotationPanelSpec) + biz_value (localized-corner 22.7x + worst-loc
in high/high quadrant; uniform control green).

## Wiring (RESOLVED — wired, not dormant)
All three report-path files were git-clean (composite session is on composite/* + CHANGELOG only), so wired default-on:
- `_reporting_configs.py`: new `ReportingConfig.calibration_heatmap_2d_charts: bool = True`.
- `_reporting.py`: reads the flag, passes through.
- `_reporting_probabilistic.py`: new param + `_render_calibration_heatmap_2d` renders the TOP-2-importance feature pair
  for binary targets when a feature frame is present; surfaces `metrics['calibration_heatmap_2d']`
  (worst_ece / worst_cell / median_cell_ece / traffic_light / feat_x / feat_y).
- Exported from `reporting/charts/__init__.py`.

## Gallery
`docs/gallery/calibration_heatmap_2d/calibration_heatmap_2d.png` (rendered in isolation to avoid touching other PNGs);
new `@entry` added to `scripts/render_gallery.py`. Red pocket clearly visible in the high/high quadrant.

## Verdict
**RESOLVED** — not redundant with iter21's 1D view: the localized pocket (ECE 0.296) is invisible to either 1D
per-feature curve because either feature alone averages the high-corner pocket against the calibrated rows at the other
feature's low values; only the joint 2D grid exposes it.
