# LOOP21 — Per-feature calibration (reliability conditioned on a continuous feature)

## Axis: ACCURACY. Verdict: RESOLVED (shipped + wired, default-on).

## Design
New `src/mlframe/reporting/charts/calibration_by_feature.py`:
- `compose_calibration_by_feature_figure(y_true, y_score, feature_values, *, feature_name, n_feature_bins=4, n_prob_bins=10)` —
  quantile-bins rows by the continuous feature (`np.quantile` + `np.searchsorted`), computes a reliability curve + standard ECE
  per feature-bin via the shared `fast_calibration_binning` njit path, renders a small-multiples row of mini reliability curves
  (one per feature-bin, shared diagonal) + an ECE-vs-feature-bin line. The max-min ECE across feature-bins is the
  "calibration heterogeneity" metric (traffic-light: <0.05 green, <0.10 amber, else red).
- `compute_calibration_by_feature_heterogeneity(...)` — headless metric dict `{per_bin_ece, bin_centers, heterogeneity,
  traffic_light, skipped}` consumed by the biz_value test + suite metrics.

Complements the categorical subgroup fairness-calibration chart by conditioning on a CONTINUOUS feature — a pooled reliability
curve can hide miscalibration that cancels across the feature range (calibrated low, overconfident high).

## Edge-safety
- NaN feature/label/score rows dropped before binning.
- Constant / low-cardinality feature: tied quantile edges collapse via `np.unique`; <2 distinct bins -> honest AnnotationPanel,
  heterogeneity NaN, traffic_light "n/a".
- Single-class bin / <30 rows / no populated prob-bin -> annotated + skipped (not crashed).
- Huge bins subsampled to 200k before the njit pass (curve read on n_prob_bins points -> visually identical, bounded cost).

## biz_value (heterogeneity numbers)
Synthetic: label ~ Bernoulli(score); the upper-half-feature score is concave-warped (overconfident), lower half left calibrated.
- Heterogeneous case (n=40k, 4 bins): low-bin ECE ~0.013, high-bin ECE ~0.26, **heterogeneity 0.249 [red]**.
- Uniformly-calibrated control: **heterogeneity <0.03 [green]**.
- Assertions: hi_ece > lo_ece+0.05; heterogeneous >=0.08; uniform <0.05; gap delta >0.05; lights red/amber vs green.
Gallery render (n=16k): per-bin ECE 0.021 / 0.013 / 0.262 / 0.261, heterogeneity 0.249 [red] — ECE line steps up sharply.

## Wired (NOT dormant)
- `training/reporting/_reporting_probabilistic.py`: new `calibration_by_feature_charts: bool = True` param +
  `_render_calibration_by_feature` + `_top_importance_features` helper. Renders per top-1..2 `model.feature_importances_`
  features pulled from `df` for binary targets; writes `<plot_file>_calibfeat_<feat>.<ext>`; stores
  `metrics["calibration_by_feature_heterogeneity"]`. Per-feature try/except so one bad column never aborts the report.
- `training/_reporting_configs.py`: new default-on `ReportingConfig.calibration_by_feature_charts`.
- `training/reporting/_reporting.py`: threads the config toggle into the probabilistic report call.
- All four shared files were git-clean before edit (verified via `git status --short`); none are training/composite/* or CHANGELOG.
- `reporting/charts/__init__.py`: re-exports both public functions.

## cProfile (n=1e6, 4 feature-bins, warm njit)
Wall 0.166s single call; 3-call profile cumtime 0.537s. Hotspots: `_per_bin_ece` (njit binning per bin) + `np.quantile`/median
(`partition`). Bounded — well under the 2s gate asserted in the harness. No actionable further speedup (njit-bound + O(n) quantile).

## Tests (tests/reporting/test_calibration_by_feature.py) — 8 passed
unit: per-bin curves+ECE present, bin-count respected (4 and 6), metric keys, constant-feature annotated, NaN dropped,
single-class bin skipped, matplotlib render. biz_value: feature-dependent miscalibration detected (numbers above).

## Gallery
- `scripts/render_gallery.py`: new `calibration_by_feature` entry (feature-dependent miscalibration synthetic).
- PNG: `docs/gallery/calibration_by_feature/calibration_by_feature.png` (rendered, ~100KB).

## Not redundant with weak-segment heatmap / subgroup calibration
Weak-segment heatmap measures ERROR by feature slice (point accuracy); subgroup fairness-calibration conditions on a CATEGORICAL
group. This conditions reliability on a CONTINUOUS feature's quantile bins and reports a calibration-specific heterogeneity metric
— a distinct diagnostic (calibration, not accuracy; continuous, not categorical). KEPT, default-on.
