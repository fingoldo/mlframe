# LOOP30 — 2-D partial-dependence response surface (INTERPRETABILITY)

## Design
New `src/mlframe/reporting/charts/pdp_2d.py`:
- `compose_pdp_2d_figure(model, X, feat_x, feat_y, *, feat_x_name, feat_y_name, grid=20, sample_rows=2000)`
  renders the model's average predicted response over a (feat_x, feat_y) quantile grid as a **filled contour** with
  overlaid white iso-response contour lines + a colorbar (predicted P(y=1) for a binary classifier, else predicted value).
- Surface computation reuses `compute_pdp_2d` from `pdp_ice.py`: the cost is `grid` predict calls over a
  (sample*grid) tiled design block (one batched predict per outer-grid value), NOT grid^2 per-cell predicts.
- Pair selection: explicit `(feat_x, feat_y)` OR default to the top SHAP-interaction pair (tree model + shap), else
  top-2 by `feature_importances_`/`coef_`, else first two columns.
- Edge-safe: constant feature -> annotated note (no degenerate contour); discrete/categorical -> discrete grid via the
  underlying sweep; non-predict-proba classifier/regressor -> predict path.
- `interaction_residual(surface)`: subtracts the additive (row-mean + col-mean - global) two-way-ANOVA reconstruction;
  returns `residual_rms` and `residual_ratio = residual_rms / surface_std` (interaction strength relative to total
  variation). This is the quantitative non-additivity metric.

Complementary to the SHAP interaction ranking (iter22): SHAP ranks pair STRENGTH, the 2-D PDP SHOWS the surface SHAPE
(saddle/twist). Not redundant — RESOLVED, not rejected.

## biz_value interaction-residual numbers (interacting vs additive)
Synthetic, RandomForestRegressor, grid=16, sample=600:
- Product model y=f0*f1 (strong interaction): `residual_rms` LARGE, `residual_ratio > 0.4`.
- Additive model y=2*f0+3*f1 (separable): `residual_rms` tiny, `residual_ratio < 0.05`.
- Assertion: `interacting_residual_rms > 3x additive_residual_rms` AND ratio_int>0.4 / ratio_add<0.05 — PASS.
Unit-level pure-grid checks: additive grid residual_rms < 1e-9 (ratio<1e-6); product grid residual_ratio > 0.5;
constant surface ratio == 0.

## cProfile wall
grid=20 / sample_rows=2000, RandomForestRegressor(60 trees), n=4000, K=5:
- warm wall ~1.7s; profiled cumtime ~2.05s.
- 88% of wall is the 20 (=grid) sklearn forest `predict` calls (1.80s cumulative across 20 calls).
- The grid^2-free batched path holds: **20 predicts, not 400**. The predict-call count (=grid) is the documented
  cost lever; grid + sample_rows caps bound the wall. No actionable in-module speedup (cost is the estimator's predict).

## Tests
`tests/reporting/test_pdp_2d.py` — 9 tests, all PASS (4.98s):
unit (surface shape, bounded values, residual additive vs product vs constant, default-pair via importance,
constant-feature annotate, custom axis names, explicit-pair figure) + biz_value (interaction residual >3x additive).

## Wiring
`ReportingConfig.pdp_2d_charts = False` (opt-in; grid extra predicts per pair). FLAG DECLARED.
Render-pipeline wiring into the report assembly = FUTURE (keeps the report hot path untouched this loop).

## Gallery
`docs/gallery/pdp_2d/pdp_2d.png` (planted f0*f1 twist; saddle surface visible). Added `@entry("pdp_2d", ...)` to
`scripts/render_gallery.py`.

## Disposition: RESOLVED
Commit hash: 8058bc2d.
