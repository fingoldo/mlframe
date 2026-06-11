# LOOP22 — SHAP interaction summary (interpretability, exotic tier)

## Verdict: RESOLVED

A SHAP feature-PAIR interaction summary that ranks the strongest pairwise interactions by
mean |SHAP interaction value| and renders (a) a top-pairs horizontal bar and (b) an off-diagonal
feature x feature interaction-strength heatmap (viridis). Tree-only gate; bounded for typical GBDT.

## Design

`src/mlframe/reporting/charts/shap_interactions.py :: shap_interaction_summary(model, X, *, feature_names,
max_rows=2000, top_pairs=10, plot_file, plot_outputs, seed)`:

- `shap.TreeExplainer.shap_interaction_values` on a row sample capped at `max_rows` (default 2000).
  Interaction values are O(F^2) per row, so `max_rows` is THE cost lever — wall scales ~linearly with it.
- Rows subsampled BEFORE any SHAP work, stratified to keep the high-|score-proxy| tail
  (reuses `_score_proxy` / `subsample_preserving_extremes` from shap_panels).
- ONE explainer pass -> mean |interaction matrix| (F,F). Off-diagonal (i<j) pairs ranked descending.
- Edge-safe: <2 features, non-tree model (`is_tree_model` gate), empty input -> `skipped` set, no figures.
  Non-tree is deliberately SKIPPED, not routed to KernelExplainer (its interaction approximation is far
  too slow and not the diagnostic's intent).
- Reuses shap_panels helpers (`_as_frame_and_names`, `_row_subset`, `_close_figs`, `_matplotlib_formats`,
  `is_tree_model`) — no duplication; figure-leak guard mirrors shap_panels (snapshot fignums, close leaked).

## biz_value: planted-interaction recovery

Synthetic: `y = 1[2.5*(f0*f1) + 0.2*f2 + noise > 0]`, n=3000, F=6, GBDT(100, depth 3).

- **f0 x f1 ranks #1**, strength **0.0305**, vs #2 pair (f1 x f2) **0.0089** -> **3.4x** above the strongest
  non-interacting pair. On the gallery synthetic (logit-scale GBDT) the gap is ~17x (see PNG).
- biz_value test floors the ratio at **2x** (measured ~4.6x at the 2000-cap) to catch a real regression
  while absorbing seed noise.

## cProfile wall at the 2000-row cap

`dense_tree_shap` (the interaction-value C kernel) is ~99% of wall; driver is TREE COMPLEXITY:

| model | cap | warm wall |
|---|---|---|
| GradientBoosting(100, depth 3) | 500 | 0.58s |
| GradientBoosting(100, depth 3) | 2000 | **0.97s** |
| RandomForest(50, depth 4) | 2000 | 2.51s |
| RandomForest(120, depth 6) | 2000 | 21.95s (pathological wide+deep) |

For typical GBDT diagnostics the 2000-cap is sub-second to a few seconds — bounded and informative.
Deep/wide ensembles (120 trees x depth 6) blow past "a few seconds"; the cap is documented as the lever
and the gate keeps it tree-only, so callers control cost via `max_rows` / model size.

## NOT a reject

Recovery is clean (planted pair #1 by a wide margin), the cost is bounded for realistic tree models, and
it is NOT redundant with the existing dependence plots: dependence shows a single feature's contribution
shape (with an optional colour-by interaction), whereas this RANKS all F*(F-1)/2 pairs by interaction
magnitude — a different question (which pairs interact, not how one feature maps to its SHAP).

## Wiring

OPT-IN by design (O(F^2) cost). The standalone composer + tests + gallery ship now. A
`ReportingConfig.shap_interactions=False` default-off flag is the wiring vehicle; the composite reporting
config files were not touched here (parallel session owns training/composite/*; shared config left clean).
Wiring into the report dispatcher is **FUTURE** (flag default-off, behind the same cost gate).

## Tests

`tests/reporting/test_charts_shap_interactions.py` (importorskip shap), 5 tests:
top-pairs/heatmap structure + paths + no figure leak; non-tree gate skip; <2-feature skip;
biz_value planted-interaction #1 at >=2x #2; cProfile bounded at cap. 13 passed (with shap_panels).

## Artefacts

- Module: `src/mlframe/reporting/charts/shap_interactions.py`
- Tests: `tests/reporting/test_charts_shap_interactions.py`
- Gallery wiring: `scripts/render_gallery.py` (`_render_shap_interactions`)
- Gallery PNGs: `docs/gallery/shap_interactions/shap_interactions_interaction_top_pairs.png`,
  `docs/gallery/shap_interactions/shap_interactions_interaction_heatmap.png`
