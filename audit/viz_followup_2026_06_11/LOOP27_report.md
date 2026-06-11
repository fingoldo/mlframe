# LOOP27 — per-instance SHAP for TOP-K worst predictions (INTERPRETABILITY, exotic)

## Design
New `src/mlframe/reporting/charts/shap_per_instance.py`:
`shap_worst_errors_explanation(model, X, y_true, y_score, *, feature_names, k=4, max_explain_rows=2000)`.

- Error severity: binary score in [0,1] -> confidence-wrong magnitude `|y_true - y_score|` (confident-wrong
  positives/negatives float to the top); else raw `|residual|`. Top-K by descending severity, worst first.
- `is_tree_model` gate: only tree/GBDT (exact fast `TreeExplainer`). Non-tree -> SKIP (no per-row
  KernelExplainer; too slow for a niche panel).
- ONE explainer on a bounded background (`max_explain_rows`, K worst rows always included) + ONE SHAP-value
  computation for the K rows. Small-multiples: one signed-SHAP horizontal bar per instance, annotated with
  predicted prob / true label / |err| and top contributing features (sorted by |SHAP|).
- Edge-safe: empty input / length mismatch / non-tree -> `skipped`; perfectly-separated (binary, max
  severity < 0.5) -> rendered with a "no misclassifications; showing largest residuals" suptitle.
- Reuses shap_panels helpers (`_as_frame_and_names`, `_shap_values_2d`, `_row_subset`, `_matplotlib_formats`).

## biz_value — responsible-feature-dominates (the headline result)
Synthetic `y = 1 iff f0 > 0`; 60 rows with `f0 < -0.3` get an f2 outlier spike (uniform 6..9) and are
MISLABELLED positive at train time, so the forest learns "big f2 -> positive". At score time those rows are
confident-wrong positives (true=0, prob~1). The per-instance attribution must blame f2.

Result: among planted rows landing in the top-K worst errors, **f2 is the largest |SHAP| in 100% of them**
(test floor: >=75%, >=4 planted rows present). Gallery PNG confirms f2 (red, ~+0.62) dominates every panel,
f0 the lone offsetting negative. The explanation correctly attributes each costly error to the responsible
feature.

## Tests (7 passed, 6.94s)
`tests/reporting/test_charts_shap_per_instance.py`:
top-K-by-severity selection, K bound, non-tree skip, no-errors annotate, paths written, biz_value
(f2 dominates planted-error attributions), cProfile (background bounded by `max_explain_rows`, asserts
`n_background <= 500` at n=8000). No-figure-leak autouse fixture.

## cProfile
At n=8000 with `max_explain_rows=500`, `n_background` is capped at 500: TreeExplainer cost scales with the
bounded background + K rows, not n. No actionable hotspot beyond TreeExplainer itself (third-party).

## Wiring (opt-in)
`ReportingConfig.shap_per_instance = False` (default off — niche + extra render cost beyond the global
beeswarm). Added to `_reporting_configs.py` (clean file). Catalog/diagnostics_dispatch dispatch wiring is
FUTURE: the catalog dispatch table is intricate and needs y_true/y_score plumbing per (model, split); the
standalone API + config flag ship now, the dispatch hookup is deferred.

## Gallery
`docs/gallery/shap_per_instance/shap_per_instance.png` (K=4 worst-error explanations); renderer
`_render_shap_per_instance()` added to `scripts/render_gallery.py` + registered in `main()`.

## Verdict: RESOLVED
Per-instance worst-error SHAP is clearly informative and NOT redundant with global SHAP: global beeswarm
ranks drivers in aggregate; this panel pins the responsible feature for a SPECIFIC costly error (the f2
attribution is invisible in a population-level beeswarm where f0 dominates).
