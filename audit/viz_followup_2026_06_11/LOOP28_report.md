# LOOP28 — SHAP interactions / per-instance suite wiring

## Where the SHAP dispatch lives
Per-(model, split) global SHAP charts (beeswarm/dependence) dispatch from
`src/mlframe/training/reporting/_reporting.py:510` via
`render_shap_diagnostic(...)` (defined in `src/mlframe/reporting/diagnostics_dispatch.py:576`).
This site has `model`, `df`, `names` (feature_names), `y_arr` (y_true), `probs`/`y_pred`
(scores), `cfg`, `plot_outputs`, `plot_file`, `metrics` — everything both niche charts need.

## What was wired

### shap_interactions
- The `ReportingConfig.shap_interactions` flag did NOT actually exist (only `shap_per_instance` did);
  ADDED it (`shap_interactions: bool = False`, `shap_interaction_max_rows: int = 2000`) in
  `training/_reporting_configs.py`.
- New `render_shap_interactions_diagnostic(...)` in `diagnostics_dispatch.py` mirrors
  `render_shap_diagnostic`: tree-only (`is_tree_model`), try/except best-effort, records
  `shap_interactions` chart + paths, calls `shap_interaction_summary(model, df, feature_names=..., max_rows, ...)`.
- Gated block in `_reporting.py` after the `shap_panels` dispatch: runs only when `cfg.shap_interactions` is True.

### shap_per_instance
- Flag already existed (`shap_per_instance: bool = False`) but had NO dispatch.
- New `render_shap_per_instance_diagnostic(...)` in `diagnostics_dispatch.py`: tree-only, best-effort,
  records `shap_per_instance` chart + paths, calls `shap_worst_errors_explanation(model, df, y_true, y_score, feature_names=...)`.
- Gated block in `_reporting.py`: runs only when `cfg.shap_per_instance` is True AND a 1-D score exists
  (`_binary_positive_score(probs)` for binary, `y_pred` for regression; else skipped — multiclass/multilabel have no single score).

Both reuse the existing `is_tree_model` guard and the chart composers' built-in hard row caps
(interactions=2000, per-instance background=2000), so RAM-safe; both wrapped in the same try/except shape
as the existing SHAP panel (a niche chart never aborts the report). The composers build their own
TreeExplainer inside the gated block (only when opted in) — no recompute on the default path.

## Default-off-stays-off proof
`test_binary_suite_renders_diagnostics_default_on` (runs with both flags at default False) now asserts
NEITHER `shap_interactions` nor `shap_per_instance` file appears. PASSED.

## e2e result
`test_opt_in_shap_interaction_and_per_instance_charts_render_when_enabled` (importorskip shap): tiny HGB
binary run with `shap_interactions=True, shap_per_instance=True` writes both artifact files. PASSED.
Full run: `2 passed` (opt-in + default-off), plus `14 passed` dispatch-followup regression.

## FUTURE-skipped hops
None. All four target files were clean (no foreign-dirty). Per-instance for multiclass/multilabel is
intentionally skipped (no single positive-class score) — by design, not a missing hop.

## Commit
See final reply for hash.
