# Wave 23 audit 2026-11-02 -- tracker

Five read-only reports by Opus 5.5 low-effort agents on areas no earlier wave covered: the `shap_proxied_fs` interaction / extractor modules, `estimators/`, the remaining `core/` and `competition/` modules, the reporting dispatcher / split comparison / plotly heatmap, and the remaining metric and evaluation modules. The agents could not write files, so the reports are their returned text, condensed. Every claim was re-checked on HEAD before a fix. Slow suites were not run locally (left to CI, per the owner's 2026-11-03 instruction). Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Severity | Location |
|---|---|---|---|
| **RESOLVED** | W23SHP-1 | High | LightGBM categorical (`==`) splits were read as numeric thresholds by the flat TreeSHAP extractor; such models now fall back to `shap` |
| **RESOLVED** | W23SHP-2 | High | LightGBM `missing_type == "Zero"` routed NaN / 0.0 by the threshold instead of `default_left`; same fallback |
| **FUTURE** | W23SHP-3 | Medium | Faith-Shap surrogate losses omit the fitted intercept `a0`, so they are shifted against the search's real losses when merged |
| **FUTURE** | W23EST-1 | High | `EarlyStoppingWrapper` has no effect on HistGradientBoosting (no `n_estimators` / `estimators_`); needs a `_predictors` truncation or a refit at the best stage |
| **RESOLVED** | W23EST-2 | High | `additive_interaction_diagnostic` ratio additive / full inverted for negative scores; the recommendation now uses the relative deficit |
| **RESOLVED** | W23EST-3 | Medium | multiclass was scored on class 1's probability only |
| **FUTURE** | W23EST-4 | Medium | `EarlyStoppingWrapper` is not a classifier / regressor to sklearn and exposes no `classes_` |
| **RESOLVED** | W23EST-5 | Medium | unseeded stratified / uniform dummy baselines |
| **FUTURE** | W23EST-6 | Low | exploration scores raise `IndexError` for a known candidate outside the search space |
| **RESOLVED** | W23CORE-1 | High | `fit_composite_similarity` predicted multiclass targets as 0 / 1 (rounded weighted mean); now a per-class vote |
| **RESOLVED** | W23CORE-2 | Medium | `ewma` propagated one NaN through the rest of the series; now matches pandas `ignore_na=False` for both adjust modes |
| **RESOLVED** | W23REP-1 | Medium | split-comparison bars drew a missing classification metric as quality 0 |
| **RESOLVED** | W23REP-2 | Medium | one non-finite residual removed its bin from the worst-bias segment search |
| **FUTURE** | W23REP-3 | Low-Medium | overfit verdict and split ordering look up `"train"` / `"test"` case-sensitively |
| **FUTURE** | W23MET-1 | Medium | gains / lift table breaks tied scores by row order and cuts deciles at fixed positions (same class as W20REP-1; changes output) |
| **RESOLVED** | W23MET-2 | Medium | Tweedie deviance with power > 2 accepted y = 0 and returned a negative value |
| **FUTURE** | W23MET-3 | Low-Medium | weighted ICE class reduction counts positives unweighted |
| **RESOLVED** | W23MET-4 | Low | ordinal cutpoints accepted labels outside `[0, n_classes)` |
| **FUTURE** | W23MET-5 | Low | KTC region for the odds-combine dispatch can route small shapes to the GPU |

## Not covered by this wave

`_shap_proxy_interactions`, `_shap_proxy_interaction_proxy`, `_shap_proxy_treeshap_gpu`, `_shap_proxy_gpu*`, `_shap_proxy_gradient`, `_shap_proxy_heuristics`, `_shap_proxied_fit_interactions` / `_residual` / `_resolvers`, `_shap_proxy_core_stability`, `_shap_proxy_refine_helpers`; `diagnostics_dispatch`, the plotly line / scatter / network renderers and the matplotlib renderer, and the remaining chart modules (`class_structure_heatmap`, `prediction_stability`, `shap_interactions`, `interaction_strength`, `training_curve`, `calibration_by_feature`, `calibration_heatmap_2d`, `pdp_2d`, `spectral_embedding`, `fuzzy_membership`, `model_card`); `_classification_extras_blocks`, `_classification_report`, the fused regression block kernels.

## Implemented

| ID | Regression test |
|---|---|
| W23SHP-1, 2 | `tests/feature_selection/shap_proxied/test_lightgbm_extractor_refuses_unmodelled_splits.py` |
| W23EST-2, 3, 5 | `tests/models/test_additive_diagnostic_negative_scores_and_dummy_seed.py` (the multiclass change has no separate test) |
| W23CORE-1 | `tests/core/test_composite_similarity_multiclass.py` |
| W23CORE-2 | `tests/core/test_ewma_skips_nan_like_pandas.py` |
| W23REP-1 | `tests/reporting/test_split_comparison_missing_metric_is_a_gap.py` |
| W23REP-2 | `tests/reporting/test_error_bias_nan_residual_does_not_hide_the_worst_bin.py` |
| W23MET-2, 4 | `tests/metrics/test_tweedie_support_and_ordinal_label_range.py` |
