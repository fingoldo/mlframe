# Wave 33 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents (models optimization and ensembling, core helpers and remaining metrics, data datasets and cleaning). Coverage was partial. The agents could not write files, so the reports are their returned text, condensed; each finding is re-checked on HEAD before it is fixed. Dispositions: RESOLVED, FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W33EN-1 | nan | High | `models/ensembling/quality_gate.py:224,111-121`: rows where any member is non-finite are dropped before the member statistics, so a member that is NaN on a few rows passes the gate and the blend is NaN there; only rows where EVERY member is non-finite (the warm-up rows) are dropped for the statistics; a member that is NaN on some rows keeps those rows, gets NaN MAE/STD and is excluded by `_exclude_if_non_finite` |
| **RESOLVED** | W33EN-2 | gate | Medium | `models/ensembling/quality_gate.py:253-264`: a median peer MAE of 0 makes the relative threshold 0, which switches the relative check off, so a catastrophic outlier is kept; the relative thresholds are floored at 1e-12 when enabled, so peers that agree exactly no longer switch the check off |
| **RESOLVED** | W33EN-3 | nan | Medium | `models/ensembling/float_aggregation.py:40-48`: a NaN member makes the robust flavour return a silent 0.0; the robust flavour is NaN-aware (nanmedian/nanmean); an all-NaN column stays NaN |
| **FUTURE** | W33OP-1 | search | Medium | `models/_optimization_search.py:649,654`: best/worst tracking runs before the finite check, so a failed (infinite) evaluation poisons the search state |
| **FUTURE** | W33OP-2 | search | Medium | `models/_optimization_search.py:254-270`, `optimization.py:198`: warm-start evaluations never count toward the returned best |
| **FUTURE** | W33OP-3 | splits | Medium | `models/selection.py:73-78` (`GroupTimeSeriesSplit.split`): a pandas Series of groups is indexed by label, not position |
| **RESOLVED** | W33EN-4 | metric | Medium | `models/ensembling/selection.py:151` (`_score_blend`): the default AUC metric is silently wrong for a non-binary 1-D target; a 1-D blend over a target with more than two values raises a ValueError asking for an explicit metric instead of casting labels to int |
| **FUTURE** | W33DA-1 | missingness | High | `data/datasets/_missing.py:77-78`: MAR and MNAR masks keep `ordering >= cut`, so ties at the cut hide far more than the requested rate |
| **FUTURE** | W33DA-2 | missingness | Medium | `data/datasets/_missing.py:77`: a NaN in the ordering column makes the quantile NaN and nothing is masked, silently |
| **FUTURE** | W33DA-3 | labels | Low | `preprocessing/align_feature_direction.py:102`: only `y == 1` is the positive class, so other two-class labels raise a wrong single-class error |
| **FUTURE** | W33CM-1 | degenerate | Low | `metrics/classification/_weighted_kappa.py:67-83`: identical single-class raters with `n_classes >= 2` give 0.0 where the docstring promises 1.0 |
| **FUTURE** | W33CM-2 | empty | Low | `metrics/_multilabel_metrics.py:137,169,206`: empty input divides by zero inside numba |
| **FUTURE** | W33CM-3 | empty | Low | `metrics/classification/_ordinal_cutpoints.py:113-115`: empty input reaches `np.quantile` on a zero-length array |

## Not covered by this wave

- models/ensembling: `base.py`, `flavour_policy.py`, `member_metrics.py`, `predict.py`, `process_method.py`, `score.py`, `score_flavours.py`, `score_levels.py`, `score_stages.py`, `score_validate.py`, `_combine_fallback.py`, `_correlation.py`; models: `tuning.py`, `_tuning_types.py`, `additive_interaction_diagnostic.py`; `tuning_catboost.py` `randint(1, 16)` stops at depth 15 (Low, possibly intended).
- core: `arrays.py`, `helpers.py`, `category_encoders_compat.py`, `frame_compat.py`, `stats.py`, `shared.py`; metrics: `regression/_regression_extras.py`, `_regression_corr.py`, `_regression_deviance.py`, `classification/_classification_extras*.py`, `_multilabel_extras.py`, `_ranking_extras.py`, hamming and subset-accuracy kernel bodies.
- data: `datasets/spec.py`, `ground_truth.py`, `_scm.py`, `_links.py`, `_oracle*.py`, `_target*.py`, `_derived.py`, `_latent.py`, `_sapp.py`, `frames.py`, `from_real.py`, `replay.py`, `_columns.py`, `synthetic.py`; preprocessing `cleaning.py`, `_cleaning_helpers.py`, `auto_transform_select.py`.

## Implemented

| ID | Regression test |
|---|---|
| W33EN-1, 2 | `tests/models/ensembling/test_quality_gate_nan_member_and_zero_median.py` |
| W33EN-3, 4 | `tests/models/ensembling/test_float_aggregation_and_default_metric_edge_cases.py` |
