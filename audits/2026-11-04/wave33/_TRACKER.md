# Wave 33 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents (models optimization and ensembling, core helpers and remaining metrics, data datasets and cleaning). Coverage was partial. The agents could not write files, so the reports are their returned text, condensed; each finding is re-checked on HEAD before it is fixed. Dispositions: RESOLVED, FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W33EN-1 | nan | High | `models/ensembling/quality_gate.py:224,111-121`: rows where any member is non-finite are dropped before the member statistics, so a member that is NaN on a few rows passes the gate and the blend is NaN there; only rows where EVERY member is non-finite (the warm-up rows) are dropped for the statistics; a member that is NaN on some rows keeps those rows, gets NaN MAE/STD and is excluded by `_exclude_if_non_finite` |
| **RESOLVED** | W33EN-2 | gate | Medium | `models/ensembling/quality_gate.py:253-264`: a median peer MAE of 0 makes the relative threshold 0, which switches the relative check off, so a catastrophic outlier is kept; the relative thresholds are floored at 1e-12 when enabled, so peers that agree exactly no longer switch the check off |
| **RESOLVED** | W33EN-3 | nan | Medium | `models/ensembling/float_aggregation.py:40-48`: a NaN member makes the robust flavour return a silent 0.0; the robust flavour is NaN-aware (nanmedian/nanmean); an all-NaN column stays NaN |
| **RESOLVED** | W33OP-1 | search | Medium | `models/_optimization_search.py:649,654`: best/worst tracking runs before the finite check, so a failed (infinite) evaluation poisons the search state; only finite evaluations reach the best/worst tracking; a failed one counts as a non-improving step |
| **RESOLVED** | W33OP-2 | search | Medium | `models/_optimization_search.py:254-270`, `optimization.py:198`: warm-start evaluations never count toward the returned best; finite warm-start evaluations seed the best / worst at construction (this also removes the earlier `NOT_READY` with known data but no best) |
| **RESOLVED** | W33OP-3 | splits | Medium | `models/selection.py:73-78` (`GroupTimeSeriesSplit.split`): a pandas Series of groups is indexed by label, not position; `groups` is converted to an array so access is positional |
| **RESOLVED** | W33EN-4 | metric | Medium | `models/ensembling/selection.py:151` (`_score_blend`): the default AUC metric is silently wrong for a non-binary 1-D target; a 1-D blend over a target with more than two values raises a ValueError asking for an explicit metric instead of casting labels to int |
| **RESOLVED** | W33DA-1 | missingness | High | `data/datasets/_missing.py:77-78`: MAR and MNAR masks keep `ordering >= cut`, so ties at the cut hide far more than the requested rate; MAR/MNAR hide an exact count (`round(share * n)`) from a lexsort with the stream breaking ties, so a tied driver no longer hides every tied row (binary driver, rate 0.1: 500 hidden before, 100 now) |
| **RESOLVED** | W33DA-2 | missingness | Medium | `data/datasets/_missing.py:77`: a NaN in the ordering column makes the quantile NaN and nothing is masked, silently; a NaN in the ordering column ranks lowest and is not masked by the rule, and the cut no longer turns NaN |
| **RESOLVED** | W33DA-3 | labels | Low | `preprocessing/align_feature_direction.py:102`: only `y == 1` is the positive class, so other two-class labels raise a wrong single-class error; the larger of two labels is the positive class in `batch_univariate_auc` (as in sklearn); 1 stays positive for other label sets |
| **RESOLVED** | W33CM-1 | degenerate | Low | `metrics/classification/_weighted_kappa.py:67-83`: identical single-class raters with `n_classes >= 2` give 0.0 where the docstring promises 1.0; identical single-class raters with n_classes >= 2 agree perfectly (1.0), matching the docstring |
| **RESOLVED** | W33CM-2 | empty | Low | `metrics/_multilabel_metrics.py:137,169,206`: empty input divides by zero inside numba; empty multilabel input returns NaN from the three public wrappers instead of dividing by zero inside numba |
| **RESOLVED** | W33CM-3 | empty | Low | `metrics/classification/_ordinal_cutpoints.py:113-115`: empty input reaches `np.quantile` on a zero-length array; empty ordinal input raises a clear ValueError instead of an opaque numpy IndexError |

## Not covered by this wave

- models/ensembling: `base.py`, `flavour_policy.py`, `member_metrics.py`, `predict.py`, `process_method.py`, `score.py`, `score_flavours.py`, `score_levels.py`, `score_stages.py`, `score_validate.py`, `_combine_fallback.py`, `_correlation.py`; models: `tuning.py`, `_tuning_types.py`, `additive_interaction_diagnostic.py`; `tuning_catboost.py` `randint(1, 16)` stops at depth 15 (Low, possibly intended).
- core: `arrays.py`, `helpers.py`, `category_encoders_compat.py`, `frame_compat.py`, `stats.py`, `shared.py`; metrics: `regression/_regression_extras.py`, `_regression_corr.py`, `_regression_deviance.py`, `classification/_classification_extras*.py`, `_multilabel_extras.py`, `_ranking_extras.py`, hamming and subset-accuracy kernel bodies.
- data: `datasets/spec.py`, `ground_truth.py`, `_scm.py`, `_links.py`, `_oracle*.py`, `_target*.py`, `_derived.py`, `_latent.py`, `_sapp.py`, `frames.py`, `from_real.py`, `replay.py`, `_columns.py`, `synthetic.py`; preprocessing `cleaning.py`, `_cleaning_helpers.py`, `auto_transform_select.py`.

## Implemented

| ID | Regression test |
|---|---|
| W33EN-1, 2 | `tests/models/ensembling/test_quality_gate_nan_member_and_zero_median.py` |
| W33EN-3, 4 | `tests/models/ensembling/test_float_aggregation_and_default_metric_edge_cases.py` |
| W33OP-1, 2 | `tests/models/test_optimizer_warm_start_and_failed_evaluations.py` |
| W33OP-3 | `tests/models/test_group_time_series_split_accepts_a_pandas_groups_series.py` |
| W33DA-1, 2 | `tests/data/test_datasets_missingness_ties_and_nan_ordering.py` |
| W33CM-1, 2, 3 | `tests/metrics/test_degenerate_inputs_kappa_multilabel_ordinal.py` |
| W33DA-3 | `tests/preprocessing/test_align_feature_direction_two_class_labels.py` |
