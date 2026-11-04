# Wave 32 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents (metrics regression/classification/quantile, models/core, data/preprocessing). Coverage was partial (about a third of metrics, 2 findings in models/core from a few files, 8 of 52 data/preprocessing files). The agents could not write files, so the reports are their returned text, condensed; each finding is re-checked on HEAD before it is fixed. Dispositions: RESOLVED, FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **FUTURE** | W32ME-1 | nan | Medium | `metrics/quantile.py:104-114` (`_fast_coverage`): a NaN in `y_true`, `q_lo` or `q_hi` counts as a missed interval instead of giving NaN |
| **FUTURE** | W32ME-2 | nan | Medium | `metrics/quantile.py:127-134` (`_fast_winkler`): NaN `y` drops the penalty and contributes only the width, so the score is finite and optimistic |
| **FUTURE** | W32ME-3 | labels | Medium | `metrics/_core_brier.py:59-136`: `y_true` is never checked to be in {0, 1}; label 2 gives a plausible Brier score |
| **FUTURE** | W32ME-4 | zero_division | Low | `metrics/_core_precision_mape.py:20-39` (`fast_precision`): the `zero_division` argument is accepted and ignored |
| **FUTURE** | W32ME-5 | zero_division | Low | `metrics/_core_precision_mape.py:99-126` (`fast_classification_report`): empty denominators are hard-set to 0.0 so the later `nan_to_num(..., nan=zero_division)` never acts |
| **FUTURE** | W32ME-6 | empty | Low | `metrics/_core_precision_mape.py:82,94`: empty input gives accuracy 0.0 and balanced accuracy 0.0 instead of NaN |
| **FUTURE** | W32MO-1 | leakage | Medium | `models/tuning_rules.py:533-556`: CatBoost early-stops on `X_test` and then scores `X_test` as the estimator's expected score, replacing the CV mean |
| **FUTURE** | W32MO-2 | scale | Low | `core/robust_location.py:65`: with MAD = 0 the scale falls back to 1.0 in absolute units, so the M-estimator is unit-dependent |
| **RESOLVED** | W32DA-1 | calibration | High | `data/datasets/generator.py:199-230` with `_noise.py:106`: a prior-valued noise rate is redrawn from the shared knob RNG on every `probability_at` call, so calibration, recorded AUC, true probabilities and labels come from different laws; the noise rate is resolved once per target (before the bisection) and passed as a pinned copy of the spec; with the old code a 0.75 calibration target recorded achieved AUC 0.722 against a shipped-probability ceiling of 0.649 |
| **RESOLVED** | W32DA-2 | rng | Medium | `data/datasets/generator.py:189,213,246`: the knob RNG is consumed once per bisection step and then passed to missingness, so prior-valued missingness depends on the calibration target; fixed by the same change: the bisection no longer consumes the knob stream, so prior-valued missingness no longer depends on the calibration target |
| **RESOLVED** | W32DA-3 | target | High | `preprocessing/temporal_drift_augment.py:83-84,140,152`: with `feature_cols=None` a numeric label is treated as a feature and overwritten by its z-score on every synthetic row; new optional `label_cols` keeps named label columns out of the default `feature_cols`, so they are carried from the entity's true last period (the default cannot guess which numeric column is the label; documented) |
| **RESOLVED** | W32DA-4 | nan | Low | `preprocessing/temporal_drift_augment.py:145-146`: a NaN last-period non-feature value takes the earlier vintage's value (a label leak); an entity found in the true-last rows takes its value even when NaN |

## Not covered by this wave

- metrics: `regression/_regression_extras.py`, `_regression_corr.py`, `_regression_deviance.py`, `_regression_benchmark.py`, all of `classification/`, `_multilabel_*`, `_ranking_extras.py`, `core.py`, `iteration_metrics.py`, the rest of `_fairness_metrics.py`.
- models: `_optimization_search.py`, `optimization.py`, `_optimization_shared.py`, `tuning_catboost.py`, `tuning.py`, `lgbm_defaults.py`, `rf_proximity.py`, `additive_interaction_diagnostic.py`, `selection.py`, `ensembling/score_gate.py` (bare `except Exception`); core: `arrays.py`, `helpers.py`, `binning.py`, `ewma.py`, `frame_compat.py`, `matrix_seriation.py`, `proportion_stats.py`, `recency_*`, `stats.py`.
- data: `synthetic.py` and most of `datasets/` and `datasets/scenarios/*`; preprocessing: `cleaning.py`, `_cleaning_helpers.py`, `auto_transform_select.py`, `align_feature_direction.py`, `outlier_detector_zoo.py`.

## Implemented

| ID | Regression test |
|---|---|
| W32DA-3, 4 | `tests/preprocessing/test_temporal_drift_augment_label_handling.py` |
| W32DA-1, 2 | `tests/data/test_datasets_prior_noise_rate_is_drawn_once.py` |
