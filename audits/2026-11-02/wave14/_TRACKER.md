# Wave 14 audit 2026-11-02 -- tracker

Three read-only reports by Opus 5.5 low-effort agents: the sklearn estimator contract, documentation against behaviour of public functions, and state handling under joblib and thread pools. The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | SKL14-1 | sklearn_contract | High | `ShapProxiedFS.transform` renamed the caller's DataFrame columns in place |
| **RESOLVED** | SKL14-2 | sklearn_contract | High | `ACESelector.transform` sliced by position |
| **RESOLVED** | SKL14-3 | sklearn_contract | High | functional adapters' `transform` sliced by position |
| **RESOLVED** | SKL14-4 | sklearn_contract | High | `CorrelatedFeaturesSelector.transform` sliced by position and raised AttributeError before fit |
| **RESOLVED** | SKL14-5 | sklearn_contract | Medium | `HybridSelector.transform` before fit raised AttributeError; the empty-selection fallback to the first column stays as documented |
| **RESOLVED** | SKL14-6 | sklearn_contract | Medium | `ClassifierWithEarlyStopping` never set `classes_` and kept stale `feature_names_in_` |
| **RESOLVED** | SKL14-7a | sklearn_contract | Medium | `BinaryPostCalibrator.predict_proba` before fit did not raise NotFittedError |
| **FUTURE** | SKL14-7b | sklearn_contract | Medium | `BinaryPostCalibrator.fit` fits the `calibrator` constructor parameter in place and keeps learned state in `p_cal` / `y_cal`; the fix (clone into `calibrator_`) changes what callers who inspect `.calibrator` after fit see |
| **FUTURE** | SKL14-8 | sklearn_contract | Medium | `RFECV.__init__` validates and overlays config objects onto flat kwargs; moving it into `fit` touches the whole parameter surface of a large class and needs its own pass |
| **RESOLVED** | SKL14-9 | sklearn_contract | Low | `HybridSelector(random_state=RandomState)` raised TypeError |
| **RESOLVED** | SKL14-10 | sklearn_contract | Low | `StabilityFESelector.transform` before fit raised RuntimeError |
| **RESOLVED** | DOC14-1 | doc_drift | Medium | `report_ndigits` documented as 4, code default 2 (three functions) |
| **RESOLVED** | DOC14-2 | doc_drift | Medium | `plot_sample_size` documented as 500, code default 5000 |
| **RESOLVED** | DOC14-3 | doc_drift | Low | `num_factors` documented as 10, code default 15 |
| **RESOLVED** | DOC14-4 | doc_drift | Low | `figsize` documented as (15, 10), code default (7.5, 2.5) |
| **RESOLVED** | DOC14-5 | doc_drift | Low | RFECV `cv` documented as None, signature default 3 |
| **RESOLVED** | DOC14-6 | doc_drift | Low | `assert_no_group_leakage`: near-duplicate error masked a later exact group leak (code), generator consumption and inclusive thresholds (doc) |
| **RESOLVED** | PAR14-1 | parallel_state | High | FE stage-concurrency and edge-pool workers got only the deadline, not the fit's scoring thread-locals |
| **RESOLVED** | PAR14-2 | parallel_state | Medium | Fleuret permutation confirmation depended on the worker count (per-worker seeds, cumulative in-place shuffle); now per-permutation seeds from the outer base seed with per-worker offsets, columns restored before each permutation. The per-worker `max_failed` early stop stays a soft limit |
| **REJECTED** | PAR14-3 | parallel_state | Low | `bootstrap_metrics` serial and parallel loops use different streams; the docstring states it (`n_jobs != 1` uses per-resample seeding, independent of worker count, statistically equivalent to the serial stream, pin `n_jobs=1` for byte identity). Making the serial loop per-resample was tried and reverted: the fused bundle and the manual per-iteration reference test depend on the single stream |

## Counts

20 rows. Dispositions: 16 RESOLVED, 2 FUTURE, 1 REJECTED (SKL14-7 is counted in two rows).

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| SKL14-1..4 | Select columns by name in the fitted selectors' transform and stop renaming the caller's frame | `tests/feature_selection/test_selectors_transform_by_column_name.py`, `tests/feature_selection/shap_proxied/test_shap_proxied_transform_keeps_callers_columns.py` |
| SKL14-4, SKL14-5, SKL14-10 | Raise NotFittedError from HybridSelector and StabilityFESelector transform before fit | `tests/feature_selection/test_selectors_raise_not_fitted_error.py` |
| SKL14-6 | Expose classes_ from the early-stopping classifier wrapper and drop stale column names on refit | `tests/estimators/test_early_stopping_classifier_exposes_classes.py` |
| SKL14-7a | Raise NotFittedError from BinaryPostCalibrator.postcalibrate_probs before fit | `tests/calibration/test_post_calibrator_not_fitted.py` |
| SKL14-9 | Accept a RandomState or Generator as HybridSelector's random_state | `tests/feature_selection/test_hybrid_selector_seed_int.py` |
| DOC14-1..6 | Check exact group leaks in every fold before near-duplicates, and fix docstring defaults | `tests/evaluation/test_group_leakage_guard_exact_overlap_wins.py` (DOC14-1..5 are docstring edits with no test) |
| PAR14-2 | Seed the Fleuret permutation confirmation per permutation, not per worker | `tests/feature_selection/info_theory/test_fleuret_confidence_independent_of_worker_count.py` |
| PAR14-1 | Publish the fit's scoring settings to the FE worker threads | `tests/feature_selection/filters/test_edge_pool_publishes_fit_context.py` |

## Not covered by this wave

- Parallel sites: the audit read about 15 of the roughly 60; whether pynndescent restores `numba.set_num_threads` after a build (`_row_attention_ann.py`) and nested oversubscription in StabilityMRMR over MRMR clones with `n_jobs=-1` were not traced.
- Documentation drift: defaults were checked with an AST scan; documented return shapes and None-handling were only spot-checked, so `metrics/` and `calibration/` return-shape claims remain for another pass.
- The stage-concurrency path of PAR14-1 is covered through the edge pool test (same capture and scope helper); the stage runner itself has no dedicated test.
