# Wave 6 audit 2026-11-02: constructor and fit() parameter validation of the public estimators

Scope: MRMR, HybridSelector, RFECV, ShapProxiedFS (targeted greps), EarlyStoppingWrapper, `pick_best_calibrator`, training entry points; read-only, Opus 5.5 low effort. Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| VAL6-1 | High | `calibration/policy.py:111-116` `_ece_score_numba_serial`: `n_bins=0` writes `sum_p[-1]` out of bounds and scores 0 for every candidate | RESOLVED |
| VAL6-2 | Medium | `calibration/policy.py:644`: `alpha` outside (0, 1) gives an infinite, inverted or NaN interval | RESOLVED |
| VAL6-3 | Medium | `calibration/_policy_pick_helpers.py:75`: `inner_cv_splits` below 2 silently raised to 2; `n_bootstrap=0` fails with the wrong cause | RESOLVED |
| VAL6-4 | Medium | `hybrid_selector.py:221, 450-451`: `tree_rich_ops="absd"` split into characters, unknown names silently dropped | RESOLVED |
| VAL6-5 | Medium | `hybrid_selector.py:245, 834-866`: `vote` below 1 confirms every cluster | RESOLVED |
| VAL6-6 | High | `estimators/early_stopping.py`: `max_iter`, `max_runtime_mins`, `patience`, `validation_fraction`, `start_iter` not validated; `fit` returned an unfitted-looking result | RESOLVED |
| VAL6-7 | Low | MRMR flat `stability_n_bootstrap=0`: the outer handler turned the failure into a warning and a classic fit | RESOLVED |

VAL6-1 to VAL6-3: `pick_best_calibrator` and `_ece_score` raise a `ValueError` naming the parameter. Test: `tests/calibration/test_calibration_parameter_validation.py`.

VAL6-4: a bare string is one operator name, unknown names raise, and the constructor stores its argument unchanged. The report's claim that `sklearn.clone` failed on this was not borne out: `HybridSelector` has no `get_params`, so it is not a clone-able estimator. Test: `tests/feature_selection/test_hybrid_selector_parameter_validation.py`.

VAL6-5: `vote` must be an integer of at least 1 (checked at the start of `fit`).

VAL6-6: `fit` raises a `ValueError` for `max_iter < 1`, `start_iter < 1`, `patience < 1`, a non-positive `max_runtime_mins` and a `validation_fraction` outside (0, 1). Test: `tests/estimators/test_early_stopping_parameter_validation.py`.

VAL6-7: the check sits ahead of the try that falls back to a classic fit, so it is an error, not a warning. Test: `tests/feature_selection/mrmr/core/test_mrmr_stability_bootstrap_bound.py`.

Checked and validated: RFECV (`cv`, `frac`, `max_refits`, epsilon and decay knobs, enum knobs, runtime), MRMR (row counts, enum knobs, `sample_weight`, concurrent `fit`), ShapProxiedFS enum knobs and target checks, `drop_columns` given as a string, `target_name` type, the calibration `selection` enum and OOF row-count checks.
