# Wave 9 audit 2026-11-02 -- tracker

Three read-only reports on `src/mlframe` by Opus 5.5 low-effort agents (swallowed exceptions, time and timezone handling, aliasing and in-place mutation). The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | EXC9-1 | swallowed_exception | High | RFECV leakage screen stopped at the first unconvertible column (`rfecv/_validate.py`) |
| **RESOLVED** | EXC9-2 | swallowed_exception | Medium | failed early-stopping cap on the OOF clone swallowed (`training/_oof_fold_policy.py`) |
| **RESOLVED** | EXC9-3 | swallowed_exception | Medium | signed linear coefficients silently left unscaled (`wrappers/_helpers_importance_agg.py`) |
| **RESOLVED** | EXC9-4 | swallowed_exception | Low | calibrated probabilities and NaN-guard statistics not attached, no log (`evaluation.py`, `_predict_guards.py`) |
| **RESOLVED** | TIME9-1 | time | Medium | stdlib datetime segment edges converted to the host's local zone (`targets/_target_temporal_types.py`) |
| **RESOLVED** | TIME9-2 | time | Low | partial_fit wrapper stamped its callback with `time.time()`, the callback reads `default_timer` |
| **RESOLVED** | TIME9-3b | time | Low | tz-aware timestamps made calendar-feature detection return nothing (`charts/_calendar_features.py`) |
| **DOC** | TIME9-3a | time | Low | calendar features derived in a local zone do not match UTC timestamps, so they can appear as drift drivers; needs a circular-shift-invariant match and a benchmark Closed as DOC: calendar features are meaningful only in the zone the timestamps were recorded in, and a circular-shift-invariant match would need its own benchmark for a Low finding; derive calendar features from one consistent zone before the fit. Residual risk: a mismatched zone can show up as a drift driver. |
| **REJECTED** | ALIAS9-0 | aliasing | Info | no aliasing or in-place mutation bug found; every `inplace=True`, params-dict and `fit_params` site copies or documents the mutation |

## Counts

9 rows: 1 High, 2 Medium (+1 Medium TIME9-1), 4 Low, 1 FUTURE, 1 REJECTED. Dispositions: 7 RESOLVED, 1 FUTURE, 1 REJECTED.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| EXC9-1 | Scan every column in the RFECV leakage screen, not only those before a failing one | `tests/feature_selection/wrappers/test_rfecv_leakage_screen_nullable_columns.py` |
| EXC9-2 | Log and record a failed early-stopping cap on the OOF clone | `tests/training/test_oof_clone_cap_failure_is_logged.py` |
| EXC9-3 | Warn when signed linear coefficients cannot be scaled by the train stds | `tests/feature_selection/wrappers/test_signed_coef_std_failure_is_logged.py` |
| EXC9-4 | Log when calibrated probabilities or NaN-guard statistics cannot be attached to a model | `tests/training/test_predict_guard_stats_attach_failure_is_logged.py` |
| TIME9-1 | Convert datetime segment edges to UTC, not to the host's local zone | `tests/training/targets/test_bin_ts_to_naive_utc.py` |
| TIME9-2 | Stamp the partial_fit wrapper's callback with the clock the callback reads | `tests/training/test_partial_fit_es_callback_clock.py` |
| TIME9-3b | Read tz-aware timestamps as UTC in calendar-feature detection | `tests/reporting/test_calendar_features_tz_aware_timestamps.py` |

## Not covered by this wave

- The exceptions agent sampled the 269 narrow-exception hits by module; lower-impact cases may remain in the feature-engineering filters.
- The aliasing agent did not cover constructor attributes reassigned in `fit()`, chained-assignment views, or numpy in-place operations on caller arrays.
- TIME9-1 and TIME9-2 had no failing-on-old-code run on this machine: TIME9-1's test only differs on a non-UTC host, and TIME9-2 was checked by reading.
