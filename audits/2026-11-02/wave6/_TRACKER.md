# Wave 6 audit 2026-11-02 -- tracker

Three read-only reports on `src/mlframe` by Opus 5.5 low-effort agents (CUDA host launchers, parameter validation of the public estimators, log forging), plus one finding made while running the tests (a circular import). The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (safe, or fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | LNC6-1 | cuda_launchers | Low | radix v3 scratch allocated without a VRAM check |
| **RESOLVED** | LNC6-2 | cuda_launchers | Low | two fused-kernel fallbacks logged at debug |
| **RESOLVED** | VAL6-1 | param_validation | High | `n_bins=0` in the njit ECE kernel writes out of bounds |
| **RESOLVED** | VAL6-2 | param_validation | Medium | calibration `alpha` outside (0, 1) |
| **RESOLVED** | VAL6-3 | param_validation | Medium | `inner_cv_splits` below 2 and `n_bootstrap=0` |
| **RESOLVED** | VAL6-4 | param_validation | Medium | `HybridSelector.tree_rich_ops` as a string, unknown names dropped |
| **RESOLVED** | VAL6-5 | param_validation | Medium | `HybridSelector.vote` below 1 |
| **RESOLVED** | VAL6-6 | param_validation | High | `EarlyStoppingWrapper` budget, patience, fraction not validated |
| **RESOLVED** | VAL6-7 | param_validation | Low | MRMR `stability_n_bootstrap=0` fell back to classic with a warning |
| **RESOLVED** | LOGF6-1 | log_forging | Low | bare `%s` with column or target names (~55 sites) |
| **RESOLVED** | LOGF6-2 | log_forging | Low | exception text through `%s` (~760 sites) |
| **RESOLVED** | LOGF6-3 | log_forging | Info | names from file paths or files (~15 sites) |
| **RESOLVED** | LOGF6-4 | log_forging | Info | f-string log in `training/phases.py:328` |
| **RESOLVED** | LOGF6-5 | log_forging | Low | no central control-character escaping |
| **RESOLVED** | LOGF6-6 | log_forging | Low | workflow-command injection through CI logs |
| **REJECTED** | LOGF6-7 | log_forging | Info | calls already using `%r` |
| **RESOLVED** | IMP6-1 | import_cycle | Medium | `mlframe.metrics.regression` imported the feature_selection package at module level, closing a cycle through `estimators` |

## Counts

17 findings: 2 High, 5 Medium, 7 Low, 3 Info. Dispositions: 16 RESOLVED, 1 REJECTED, 0 FUTURE.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| LNC6-1, LNC6-2 | Check free VRAM before the radix v3 scratch and log the three radix fallbacks | `tests/feature_selection/gpu/test_radix_select_fallback_visible.py` |
| VAL6-1, VAL6-2, VAL6-3 | Refuse degenerate calibration parameters before the ECE kernel runs | `tests/calibration/test_calibration_parameter_validation.py` |
| VAL6-6 | Validate EarlyStoppingWrapper's budget and fraction before fitting | `tests/estimators/test_early_stopping_parameter_validation.py` |
| VAL6-4, VAL6-5 | Reject unknown tree_rich_ops and a vote below 1 in HybridSelector.fit | `tests/feature_selection/test_hybrid_selector_parameter_validation.py` |
| VAL6-7 | Raise on stability_n_bootstrap below 1 instead of warning and fitting classic | `tests/feature_selection/mrmr/core/test_mrmr_stability_bootstrap_bound.py` |
| IMP6-1 | Break the metrics to feature_selection to estimators import cycle | `tests/test_import_entry_points_have_no_cycle.py` |
| LOGF6-1 to LOGF6-6 | Escape control characters in the arguments of mlframe log records | `tests/utils/test_log_sanitize.py` |

## Not covered by this wave

- The three `_ktc` launcher files and `_fe_batched_mi_guards.py` were only grepped.
- ShapProxiedFS and the training suite entry points were checked by targeted greps only.
- A hostile argument longer than 512 characters or with more than 3 newlines still keeps its line breaks in the log (the multi-line reports depend on that).
