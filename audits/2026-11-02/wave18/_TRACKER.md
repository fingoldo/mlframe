# Wave 18 audit 2026-11-02 -- tracker

Six read-only reports by Opus 5.5 low-effort agents on the areas wave 17 left unread: metric cores, `models/` ensembling, calibration and evaluation modules, composite HPO and transforms, the MRMR feature-engineering step, and the rest of the training core. The agents could not write files, so the reports are their returned text, condensed. Every claim was re-checked on HEAD before a fix. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Severity | Location |
|---|---|---|---|
| **RESOLVED** | W18MET-1 | High | `fast_roc_curve` read `sample_weight` past its end when shorter than `y_true` |
| **RESOLVED** | W18MET-2 | Medium | `fast_roc_auc_unstable` did not normalise labels to {0,1} |
| **RESOLVED** | W18MET-3 | Medium | tie-free bootstrap AUC paths skipped the label normalisation |
| **RESOLVED** | W18CAL-1 | High | `odds_ratio_combine` counted the base rate once per member; subtracts `(sum w - 1)` prior logits, prior estimated from the members unless given (`prior=None` keeps the old sum) |
| **RESOLVED** | W18CAL-2 | Medium | `_check_target_identity` refused disjoint binary splits with equal class counts |
| **RESOLVED** | W18CAL-3 | Medium | a constant member was scored as correlation 0 and diluted the redundancy summary |
| **RESOLVED** | W18CAL-4 | Low | the correlation fallback ignored caller weights |
| **RESOLVED** | W18CAL-5 | Low | DeLong p-value used `1 - cdf`, exactly 0 beyond \|z\| 8.3 |
| **RESOLVED** | W18MOD-1 | High | blend weights were never fitted for binary classifiers: (N, 2) probabilities were flattened to 2N values and skipped |
| **RESOLVED** | W18MOD-2 | High | NaN warm-up rows of time-ordered OOF switched the member quality gate off |
| **RESOLVED** | W18MOD-3 | Medium | one NaN cell in one member disabled the outlier gate in `ensemble_probabilistic_predictions` |
| **RESOLVED** | W18MOD-4 | Medium | a weight-0 member predicting exactly 0 zeroed the weighted harmonic blend |
| **RESOLVED** | W18MOD-5 | Medium | the streaming harmonic mean gave 1.0 where the materialised path gives 0 |
| **RESOLVED** | W18MOD-6 | Low | blend weights are now applied only when the gate source is `oof` or `calib`; on val/test/train the fit is recorded in `res["_stacking_gate"]` with `weights_skipped_reason` and the blend stays uniform (`tests/models/ensembling/test_stacking_weights_need_oof_source.py`) |
| **RESOLVED** | W18CMP-1 | High | `select_oof_pool_ensemble` always raised for time-ordered HPO (the first block is never tested, NaN in every trial) |
| **DOC** | W18CMP-2 | Medium | pruned Optuna trials are absent from `trials` and the OOF pool, so pool positions are not `study.trials` numbers; both lists stay aligned with each other, the docstring now says so |
| **RESOLVED** | W18TRN-1 | High | `metadata["model_schemas"]` was keyed by model file name alone; a second target or pre-pipeline overwrote the first record (also closes W17TRN-4) |
| **RESOLVED** | W18MRM-1 | Medium | a failing fit-setup step left fe_auto's temporary `fe_*_enable` flags switched on |
| **RESOLVED** | W18MRM-2 | Low | `_fe_recommended_flags_` survived into later fits |
| **FUTURE** | W18MRM-3 | Low | the inert-leaf filter parses operand names from the engineered feature name; a column name with characters outside `[A-Za-z0-9_.]` skips the check; the agent did not confirm the digit-name case; the fix carries operand indices from the recipe |

## Not covered by this wave

Metric cores `_ice_metric`, `classification/`, `regression/`, `rank_correlation`, `_gpu_metrics`; `models/` outside ensembling; `estimators/`; calibration `post`/`quality`/`_ktc_dispatch`; evaluation `reports` and the leak / adversarial / drift / subset-search modules; composite `ensemble/` and `discovery/`; MRMR recipe replay at transform time; `training/feature_handling/` and most of `training/neural/`.

## Implemented

| ID | Regression test |
|---|---|
| W18MET-1, 2, 3 | `tests/metrics/test_auc_label_and_weight_validation.py` |
| W18CAL-1, 5 | `tests/calibration/test_odds_ratio_combine_counts_the_prior_once.py`, `tests/evaluation/test_delong_p_value_does_not_underflow.py` (the closed-form tests in `test_biz_val_calibration_ensembling.py` now pass their known base rate 0.5) |
| W18CAL-2, 3, 4 | `tests/calibration/test_calibration_ensembling_edge_fixes.py` |
| W18MOD-1 | `tests/models/test_stacking_gate_accepts_binary_probability_matrices.py` |
| W18MOD-2, 3 | `tests/models/test_member_gate_survives_nan_warmup_rows.py` |
| W18MOD-4, 5 | `tests/models/test_harmonic_blend_zero_member_handling.py` |
| W18CMP-1 | `tests/training/composite/test_oof_pool_selection_ignores_uncovered_warmup_rows.py` |
| W18TRN-1 | `tests/training/test_model_schema_records_do_not_overwrite_across_targets.py` |
| W18MRM-1, 2 | `tests/feature_selection/filters/test_mrmr_fe_auto_flags_restored_after_failed_setup.py` |
