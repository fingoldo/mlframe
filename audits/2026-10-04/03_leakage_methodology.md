# 03 Data leakage and ML methodology audit (read-only), 2026-10-04

## Scope and method
Read the tracker first: `audits/full_audit_2026-09-20/_TRACKER.md` plus `ensembling_models.md`, `evaluation_reporting.md`, `feature_engineering.md`,
`feature_selection.md`. Closed and not re-reported: EVR-02/03 (leak-scan tolerance, group min-variance), FE-03/FE-05 (forward-looking TE, shuffled KFold default in
the shortlist adapter), FS-15 (prescreen universe), ENS-13.

Read in full or in the relevant part: `training/core/_phase_finalize_calibration.py`, `training/_calibration_models.py`, `training/evaluation.py` (post_calibrate_model),
`training/trainer.py` (`_compute_oof_preds*`), `training/core/_ensemble_chooser.py`, `models/ensembling/process_method.py`,
`training/core/_phase_composite_post_xt_ensemble/__init__.py` (stack branch), `training/pipeline/_target_encoding_composite_fe.py`, `_pipeline_fit_transform.py`,
`calibration/threshold_optimizer.py`, `training/_honest_decision_threshold.py`, `training/honest_diagnostics.py`, `evaluation/bootstrap.py`,
`feature_selection/{forward_select,hybrid_selector,ace,cv_policy}.py`, `preprocessing/outliers.py`, `training/core/_setup_helpers_outliers.py`,
`feature_engineering/{bruteforce,two_step_target_encode}.py`, `competition/train_test_union_frequency.py`.

Greps: fit_transform sites (about 45 hits in training/pipeline/preprocessing/feature_engineering), KFold/train_test_split sites (about 45 hits across
feature_selection/calibration/evaluation/training), val-fed fit/calibrate calls (0 real hits), target-encoding modules (about 30 files).
Probes (2, seconds each, 1 python process): bootstrap CI coverage on a panel; sklearn isotonic with NaN input.
Not reached (time-boxed): mrmr internals (covered by `mrmr_audit_2026-09-14`), data_valuation, ranking suite internals, metrics/ ties, votenrank.

Overall: the default training path is well guarded. Preprocessing and encoders fit on train and transform val/test; calibrators, threshold optimizer and conformal
use a disjoint calib slice with a hard calib==test guard; RFECV/ACE/hybrid/shap-proxy selectors are wired to the temporal/group policy
(`wire_selector_policy`); the outlier detector filters val but never test. No P1 found.

## Findings
| ID | file:line | sev | evidence | mechanism and demonstrated effect | proposed fix | Disposition |
|---|---|---|---|---|---|---|
| LK-01 | `evaluation/bootstrap.py:57` (`bootstrap_metric`), caller `training/honest_diagnostics.py:245,275` | P2 | `bootstrap_metric(..., stratify=...)` resamples rows i.i.d. (class stratification only, no group/block arg); `_bootstrap_block(y_test, p_test, ...)` is called per model with no `group_ids` although `ctx.group_ids` exists | The reported test CI (AUC/Brier/log-loss/ECE/RMSE) treats correlated rows as independent. Probe: 40 groups x 50 rows with a group effect: true SD of AUC across fresh panels 0.0242; mean iid-bootstrap CI width 0.0424 implies SD 0.0108, so the CI is about 2.2x too narrow (nominal 95% covers roughly 70%). Same for autocorrelated series. | Add `groups=` / `block_length=` to `bootstrap_metric(s)` (cluster / moving-block resampling) and pass `ctx.group_ids` / timestamps from `honest_diagnostics`; at minimum stamp `iid_resampling: true` and warn when groups exist. | OPEN |
| LK-02 | `training/evaluation.py:359-385` (binary), `:300-315` (multi-output); producer `training/trainer.py:317-361` | P2 | `_oof_arr = np.asarray(_oof_probs_attr)` then `meta_model.fit(_binary_fit_X, _binary_fit_y, ...)`; producer docstring: "warm-up rows ... are NaN. Downstream OOF consumers mask non-finite rows" | In a temporal suite `oof_has_time` defaults to `policy.temporal` (`_cv_policy_setup.py`), so `oof_probs` carries NaN warm-up rows; `post_calibrate_model` does not mask them. Probe: `IsotonicRegression().fit` with a NaN raises `ValueError: Input X contains NaN`. OOF-based post-calibration is unusable on exactly the temporal suites; `pick_best_calibrator(oof_probs=...)` has the same unmasked input (unverified). | Mask `np.isfinite(oof)` rows jointly with `oof_target` before every fit/pick in `post_calibrate_model`; regression test with time-aware OOF. | OPEN |
| LK-03 | `training/core/_phase_composite_post_xt_ensemble/__init__.py:598-610` | P2 | else branch when `_oof_pred_matrix is None`: `_pred_matrix[:, _ci] = _get_train_pred(_comp, _frame_key2)`; comment at :547 "Honest OOF preds if available, else biased train-set preds" | Default `cross_target_ensemble_strategy="nnls_stack"` (`_composite_target_discovery_config.py:235`). If `compute_oof_holdout_predictions` raises (logged only as "Falling back to train-RMSE proxy", :466) the NNLS/linear stack weights are fitted on in-sample component predictions: over-weights the most overfit member, gate RMSE optimistic. No warning where the in-sample stacking actually happens, no metadata flag. | On this fallback use `from_uniform_weights` or refuse the stack; else stamp `metadata["xt_ensemble_stack_source"]="train_insample"`. | OPEN |
| LK-04 | `training/trainer.py:257-266` | P3 | `estimator.set_params(early_stopping_rounds=None)` on the OOF clone; defaults `iterations=700`, `early_stopping_rounds=100` (`_model_configs.py:332-333`) | OOF fold models run the full 700 rounds while the deployed model stops near its val-optimal round, so OOF predictions come from more overfit models than the shipped one. OOF drives ensemble-flavour choice, confidence shrinkage and conformal-from-OOF. Pessimistic, mostly rank-preserving; impact not measured. | Refit OOF folds with `n_estimators = best_iteration_` of the deployed model when known. | OPEN |
| LK-05 | `training/trainer.py:299-303` | P3 | `splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)`; `except (ValueError, ...): logger.info("OOF prediction skipped: %s")` | I.i.d. classification OOF is plain `KFold`, not stratified. With a rare class a fold can lack a class and the whole OOF silently becomes "skipped" at INFO, which then switches the ensemble chooser to val and skips the threshold block. | `StratifiedKFold` for classifiers when every class has >= n_splits members; WARNING when OOF is skipped for a classifier. | OPEN |
| LK-06 | `training/core/_ensemble_chooser.py:20-24`, `_model_configs_behavior.py:289` | P3 | `oof_n_splits: int = Field(default=0, ge=0)`; chooser falls back to `val.*` | At defaults there is no OOF, so the ensemble flavour is chosen on val (the ES surface of every member): optimistic pick. A WARN is emitted (`_ensemble_chooser.py:174`), test stays untouched. Disclosed by design. | None required. | OPEN |
| LK-07 | `training/core/_phase_finalize_calibration.py:121-175`, `training/_calibration_models.py:392-460` | P3 | threshold fit on raw `calib_probs`; afterwards `entry.model = wrapped` (isotonic wrapper); stored report has no scale tag | `metadata["decision_threshold"]` is in RAW-probability units while the shipped model's `predict_proba` is isotonic-calibrated. No consumer today (grep: only the writer), but any consumer applying it to shipped probabilities gets a shifted operating point. | Record `scale: raw_base_proba`, or fit the threshold on isotonic(calib_probs). | OPEN |
| LK-08 | `feature_engineering/two_step_target_encode.py:18-30` (default `causal=False`) | P3 | docstring: "default False ... a target-leakage source if the output is used as a per-EVENT training feature" | Standalone default leaks later events of an entity into earlier rows. Suite wiring (`_target_encoding_composite_fe.py:77-86`) correctly uses `causal=True` for train and a train-only lookup for val/test, so production is safe; direct callers are exposed. | Default `causal=True`, or warn when `causal=False` and output is per-event. | OPEN |
| LK-09 | `training/core/_phase_finalize_calibration.py:179-255` | P3 | `_e.test_preds = _shrunk_preds` (docstring says "test/val predictions") | Shrinkage rewrites `test_preds` after report metrics were computed and leaves `val_preds` alone; stored metrics describe un-shrunk predictions (the regression-recalibration step tags `metrics_are_pre_recalibration`, this one does not). Inert at defaults (needs `oof_n_splits>=2`). | Tag `metrics_are_pre_shrinkage`, apply to val for symmetry. | OPEN |
| LK-10 | `training/trainer.py:317-361`, `_phase_finalize_calibration.py:223` | P3 | warm-up rows NaN; guard checks only `oof_preds.size != train_target.size` | `compute_oof_confidence` receives an OOF vector with leading NaN in temporal suites; NaN handling there was not verified. | Mask non-finite OOF rows at the call site. | OPEN (unverified) |

## Checked and clean
- Train-only fit of encoders/pipelines: `_pipeline_fit_transform.py:177-181`, `_pipeline_extensions.py:523-526`.
- `bruteforce.py:262-293`: full-sample CatBoostEncoder is the legacy path, default `leakage_free=True`, loud WARN otherwise.
- Calibrators/conformal/threshold: calib slice + calib!=test guard (`_calibration_models.py:421-428`); stability report uses train-side folds (`threshold_optimizer.py:60-110`).
- Outlier detector: fit on train, applied to val with collapse guard, test untouched (`_setup_helpers_outliers.py:217-262`).
- Holdout selectors (`hybrid_selector.py:301-312`, `ace.py:139-146`) use `holdout_indices(get_cv_policy(...))` first; `cv`-param selectors are rewired by `wire_selector_policy` (`cv_policy.py:281-289`).
- `competition/*` is quarantined and documented non-production.
- Composite discovery carves an honest holdout before screening and refuses train/val/test overlap (`discovery/_fit.py:403-406`).

## Verdict
No P1. Three P2: iid bootstrap CI on grouped/temporal test sets (LK-01, demonstrated 2.2x too narrow), unmasked temporal OOF in `post_calibrate_model` (LK-02, sklearn failure demonstrated), in-sample stacking fallback in the cross-target ensemble (LK-03).
Counts: P1 0, P2 3, P3 7.
Not worth fixing: LK-06 (disclosed, test stays honest); LK-08 alone (production wiring already causal); LK-04 only if OOF-derived widths matter.
