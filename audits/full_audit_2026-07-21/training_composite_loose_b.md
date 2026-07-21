# training/composite loose modules B (conformal/quantile/forecasting/GLM strategies) -- mlframe audit

## Scope

All 38 files were read in full (not sampled/skipped).

- src/mlframe/training/composite/conformal.py (770 LOC)
- src/mlframe/training/composite/hpo.py (750 LOC)
- src/mlframe/training/composite/provenance_formulas.py (690 LOC)
- src/mlframe/training/composite/qrf.py (615 LOC)
- src/mlframe/training/composite/_moe_gate.py (470 LOC)
- src/mlframe/training/composite/ranking.py (444 LOC)
- src/mlframe/training/composite/model_card.py (421 LOC)
- src/mlframe/training/composite/highlevel.py (409 LOC)
- src/mlframe/training/composite/quantile.py (391 LOC)
- src/mlframe/training/composite/_regime_headroom.py (383 LOC)
- src/mlframe/training/composite/sklearn_compat.py (371 LOC)
- src/mlframe/training/composite/extremes.py (353 LOC)
- src/mlframe/training/composite/bagging.py (350 LOC)
- src/mlframe/training/composite/streaming.py (327 LOC)
- src/mlframe/training/composite/glm.py (318 LOC)
- src/mlframe/training/composite/compare.py (301 LOC)
- src/mlframe/training/composite/_heteroscedastic.py (294 LOC)
- src/mlframe/training/composite/missing.py (272 LOC)
- src/mlframe/training/composite/orthogonal.py (261 LOC)
- src/mlframe/training/composite/grouped_block_stacking.py (249 LOC)
- src/mlframe/training/composite/cv.py (247 LOC)
- src/mlframe/training/composite/segmented_model_factory.py (239 LOC)
- src/mlframe/training/composite/meta.py (229 LOC)
- src/mlframe/training/composite/direct_multi_horizon.py (224 LOC)
- src/mlframe/training/composite/pseudo_labeling.py (219 LOC)
- src/mlframe/training/composite/suite_features.py (213 LOC)
- src/mlframe/training/composite/gated_regression_mixture.py (212 LOC)
- src/mlframe/training/composite/group_aggregate_macro.py (196 LOC)
- src/mlframe/training/composite/segment_routed.py (191 LOC)
- src/mlframe/training/composite/count_weighted_blend.py (179 LOC)
- src/mlframe/training/composite/row_level_average.py (157 LOC)
- src/mlframe/training/composite/conformal_glm.py (149 LOC)
- src/mlframe/training/composite/multi_output_conformal.py (143 LOC)
- src/mlframe/training/composite/long_format_gbm.py (139 LOC)
- src/mlframe/training/composite/transform_priority.py (117 LOC)
- src/mlframe/training/composite/calendar_anomaly.py (114 LOC)
- src/mlframe/training/composite/monitoring_rediscovery.py (113 LOC)
- src/mlframe/training/composite/spec.py (79 LOC)

Total files reviewed: 38. Total LOC reviewed: 11599 (sum of the per-file `wc -l` counts above).

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness | calendar_anomaly.py:73-82 | `corrected` formula divides low-side anomalies by their deviation ratio instead of multiplying, pushing low outliers further from baseline instead of correcting them toward it. |
| F2 | P1 | correctness/edge-case | quantile.py:296,344-364 | `CompositeQuantileEstimator.fit` accepts an asymmetric `quantiles` grid on a decreasing-inverse transform without validation, guaranteeing a `ValueError` at `predict_quantile()` even when called with the estimator's own fitted grid. |
| F3 | P1 | correctness | grouped_block_stacking.py:173,205,209 | `fit(..., sample_weight=...)` is silently dropped for every per-group submodel fit (`composite_oof_predictions` / `full_model.fit`); only the meta-model receives the weights. |
| F4 | P1 | correctness | gated_regression_mixture.py:174-191 | `predict()` silently returns `0.0` (the `np.zeros` init value) for rows routed to a branch that had zero rows at fit time and fall outside the soft-routing band, instead of falling back to the other branch or raising. |
| F5 | P1 | correctness/design | pseudo_labeling.py:196 | For `task="classification"`, pseudo-labels are hardened to `{0.0, 1.0}` via a 0.5 threshold before being fed back into the final fit, directly contradicting the module's stated "soft (probability/regression) labels ... rather than hard labels" design and discarding the calibration signal the technique is built around. |
| F6 | P1 | ML-best-practice | _heteroscedastic.py:162-198 | The predictive-variance head and its global calibration factor `sigma_calibration_` are fit and calibrated entirely IN-SAMPLE (same rows used to fit the mean model), unlike the rest of the package's conformal machinery which enforces a held-out calibration split; predictive intervals are likely systematically too narrow on genuinely new data, with no caller-facing warning. |
| F7 | P1 | performance | ranking.py:103,151,167,292,399,413 | `_within_group_residual`, `_residual_to_gains`, `_ndcg_at_k`, `_build_pairs`, `predict`, and `rank` all loop `for gid in np.unique(group): mask = group == gid`, an O(n * n_groups) pattern; the file itself demonstrates the O(n) sort+segment fix in `_group_boundaries` but never applies it to these hot paths, a real perf cliff on realistic learning-to-rank datasets with thousands of query groups. |
| F8 | P2 | reproducibility | conformal.py:88 | `_conformal_internal_split` hardcodes `np.random.default_rng(0)` for the sigma_hat fit/calibrate split; `calibrate_conformal` exposes no `random_state`, so every call gets the identical split regardless of caller intent. |
| F9 | P2 | reproducibility | ranking.py:291 | `_build_pairs` hardcodes `np.random.default_rng(0)` for within-group pair subsampling on wide groups; `CompositeRankEstimator` has no `random_state` constructor param to vary or reproduce it independently. |
| F10 | P2 | correctness/edge-case | conformal.py:469,592-601 | `calibrate_conformal_mondrian` factorizes group labels with `use_na_sentinel=False` (keeping NaN as its own certified group), but `predict_interval_mondrian` re-factorizes predict-time labels in a SEPARATE call; the resulting NaN key is a different float object, so `lab in per_group` always misses even a certified NaN group, silently routing it to the (safe but unintended) OOD fallback and firing the "unseen groups" warning every time. |
| F11 | P2 | silent-failure | glm.py:142-160 | `_set_inner_objective` wraps the whole LightGBM-objective-coercion step in a bare `except Exception: ... pass` (debug-only log); if `set_params` fails for any reason the inner silently keeps a mismatched deviance objective with no warning to the caller. |
| F12 | P2 | ML-best-practice | missing.py:213-220 | `missing_offset_` (the MNAR correction) is estimated by calling `self.composite_.predict(X_imp)` on the SAME rows the inner composite was just fit on, an in-sample residual reuse that can understate the true offset needed on genuinely new missing-base rows. |
| F13 | P2 | docs/consistency | provenance_formulas.py:618 | `linear_residual_multi_robust` is aliased directly to `_f_linear_residual_multi` in `_TRANSFORM_FORMULA_BUILDERS` with no dedicated delegating function (unlike the single-base `_f_linear_residual_robust`, which exists purely to document the alias); the rendered formula text never mentions the robust fit, unlike every other "robust" transform in the file. |
| F14 | P2 | API consistency | quantile.py:366-377 | `CompositeQuantileEstimator.predict()` (median convenience) picks the head "nearest to 0.5" but does not apply the `_inverse_decreasing_` complementary-head lookup that `predict_quantile()` uses; on a decreasing-inverse transform with a quantile grid that omits 0.5 exactly, the returned "median" can be mislabeled (usually numerically close since both candidate heads are near 0.5, but inconsistent with the rest of the class's handling). |

### F1 -- calendar_anomaly.py `corrected` formula wrong for low-side anomalies (P0)

`detect_calendar_anomalies` computes `deviation_ratio = hi/lo` where `hi = max(y, baseline)`, `lo = min(y, baseline)` (line 74-76), then corrects every flagged row via `corrected[flagged] = y[flagged] / deviation_ratio[flagged]` (line 82). For a HIGH spike (`y > baseline`), `ratio = y/baseline`, so `y/ratio = baseline` -- correct. For a LOW anomaly (`y < baseline`), `ratio = baseline/y`, so `y/ratio = y / (baseline/y) = y^2/baseline`, which is SMALLER than `y` itself (since `y/baseline < 1`), not equal to `baseline`. Concretely: `baseline=100, y=10` (a 10x low dip) gives `ratio=10`, `corrected = 10/10 = 1` -- the "corrected" series pulls the outlier value from 10 down to 1, i.e. further AWAY from the baseline it was supposed to be pulled toward. Any caller using `corrected` (or `apply_calendar_anomaly_flag`'s second return value) as a de-spiked feature will silently get a badly wrong value on every low-side calendar anomaly (e.g. a holiday closure with near-zero sales). Fix direction: branch on whether `y > baseline` or `y < baseline` and multiply vs. divide accordingly (or equivalently `corrected = baseline * (y / baseline) ** 0` is not it either -- the simplest fix is `corrected = np.where(y >= baseline, y / deviation_ratio, y * deviation_ratio)`).

### F2 -- quantile.py asymmetric grid + decreasing-inverse transform crashes at predict (P1)

`_transform_inverse_decreasing` correctly detects transforms like `reciprocal_residual` whose inverse flips quantile order, and `predict_quantile` looks up the COMPLEMENTARY head (`1 - q`) for such transforms. But `fit()` only ever fits heads at the literal `quantiles` values passed in -- it never checks that the grid is symmetric around 0.5, nor fits/validates the complements. If a caller configures e.g. `quantiles=(0.1, 0.3, 0.5)` with `transform_name="reciprocal_residual"`, `fit()` succeeds, but any later `predict_quantile()` call -- INCLUDING the no-argument default that requests the estimator's OWN fitted grid `self.quantiles_` -- raises `ValueError` from `_lookup_head`, because heads at `0.9` / `0.7` were never fitted. Fix direction: at fit time, either (a) reject/warn on a non-self-complementary grid when the transform is decreasing, or (b) auto-augment the fitted grid with each level's complement.

### F3 -- grouped_block_stacking.py drops sample_weight for per-group submodels (P1)

`GroupedBlockStacker.fit(self, X, y, sample_weight=None)` accepts `sample_weight` and forwards it only to `self.meta_model_.fit(meta_X, y_arr, **fit_kwargs)` (line 223-224). The per-group OOF call `composite_oof_predictions(submodel_factory, X_group_valid, y_arr[valid_mask], n_splits=..., random_state=...)` (line 205) and the group's full-data refit `full_model.fit(X_group_valid, y_arr[valid_mask])` (line 209) never see `sample_weight` at all, not even sliced to `valid_mask`. A caller passing per-row weights (e.g. to downweight noisy sensors) gets every per-group submodel silently trained unweighted while only the meta-blender honors the weights -- a real weighted-fit-goes-unweighted regression per the checklist's sample-weight-propagation pattern. Fix direction: thread `sample_weight[idx]` / `sample_weight[valid_mask]` into both calls (guarded on whether the submodel factory's `fit` accepts the kwarg, as `segment_routed.py` and `count_weighted_blend.py` already do elsewhere in this cluster).

### F4 -- gated_regression_mixture.py silently predicts 0.0 for a branch with no fitted model (P1)

`fit()` logs a warning and simply `continue`s when a branch (`"low"` or `"high"`) has zero routed rows (line 148-150), leaving that branch absent from `self.branch_models_`. `predict()` initializes `out = np.zeros(n, ...)` (line 175) and only fills `out[mask]` inside the `for branch in (_LOW, _HIGH)` loop when `branch in self.branch_models_` (line 189); rows that route to the missing branch and fall outside the (opt-in, default-off) soft-routing band are simply never written -- they silently return `0.0`. On an imbalanced `subpop_label` distribution (plausible: the source pattern is literally for rare-outlier routing) or an extreme `threshold`, one branch can easily see zero fit-time rows while still receiving predict-time rows, producing silently-wrong (zero) predictions with no error or warning at predict time. Fix direction: at predict, fall back to the other (fitted) branch's prediction, or to a stored global fallback, for rows whose branch has no model; raise clearly if neither branch has a model.

### F5 -- pseudo_labeling.py hardens classification pseudo-labels, contradicting the "soft labels" design (P1)

The module docstring's entire premise (lines 1-19) is "use soft (probability/regression) labels rather than hard labels". `_fold_ensemble_score` correctly returns a continuous `mean_pred` (a probability, via `predict_proba`), but `fit()`'s `soft_labels = np.where(mean_pred >= 0.5, 1.0, 0.0) if self.task == "classification" else mean_pred` (line 196) discards that continuous signal for classification and feeds back hard `{0,1}` labels to the final `fit`. This defeats the stated purpose of the class for every classification use (regression, the `else` branch, is unaffected and correctly soft). Fix direction: feed `mean_pred` itself as the pseudo-label and require/accept a regressor-style final model for classification (as ngboost/soft-label self-training literature does), or explicitly document + rename the parameter if hard labels are actually intended.

### F6 -- _heteroscedastic.py calibrates predictive variance entirely in-sample (P1)

`fit()` computes `t_hat_train = mean.estimator_.predict(X)` and `resid = t_target - t_hat_train` on the SAME `(X, y)` the mean composite was just fit on (lines 167-173), fits the variance head on those residuals, and then calibrates `sigma_calibration_` via `_fit_calibration(resid[finite], sigma_train[finite])` using those same in-sample residuals and the variance head's own in-sample predictions (line 190). Because the variance head is trained to predict exactly these residuals, `resid / sigma_train` is close to 1 in-sample essentially by construction, so the calibration factor cannot detect or correct the optimism from evaluating on training rows -- unlike `conformal.py` in the same cluster, which goes out of its way to require and document a held-out calibration split for exactly this reason. `predict_interval` / `predict_std` are consequently at risk of under-covering on genuinely new data, with nothing in the docstring warning callers to hold out a calibration split. Fix direction: support (or require) an OOF/held-out residual split for the variance-head fit and the calibration factor, mirroring `conformal.py`'s pattern.

### F7 -- ranking.py repeats O(n * n_groups) mask loops the file's own utility already fixes (P1)

`_group_boundaries` (line 79-91) demonstrates the O(n) sort+segment approach used for the lambdarank path, but `_within_group_residual` (line 94-112), `_residual_to_gains` (141-161), `_ndcg_at_k` (164-180), `_build_pairs` (281-313), `predict` (386-402), and `rank` (404-417) all instead loop `for gid in np.unique(group): m = group == gid`, rescanning the full `n`-length array once per unique group -- O(n * n_groups) total. The module's own cProfile note benchmarks only 200 groups x 20 items (4,000 rows); a realistic learning-to-rank corpus with thousands of query groups (a common production shape) would pay a real, avoidable quadratic-ish cost on every fit/predict call across six separate call sites. Fix direction: reuse `_group_boundaries`'s sort-and-slice pattern (or `pandas.factorize` + `np.bincount`/`reduceat`, as `conformal.py`'s Mondrian path and `_moe_gate.py` already do elsewhere in this cluster) in all six functions.

### F8 -- conformal.py hardcoded seed in the sigma_hat internal split (P2)

`_conformal_internal_split(n_cal, time_ordering=None)` uses `rng = np.random.default_rng(0)` (line 88) whenever `time_ordering` is falsy, and `calibrate_conformal` (the only caller) has no `random_state` parameter to override it. Every call with the same `n_cal` gets the identical fit/calibrate row split, regardless of which calibration set was passed. Not incorrect (the split is still a valid random half), but it means two different calibration runs (e.g. different bootstrap resamples of the same calibration pool) get correlated splits, and it silently prevents a caller from decorrelating repeated calibration draws. Fix direction: add an optional `random_state` parameter to `calibrate_conformal` / `_conformal_internal_split`.

### F9 -- ranking.py hardcoded seed for pair subsampling (P2)

`_build_pairs` uses `rng = np.random.default_rng(0)` (line 291) to subsample pairs on wide groups (`> _MAX_PAIRS_PER_GROUP`), but `CompositeRankEstimator.__init__` exposes no `random_state`. Every fit on a wide-group dataset samples the exact same subset of pairs deterministically, with no way for a caller to vary it (e.g. for bagging/seed-ensembling multiple rankers). Fix direction: add a `random_state` constructor parameter and thread it through.

### F10 -- conformal.py Mondrian NaN group labels never match at predict (P2)

`calibrate_conformal_mondrian` factorizes `groups_cal` with `pd.factorize(g, sort=True, use_na_sentinel=False)` (line 469), deliberately keeping a NaN label as its own certifiable group (per the docstring). `predict_interval_mondrian` independently re-factorizes the predict-time labels (`pd.factorize(g, sort=False, use_na_sentinel=False)`, line 588) and checks `if lab in per_group and lab not in uncertified` (line 593). Because the two factorize calls produce DIFFERENT `float('nan')` objects and `nan != nan`, `lab in per_group` is always `False` for a NaN group even when it WAS certified at calibration -- every predict-time NaN-group row falls through to the OOD-adaptive fallback and fires the "groups not seen at calibration" warning, even on the exact NaN group that was fit. This fails safe (a wider, more conservative band) rather than under-covering, so the practical harm is bounded, but it silently contradicts the documented "SEEN-and-CERTIFIED" routing contract for any group scheme that can legitimately produce a NaN/missing group label. Fix direction: normalise NaN group labels to a stable sentinel object (e.g. a fixed string `"__nan__"`) before building/using `_label_to_code` / `per_group`, as `_moe_gate.py`'s `_factorize` already does by routing NaN to code `-1` consistently.

### F11 -- glm.py silently swallows objective-coercion failures (P2)

`_set_inner_objective` (lines 142-160) wraps the entire LightGBM `isinstance` check + `model.set_params(...)` call in `except Exception as e: logger.debug(...); pass`. If `set_params` ever raises (e.g. a LightGBM version that renamed `tweedie_variance_power`, or a caller-supplied subclass that rejects the kwarg), the inner silently keeps whatever objective it already had -- possibly the wrong deviance for the requested `family` -- with no warning above `logger.debug`, and `fit()` proceeds to train a model whose gradient does not match the documented "matching deviance objective" contract. Fix direction: narrow the `except` to the specific failure modes expected (missing param names) and `logger.warning` (not `debug`) on any other exception, or re-raise.

### F12 -- missing.py in-sample offset estimation (P2)

`fit()`'s `missing_offset_` is computed as the mean gap between `y_arr[missing_mask]` and `self.composite_.predict(X_imp)[missing_mask]` (lines 213-220), where `X_imp`/`y_arr` are the exact rows `self.composite_` was just fit on. This is not leakage in the traditional sense (no future/test rows are touched), but it is an in-sample residual reuse: the composite's predictions on rows it was trained on are typically more accurate than on genuinely new missing-base rows, so `missing_offset_` risks understating the true MNAR correction needed at predict time on unseen data. Lower severity than F6 because the offset is a single scalar correction (not a full predictive band) and the module already documents a `max_missing_frac` safety valve. Fix direction: compute the offset from an internal K-fold OOF pass (mirroring `composite_oof_predictions` used elsewhere in this cluster) instead of the in-sample composite prediction.

### F13 / F14 -- minor documentation/consistency nits (P2)

See table; both are cosmetic (a formula string that doesn't mention "robust", and a median convenience method that skips a complementary-head lookup other methods in the same class apply) rather than functional bugs with meaningful blast radius.

## Proposals

| ID | Category | File | Summary |
|----|----------|------|---------|
| PR1 | test-coverage | grouped_block_stacking.py | No test currently exercises `sample_weight` end-to-end for `GroupedBlockStacker` (a weighted-fit regression test would have caught F3); add one asserting the per-group submodel actually receives the sliced weight (e.g. via a spy/mock `submodel_factory`). |
| PR2 | test-coverage | gated_regression_mixture.py | No test drives an extreme `threshold` / imbalanced `subpop_label` that leaves one branch with zero fit-time rows, so F4's silent-zero-prediction path is untested; add a regression test asserting `predict()` never silently returns the branch-missing fallback of 0.0 for a routed row. |
| PR3 | test-coverage | calendar_anomaly.py | The existing `test_biz_val_calendar_anomaly.py` apparently did not catch F1 (a low-side-anomaly correction sign error); add a unit test with a synthetic LOW-value spike (`y << baseline`) asserting `corrected` moves TOWARD `baseline`, not further away. |
| PR4 | ML-practice | _heteroscedastic.py | Offer an explicit held-out-residual path (an optional `X_cal`/`y_cal` pair, or an internal K-fold OOF pass for the variance head) so `predict_std`/`predict_interval` calibration is not entirely in-sample; document the current in-sample limitation prominently until then. |
| PR5 | refactor/perf | ranking.py | Replace the six `np.unique(group)` + boolean-mask loops with the sort-and-segment pattern `_group_boundaries` already implements in the same file, cutting fit/predict from O(n * n_groups) to O(n log n) for realistic group counts. |
| PR6 | API | conformal.py, ranking.py | Add an explicit `random_state` parameter to `calibrate_conformal` (for `_conformal_internal_split`) and to `CompositeRankEstimator` (for `_build_pairs`'s pair subsampling) instead of the current hardcoded seed-0 RNGs, so repeated/bagged calibration or ranking runs can be decorrelated when desired. |
| PR7 | robustness | conformal.py | Normalise NaN/missing group labels to a stable sentinel (matching `_moe_gate.py`'s `-1`-code convention) in both `calibrate_conformal_mondrian` and `predict_interval_mondrian` so a certified NaN group is actually reachable at predict time. |
| PR8 | quality | quantile.py | Validate at `fit()` time that a decreasing-inverse transform's `quantiles` grid is self-complementary (or auto-augment it), converting F2's deferred `predict_quantile()` crash into an immediate, actionable `fit()`-time error or a working estimator. |

## Coverage notes

- `cv.py`'s purge/embargo index arithmetic (`PurgedTimeSeriesSplit.split`) is intricate (interacting `purge`, `embargo`, `max_train_size`, and both the default and explicit `test_size` branches). I read the whole file and worked through the default-`test_size` case by hand (confirmed self-consistent for several `n`/`n_splits` combinations), but I did not exhaustively re-derive or numerically fuzz every combination of `purge > 0` + `embargo > 0` + `max_train_size` + explicit `test_size` against a reference implementation, so a narrow off-by-one in that interaction cannot be fully ruled out from static reading alone. No concrete failure scenario was found, so nothing is reported as a finding here -- flagging this as an area worth a targeted property-based/fuzz test rather than a static-review gap.
- Per the task's exclusion list, `src/mlframe/feature_selection/filters/**` and `src/mlframe/feature_selection/shap_proxied_fs/**` (and their test mirrors) were not opened at all, including where this cluster's files (e.g. `grouped_block_stacking.py`) reference DCD/MRMR symbols by name in comments -- only the referencing comment text was read, never the referenced package's internals.
- I did not execute any code (no pytest, no scripts) per the read-only audit constraint; all findings above are from static reading and manual trace-through of the documented formulas/contracts, not from a failing test run.
