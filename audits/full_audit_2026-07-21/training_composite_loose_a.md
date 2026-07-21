# training/composite loose modules A (stacking/ensembling core, model-card, provenance, monitoring) -- mlframe audit

## Scope

All 39 files in the assigned cluster were read in full (no file was too large/complex to review completely; the largest, `diagnostics.py` at 867 LOC, was read end-to-end).

- src/mlframe/training/composite/diagnostics.py (867 LOC)
- src/mlframe/training/composite/cache_store.py (699 LOC)
- src/mlframe/training/composite/cache.py (684 LOC)
- src/mlframe/training/composite/_value_report.py (536 LOC)
- src/mlframe/training/composite/__init__.py (474 LOC)
- src/mlframe/training/composite/provenance.py (447 LOC)
- src/mlframe/training/composite/classification_discovery.py (424 LOC)
- src/mlframe/training/composite/report.py (414 LOC)
- src/mlframe/training/composite/monitoring.py (399 LOC)
- src/mlframe/training/composite/multi_output.py (387 LOC)
- src/mlframe/training/composite/serving.py (379 LOC)
- src/mlframe/training/composite/autoconfig.py (353 LOC)
- src/mlframe/training/composite/survival.py (353 LOC)
- src/mlframe/training/composite/simplex.py (333 LOC)
- src/mlframe/training/composite/distributional.py (308 LOC)
- src/mlframe/training/composite/_winkler.py (305 LOC)
- src/mlframe/training/composite/classification.py (297 LOC)
- src/mlframe/training/composite/conformal_online.py (270 LOC)
- src/mlframe/training/composite/venn_abers.py (263 LOC)
- src/mlframe/training/composite/attribution.py (248 LOC)
- src/mlframe/training/composite/panel.py (241 LOC)
- src/mlframe/training/composite/post_shim.py (239 LOC)
- src/mlframe/training/composite/_pseudo_bma.py (230 LOC)
- src/mlframe/training/composite/stacking_multi_stage.py (228 LOC)
- src/mlframe/training/composite/_estimator_dispatch.py (219 LOC)
- src/mlframe/training/composite/chained_window_forecast.py (216 LOC)
- src/mlframe/training/composite/additive_decomposition.py (207 LOC)
- src/mlframe/training/composite/conformal_classification.py (204 LOC)
- src/mlframe/training/composite/feature_subset_bagging.py (186 LOC)
- src/mlframe/training/composite/dual_direction.py (164 LOC)
- src/mlframe/training/composite/gated_outlier.py (158 LOC)
- src/mlframe/training/composite/per_group_router.py (150 LOC)
- src/mlframe/training/composite/regime_split_ensemble.py (145 LOC)
- src/mlframe/training/composite/multitask_auxiliary_loss.py (143 LOC)
- src/mlframe/training/composite/hpo_ensembling.py (135 LOC)
- src/mlframe/training/composite/row_level_average_importance.py (114 LOC)
- src/mlframe/training/composite/_profile_pipeline.py (104 LOC)
- src/mlframe/training/composite/_booster_margin.py (69 LOC)
- src/mlframe/training/composite/_hpo_metrics.py (27 LOC)

**Total files reviewed: 39. Total LOC reviewed: 11619.**

Test-coverage cross-checks were done with `find`/`grep` against `tests/` (excluding the two audit-excluded MRMR/SHAP mirror trees, which this cluster never imports).

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | silent-failure | monitoring.py:78-84, 108-130, 299-318 | Drift-monitor PSI/KS signals silently report "no drift" when the new batch's base column or predictions are empty/all-non-finite, instead of alerting on missing/broken data. |
| F2 | P1 | silent-failure | _booster_margin.py:39-58 | Broad `except Exception: pass` around each booster family's raw-margin call swallows a genuine error from a *matched* booster and re-surfaces it as a misleading "no raw-margin path" `NotImplementedError`. |
| F3 | P1 | correctness / dead-config | multitask_auxiliary_loss.py:55,62,109-128; additive_decomposition.py:70,79,161-180 | `batch_size` is a documented, stored constructor parameter that is never read in `fit()` -- training is always full-batch regardless of the value passed, silently ignoring a caller's mini-batch request. |
| F4 | P1 | memory-discipline | stacking_multi_stage.py:210-214, 220-225 | `_concat_meta`'s pandas branch does a full `X.copy()` on every `predict()` call just to append a handful of meta-feature columns, violating the project's "never copy a large dataframe" convention. |
| F5 | P2 | code-quality | classification.py:207-211 | `decision_function`'s `NotFittedError` message reads "predict called before fit" instead of naming `decision_function`, misleading debugging. |
| F6 | P2 | architecture / duplication | classification.py:236-269; diagnostics.py:692-733 | `CompositeClassificationEstimator.calibration_report` and `diagnostics._bin_top_label_calibration` independently reimplement the same top-label reliability binning; the latter's docstring claims to "mirror" the former, but the two are not actually shared code and can silently drift. |
| F7 | P2 | perf / wasted-work | per_group_router.py:136-147 | `predict()` unconditionally runs the global fallback model's `predict()` over the FULL `X` before overwriting most rows with per-group predictions, wasting inference work whenever most groups have their own submodel. |
| F8 | P2 | docs-contract | survival.py:321-353 | `predict()`'s docstring headline claims "Predict the MEDIAN survival time," but for `censoring="aware"` the implementation (`_predict_aware_resid_log`) explicitly disclaims being a calibrated median. |
| F9 | P2 | docs / cosmetic | autoconfig.py:202-228 | When `skew > _SKEW_TRANSFORM_MIN` but the candidate transforms are already in the default set, `added_transforms` is empty yet `rationale["transforms"]` is still emitted describing transforms as "added," while `suggested["transforms"]` is never actually set. |
| F10 | P2 | silent-numeric-substitution | serving.py:326-331, 363-366 | `load_serving_spec`'s fallback constant silently substitutes `0.0` when `y_train_median` is itself non-finite (a training set with no finite target), rather than propagating NaN or raising; parity with the live estimator's own analogous fallback (in `estimator.py`, out of this cluster's scope) could not be verified. |
| F11 | P2 | edge-case | chained_window_forecast.py:200 | `diagnose_error_accumulation`'s `growth_ratio = chain_mse / chain_mse[0]` has no zero-guard; a perfect-fit position 0 (`chain_mse[0]==0`) produces `inf`/`nan` growth ratios with no handling. |
| F12 | P2 | edge-case / validation-gap | additive_decomposition.py:126-151, 182-191 | `component_names=()` is never validated; `fit()`/`predict()` crash with a cryptic torch `TypeError` (`sum([])` yields a Python `int`, which has no `.numpy()`/tensor ops) instead of a clear `ValueError`. |
| F13 | P2 | docs-drift | _profile_pipeline.py:1-5, 56-104 | The module docstring documents a `--full` CLI flag that is never parsed or used anywhere in the script. |
| F14 | P2 | docs / perf-claim | venn_abers.py:86-155 | The docstring for `_ivap_saddle_njit` describes the kernel as "near-linear," but the explicit `for lo in range(i, -1, -1)` scan inside the outer loop is O(g) per bin (O(g^2) total for `g` unique calibration scores), not near-linear in `g`. |
| F15 | P2 | edge-case | multi_output.py:241-243 | `_resolve_specs`'s "already declared a base" check uses `specs[k].get("base_column") or specs[k].get("base_columns")`, so an explicit (if unusual) empty-string `base_column` is treated as "not declared" and silently overridden by `base_columns_map`. |
| F16 | P2 | test-coverage | per_group_router.py (whole file) | No test file exists anywhere under `tests/` for `PerGroupCompositeRouter` (grep for `per_group_router` across `tests/` returns zero hits). |
| F17 | P2 | test-coverage | row_level_average_importance.py (whole file) | `extract_model_importance` / `compute_row_level_feature_importance_oof` / `compute_row_level_feature_importance_single_model` are never referenced by any test (the sibling `test_biz_val_row_level_average.py` tests only the parent aggregation function). |
| F18 | P2 | test-coverage | _booster_margin.py (whole file) | No dedicated test file exercises `inner_raw_margin`'s family dispatch or its error path directly (only indirectly via the classification/GLM wrappers' own tests, which would not exercise F2's swallowed-error scenario). |

### Details

**F1 -- monitoring.py.** `_bin_fractions` (line 78-84) returns the uniform reference distribution when the new batch has zero finite values (`v.size==0`), and `_ks_statistic` (line 118-120) returns `0.0` in the same case. Concretely: if an upstream feed starts sending all-NaN for a base column (a broken pipeline, a renamed/typo'd source column that resolves to nulls), `CompositeDriftMonitor.monitor()`'s `base_psi[col]` / `base_ks[col]` signals both read as "perfectly matches the reference" (PSI=0, KS=0) instead of flagging the missing-data condition -- the exact opposite of the intended "catch drift" purpose. The same applies to `prediction_psi` if `y_hat` for the batch collapses to all-NaN (e.g. every row hit a domain-violation fallback). Fix direction: treat an empty/near-empty finite sample as its own alert condition (e.g. a `"base_missing[col]"` signal) rather than degrading silently into a "no drift" PSI/KS reading.

**F2 -- _booster_margin.py.** Each of the three family-dispatch blocks (LightGBM/XGBoost/CatBoost) wraps BOTH the `isinstance` check and the actual `.predict(...)` call inside one `try/except Exception: pass`. If `model` genuinely is e.g. an `LGBMClassifier` but its `.predict(X, raw_score=True)` call raises for a real reason (model not fitted, `X` has the wrong number of columns, a version-mismatch pickle), that exception is silently discarded, `out` stays `None`, and the function falls through to the XGBoost/CatBoost `isinstance` checks (which correctly fail) before finally raising `NotImplementedError: ... has no raw-margin path` -- an error message that is actively wrong (the inner DOES have a raw-margin path; it just failed to execute) and hides the real stack trace from the caller. Fix direction: split the `isinstance` check out of the `try` so only the import itself is guarded, and let a real `.predict()` failure propagate (or re-raise with context) instead of being absorbed by the same `except`.

**F3 -- multitask_auxiliary_loss.py / additive_decomposition.py.** Both `__init__` signatures declare `batch_size: Optional[int] = None`, store it as `self.batch_size`, and the docstring explicitly frames it as "Training configuration (full-batch Adam by default, `batch_size=None`)" -- wording that implies a non-`None` value changes behaviour. `fit()` in both classes never reads `self.batch_size` anywhere; the training loop is unconditionally `hidden = self.trunk_(X_t)` over the FULL `X_t` every epoch. A caller passing `batch_size=64` (e.g. because the dataset is too large to fit a full-batch forward/backward pass in GPU/CPU memory, or because they want SGD-style regularization) gets no error and no different behaviour -- just silent full-batch training. Fix direction: either implement mini-batch iteration honoring `batch_size`, or remove the parameter (and its docstring claim) until it is implemented.

**F4 -- stacking_multi_stage.py.** `_concat_meta`'s pandas branch (`out = X.copy(); ...; return out`) is called from `predict()` on every inference call. Per this repo's own stated convention ("frames can be 100+ GB -- never `.copy()`/`.clone()` ... to work around a bug; mutate-and-restore or use a view"), this doubles peak memory at predict time just to attach a small number of meta-feature columns. Compare `dual_direction.py`'s analogous `X.copy(deep=False)` (a cheap shallow copy that shares data buffers) for the same "add one column to X" need -- that's the pattern this file should use instead of a full `.copy()`.

**F5 -- classification.py.** `decision_function` raises `NotFittedError("CompositeClassificationEstimator.predict called before fit.")` when `estimator_` is missing, but the method that actually failed is `decision_function` (called directly, or transitively via `predict_proba`/`predict`). A user calling `est.decision_function(X)` on an unfitted estimator sees a message naming the wrong method, which is confusing when debugging a NotFittedError raised from a stack a few frames removed from `predict()`.

**F6 -- classification.py / diagnostics.py.** `diagnostics._bin_top_label_calibration`'s docstring says it mirrors `CompositeClassificationEstimator.calibration_report` "so the standalone plotter produces the identical curve," but the two functions are two independent implementations of equal-width top-label binning (different variable names, different digitize/searchsorted mechanics, slightly different `pred` derivation via `classes[np.argmax(...)]` vs `classes[argmax]` with a class-count guard). A future change to one binning scheme (e.g. a fix to a tie-breaking edge case) will not automatically propagate to the other, silently reintroducing a curve mismatch the docstring promises does not exist.

**F7 -- per_group_router.py.** In `predict()`, `preds = np.asarray(self.global_estimator_.predict(...))` runs on the entire `X` unconditionally, and only afterward do per-group predictions overwrite `preds[mask]` for each routed group. When most groups have their own fitted submodel (the common/intended case), the bulk of that global-model inference is wasted work whose output is immediately discarded. This is the exact "caller uses only part of a kernel's output" pattern the project's own review conventions flag. Fix direction: compute the global prediction only for rows whose group has no submodel (the `unseen`/fallback rows), mirroring the pattern already used correctly in `regime_split_ensemble.py`'s `"route"` combine mode.

**F8 -- survival.py.** `CompositeSurvivalEstimator.predict()`'s docstring says (unconditionally) "Predict the MEDIAN survival time," but `_predict_aware_resid_log` (used whenever `censoring_mode_ == "aware"`, i.e. whenever scikit-survival is available and `censoring` isn't forced to `"observed_only"`) explicitly documents itself as preserving only the C-index RANKING, "without claiming a calibrated median." A caller reading only the public `predict()` docstring would reasonably believe the aware-mode output is a calibrated median survival time when it is not.

**F9 -- autoconfig.py.** When `skew > _SKEW_TRANSFORM_MIN` and both `signed_power_y`/`log_y` are already present in `CompositeTargetDiscoveryConfig().transforms` (so `added_transforms` stays `[]`), `rationale["transforms"]` is still set to a message like "... added tail-compressing y-transform(s) [] ..." while `suggested["transforms"]` is never populated (the `if added_transforms:` guard on line 227 is False). The rationale dict then describes an action that did not happen, which could mislead an operator inspecting `rationale` to understand why a config was chosen.

**F10 -- serving.py.** `fallback_const = float(_med) if np.isfinite(_med) else 0.0` (lines 330-331) means a fitted estimator whose `fitted_params_["y_train_median"]` is NaN (e.g. every training row for that composite happened to be invalid/dropped, an edge case `estimator.py`'s own `_y_train_clip_bounds` presumably has to handle too) silently degrades to predicting a hard `0.0` for every out-of-domain row at serve time, with no warning surfaced anywhere in `load_serving_spec`. Whether this matches `CompositeTargetEstimator.predict`'s own analogous fallback bit-for-bit could not be confirmed from this cluster alone since `estimator.py` is out of scope; if the live estimator instead returns NaN or raises in this case, the lightweight serving path silently diverges from "bit-identical to `CompositeTargetEstimator.predict`," which is the module's own stated contract.

**F11 -- chained_window_forecast.py.** `growth_ratio = chain_mse / chain_mse[0]` divides every position's MSE by position 0's MSE with no protection against `chain_mse[0] == 0`. A backtest window where the forecaster predicts the first position perfectly (MSE exactly 0, plausible on a synthetic/degenerate test fixture) makes every other `growth_ratio` entry `inf` or `nan`, and the subsequent `trustworthy_horizon` loop (`chain_mse[i] > accumulation_threshold * chain_mse[0]`) would then flag position 1 as untrustworthy purely from a `0 * threshold == 0` comparison against any nonzero MSE, regardless of how small.

**F12 -- additive_decomposition.py.** Neither `fit()` nor the `__init__`/`_validate_component_constraints` path validates that `component_names` is non-empty. With `component_names=()`, `component_preds={}` and `primary_pred = sum(component_preds.values())` evaluates to the Python `int` `0` (not a tensor), so `mse(primary_pred, y_primary_t)` raises deep inside PyTorch's functional dispatch with a message unrelated to the actual misconfiguration. A clear `ValueError` at construction time (mirroring the existing `_validate_component_constraints` pattern) would surface the real problem immediately.

**F13 -- _profile_pipeline.py.** The module-level docstring's usage line is `python -m mlframe.training.composite._profile_pipeline [n] [--full]`, but `main(argv)` only ever reads `argv[1]` as `n`; there is no code anywhere in the file that inspects `argv` for `--full` or branches on it. A maintainer following the documented usage would get no error and no different behaviour from passing `--full`.

**F14 -- venn_abers.py.** `_ivap_saddle_njit`'s docstring states "the kernel is near-linear" and that it replaces "the prior `O(grid)` sklearn refits" implying an overall near-O(g) cost. The actual loop nesting is `for i in range(g): for lo in range(i, -1, -1): for t in ...: (bounded by hull size)`. The middle `lo` loop alone contributes `sum_{i=0}^{g-1}(i+1) = O(g^2)` iterations regardless of how cheap the innermost hull-tangent scan is. This is still a large improvement over the previous O(grid * n log n) per-point sklearn refit (since `g` -- unique calibration SCORES -- is typically much smaller than `n` rows), but "near-linear" overstates the complexity and could mislead a future maintainer deciding whether the calibration set size needs a cap for very large `n_cal`.

**F15 -- multi_output.py.** `already = specs[k].get("base_column") or specs[k].get("base_columns")` in `_resolve_specs` treats any falsy value (including an explicit, deliberately-set empty string) as "not yet declared," so `base_columns_map[k]` would silently override it even when `column_specs` was a list with an explicit (if unusual) `base_column=""`. In practice this is a very narrow edge case (an empty-string base column name is not a realistic input), so it is flagged as low-severity rather than a functional concern.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| P1 | test-gap | monitoring.py | Add a regression test that a batch whose base column (or predictions) is entirely NaN/empty raises or emits an explicit "insufficient data" signal, pinning whichever fix is chosen for F1. |
| P2 | refactor | classification.py / diagnostics.py | Have `diagnostics._bin_top_label_calibration` call `CompositeClassificationEstimator.calibration_report`'s binning logic directly (or extract a single shared helper both call), closing the drift risk in F6. |
| P3 | perf | per_group_router.py | Only run the global fallback model on rows whose group has no submodel (mirror `regime_split_ensemble.py`'s `"route"` mode), addressing F7 with no correctness change. |
| P4 | memory | stacking_multi_stage.py | Swap `_concat_meta`'s pandas `X.copy()` for a shallow copy (`X.copy(deep=False)`, as `dual_direction.py` already does) before assigning new columns. |
| P5 | test-gap | per_group_router.py | Add a `test_biz_val_per_group_router.py` and a basic unit test -- currently zero test files reference this class at all. |
| P6 | test-gap | row_level_average_importance.py | Add unit tests for `extract_model_importance` / `compute_row_level_feature_importance_oof` / `compute_row_level_feature_importance_single_model` directly (currently only the unrelated parent aggregation function is tested). |
| P7 | feature | multitask_auxiliary_loss.py / additive_decomposition.py | Either implement real mini-batch training honoring `batch_size` (useful for the same reason the docstring already anticipates it), or remove the unused parameter and its misleading docstring line. |
| P8 | validation | additive_decomposition.py | Validate `len(component_names) >= 1` in `__init__`/`fit`, raising a clear `ValueError` instead of a deep torch `TypeError`. |
| P9 | docs | serving.py | Document (or actively test) what `load_serving_spec` does when `y_train_median` is itself NaN, and confirm/enforce parity with `CompositeTargetEstimator.predict`'s own analogous fallback. |
| P10 | docs | _profile_pipeline.py | Either implement the documented `--full` flag or remove it from the usage line. |

## Coverage notes

- `estimator.py`, `transforms/`, `discovery/`, `ensemble/`, `spec.py`, and every other module this cluster imports FROM but does not itself define, were read only enough to understand call signatures/contracts (per the task's read-for-context allowance); their internals were not audited since they belong to other clusters. Two findings (F8, F10) explicitly note where a full verdict would require reading `estimator.py`, which is out of this cluster's scope.
- The two explicitly excluded trees (`feature_selection/filters/**`, `feature_selection/shap_proxied_fs/**`, and their test mirrors) were not touched at all, per instructions; nothing in this cluster imports from them.
- `venn_abers.py`'s `_ivap_saddle_njit` correctness (not just its complexity, flagged as F14) was not independently re-derived from first principles -- the module's own comments claim it was validated to ~1e-16 against a per-point sklearn `IsotonicRegression` refit, which this audit did not re-run (no code execution was performed, per the read-only mandate).
- No pytest run was performed (read-only mandate); all test-coverage claims (F16-F18, P5-P6) are `find`/`grep`-based absence checks against `tests/`, not confirmed-failing test runs.
