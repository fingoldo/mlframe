# Plan: carry the feature-base target mix into the training suite (draft v1, 2026-10-11)

Goal. When MRMR's feature-base mix finds a base `u` and the held-out gate accepts the offset `beta * u`, the final regression model of the training suite is fitted on `T = y - beta * u(X)` and predicts `f(X) + beta * u(X)`, for every regression target that uses MRMR with the mix on. Default ON only after the suite tests and the OOS check pass.

Evidence this rests on (audits/2026-10-11/fixed_cost_per_fit_profile.md). On synthetic OOS (6 seeds, n_train 6000, paired against plain MRMR with the mix off):
* HGB: two_dominant_weak_pair -46.6% (-4.6 se), monotone_control -50.2% (-5.2 se); derived_base, regime, noise_control within +-2% after the gate (derived_base was +10% before the gate).
* Ridge: the gain is from the mix features, not the offset (arm C equals arm B on every process).
* The offset is rejected by the gate on most processes, so the default must be a no-op where there is no held-out gain.

Current state (what exists).
* `filters/_target_offset_regressor.py`: `TargetOffsetRegressor(selector, regressor, offset_alpha=0.05, use_offset=True)`. Fits the selector, then the regressor on `selector.transform(X)` against `y - beta * u(X)`; held-out gate on even/odd training rows.
* `MRMR.fe_target_mix_provenance_`: per base `{base, beta, features, recipe}`; `recipe` is `{"raw": name}` or an EngineeredRecipe replayable from `X`.
* The training suite: `pre_pipeline = strategy.build_pipeline(...)` (`training/strategies/base.py`), then model fit/predict separately (`training/core/_phase_train_one_target_steps.py`). The model never sees raw `X`; `pre_pipeline.transform` returns features only.

Blocker. The offset needs raw `X` at predict time, but the suite hands the model preprocessed features only.

Proposed design (v1).
1. Base column. In `build_pipeline` for a regression target, after the selector, append a column `__mix_base__ = replay(recipe, X)` computed from the raw frame before the selector. The column is produced in the preprocessing, so it exists at fit and at predict.
2. Model wrapper. Wrap the model in `CompositeTargetEstimator(base_estimator=model, transform_name="linear_residual", base_column="__mix_base__")` (training/composite/estimator). It fits `y - (alpha + beta * base)` with OLS on train, inverts at predict. Gate: apply the held-out test (same as TargetOffsetRegressor) before keeping the wrapper; else the plain model.
3. Column hygiene. `__mix_base__` must not be a model feature. Either drop it inside the wrapper before the inner fit, or exclude it in the feature list of the strategy.
4. Selection. The base recipe is replayed from raw `X`, so MRMR's own selection does not have to keep the base; the column is appended after selection.
5. Off switch. `fe_target_mix_suite_enable` (default False until the gate is validated) in the suite config; the MRMR mix stays as configured.

Open questions for the critics.
* Is `linear_residual` (OLS alpha, beta on train) the right transform, or should it be a fixed-beta `diff` with the beta found by MRMR?
* Does the composite's own `mi_gain` gate (known broken, L-22) interfere when `transform_name` is set directly?
* Where in the suite do predictions get stored and consumed (metrics, OOF, ensembles, calibration), and does each consumer see the add-back?
* Classification targets: the offset is regression-only; confirm the suite never builds the wrapper for a classifier.

Tests required before default ON.
* Unit: the base column is replayed identically at fit and at predict (raw X order, no leakage).
* Unit: the wrapper never sees the base column as a feature.
* Unit: with the gate off, the predictions equal the plain pipeline's bit for bit.
* Suite: the existing regression tests of `training/` pass with the flag on and off.
* OOS: exp_target_offset_model with the suite path, same seeds; no regression on noise_control or heteroscedastic.

Risks.
* Every MRMR regression run changes its predictions when the flag is on (mitigation: the gate keeps the plain model where no offset is found).
* Prediction-time cost of the replay (small: one recipe per base).
* `__mix_base__` leaking into saved models or explanations (SHAP, feature importance) if not excluded.

## Critique round 1 (2026-10-11): what was accepted, what was not, and the revised design

Two critics read the plan against the code: one on statistics and ML validity, one on the training suite's call sites. Their findings are below with the disposition of each.

### Accepted and already fixed in code
* Selection and gate shared rows (critic ML, blocker). `TargetOffsetRegressor.fit` now fits the selector and takes beta on even rows only; the held-out gate fits and scores on odd rows, which neither the selector nor beta has seen. Cost: the selector gets half the training rows. Tests updated and passing (6 passed).

### Accepted, to be implemented
* Fixed-beta transform, not `linear_residual`. The measured estimator has no intercept refit; `linear_residual` refits alpha and beta by OLS and would not reproduce the OOS numbers. Use `T = y - beta*u` with beta from the gate-accepted MRMR slope.
* Do not add a `__mix_base__` column to the sklearn chain. `build_pipeline` runs the selector first; a column appended after the selector is imputed and scaled (`strategies/base.py:695-741`), and the recipe's source columns are not guaranteed to survive selection.
* Reuse the existing routing for a wrapper's base from the raw frame: `PrePipelinePredictShim` (`composite/post_shim.py:173-197`) and `CompositeTargetEstimator._routes_inner_input` (`composite/estimator/_estimator.py:156`). Predict routing currently falls back to `df` when required columns are missing (`core/_predict_composite_routing.py:38-56`), so `_required_columns` must include the base.
* Cache keys: add a digest of the mix provenance (recipe and beta) to `_compute_pipeline_cache_key` (`core/_phase_train_one_target.py:439-474`), to `compute_model_pipeline_cache_key` (`core/_phase_train_one_target_cache_helpers.py`), and to `composite_spec_digest` / `_composite_cache_mismatch` (`train_eval.py`). Without this a mix-on run reuses a mix-off cache.
* Reload: the `estimator_` unwrap (`train_eval.py:497-500`) evaluates on the T scale. A saved mix wrapper must keep its add-back for prediction and evaluate on the y scale.
* Consumers: every path that reads a regression prediction (metrics, calibration, OOF, ensembles, feature importances, SHAP) must add the offset back or be checked as not affected. Ensemble code special-cases `CompositeTargetEstimator` (`composite/ensemble/__init__.py:648`, `:880`), so a mix wrapper must not be mistaken for a composite spec.
* Missing base at predict: raise, or fall back to the plain model explicitly. No silent zero-fill (`_target_offset_regressor.py` `nan_to_num` on the base is to be replaced).
* Off switch: `fe_target_mix_suite_enable` does not exist yet; declare it next to `fe_target_mix_enable` (`fs_params/mrmr.py:575`) and thread it into the MRMR fit, not into `build_pipeline`.
* Regression-only: assert in code that the wrapper is never built for a classification target (not yet verified against `build_pipeline` call sites).
* Statistics: the OOS evidence rests on 6 seeds. Before default ON, run at least 20 paired seeds with multiplicity correction across bases and targets, and a null calibration (y independent of X) that shows the false-accept rate is at or below `offset_alpha`.

### Rejected, with the reason
* "noise_control accepts the offset 6/6 is a false positive." Rejected: noise_control is `y = 3*x1 + noise`, so `x1` is a true linear signal and the acceptance is correct. The -1.5% (-9.7 se) HGB gain there is small but real. The null calibration above is still required, because the true null has not been tested.
* "The plan's risk framing is wrong: the mix is not default on today." Partly accepted: `fe_target_mix_enable` already defaults to True (`fs_params/mrmr.py:575`), so today's MRMR regression runs already change predictions through the mix features (arm C). The real new change is the offset only; the plan's risk section is reworded accordingly, and the mix's own cost (3-10 s per fit at 10k) is recorded.

### Revised design (v2)
1. Keep `build_pipeline` step order unchanged; no base column in the sklearn chain.
2. In the suite, before `pre_pipeline` fit and predict, for each regression target with an accepted mix recipe, replay the recipe on raw X to obtain `u`; carry it under a reserved name excluded from features, schema fingerprint and metadata.
3. Route the wrapper through `_routes_inner_input` with the reserved base in `_required_columns`.
4. Fixed-beta transform `T = y - beta*u`, beta from the MRMR gate.
5. Held-out even/odd gate as in `TargetOffsetRegressor`; plain model kept when the gate rejects.
6. Digests of the mix provenance in the three cache keys above.
7. Reload keeps the wrapper and evaluates on y scale.
8. `fe_target_mix_suite_enable` default False, declared in the suite config.
9. Consumer audit recorded in this file before default ON.

### Tests required before default ON
* `u` replayed from raw X is identical at fit and predict; no leakage across the held-out split.
* The reserved base column is absent from `feature_names_in_`, the schema fingerprint and the metadata feature list.
* Gate off: predictions equal the plain pipeline bit for bit.
* A mix-on run after a mix-off run with identical inputs does not reuse the cache; a mix-off model is not accepted for a mix-on run.
* A saved and reloaded wrapper predicts on y scale and matches the in-memory prediction.
* The wrapper is never built for a classification target.
* Suite: existing `training/` regression tests pass with the flag on and off.
* OOS: at least 20 paired seeds through the suite path, multiplicity-adjusted; noise_control and heteroscedastic not worse; null calibration passes.


## Decision (2026-10-11): option C, the wrapper replays the base from the raw frame

Options weighed for keeping `__mix_base__` away from the selector:
* A. Move the base out of the frame before the selector and back after. Needs the base carried beside the pre-pipeline through every fit/predict signature. Rejected.
* B. New exclusion parameter on MRMR. MRMR has no exclude or forced-keep parameter; adding one touches `fs_params/mrmr.py`, the setstate defaults and the selection cache key, and the column stays in the frame. Rejected.
* C. No base column anywhere. The wrapper stores the recipe, replays it from the raw frame at fit and at predict, and declares the recipe's source columns in `_resolve_base_columns` so the routing keeps them. Chosen.

Verified in code for C:
* Predict: `composite_predict` / `composite_stage_frame` (`core/_predict_composite_routing.py:38-68`) pass the suite-stage raw frame to a wrapper with `_routes_inner_input`, and fall back to the raw `df` when the required columns are missing.
* Fit: `_wrap_one_composite` (`core/_phase_composite_wrapping.py:364-433`) wraps fitted entries with `filtered_train_df`, the raw training split, so the wrapper can compute the same base at fit time.
* Target: the composite targets are built upstream and stored in `target_by_type` (T-scale), then wrapped post hoc with `from_fitted`. The mix offset must follow the same pattern: the model trains on `T = y - beta*u(X_train)` where `u` is replayed from the raw training frame, and the wrapper adds `beta*u` back at predict.

Remaining work for C, in order:
1. Target build: a mix-offset target entry in `target_by_type`, `u` replayed from the raw training split, computed once per fit (not per model).
2. Wrapper class over a fitted model with the recipe (not a column), `_routes_inner_input = True`, `_resolve_base_columns` = the recipe's raw source columns.
3. Held-out gate as in `TargetOffsetRegressor`, applied before the entry is kept.
4. Cache-key and composite-spec digests carry the recipe and beta.
5. Save/load keeps the wrapper; metrics on the y scale.
6. Regression-only guard; suite tests with the flag on and off; OOS on 20+ seeds.


## Status (2026-10-11, after the loop ticks)

Implemented and committed (local, not pushed):
* Mix-offset target built from the training split only (`build_mix_offset_target`), companion `<name>__mixoff` added by the discovery hook (`_add_mix_offset_targets`).
* Suite wiring behind the explicit flag `mix_offset_enabled` (default False): `_mix_offset_inputs_for_discovery` assembles the full frame from the split frames and refuses unless the splits partition the table.
* Predict paths (`_predict_main_suite.py`, `_predict_main_from_models.py`) add the offset back for companion models, using the recorded recipe and beta; helper `mix_offset_record` in the feature-selection module.
* End-to-end: suite trains `target__mixoff` and records it (test passes); predicting through the suite returns the companion on the y scale (RMSE against y below 1.0 on the fixture; test passes).

Still open before default ON: caching digests (plan v2 step 6), reload of a saved wrapper (step 7), the regression-only guard on every build path, and the 20-seed OOS with multiplicity correction (plan v2 step 8).
