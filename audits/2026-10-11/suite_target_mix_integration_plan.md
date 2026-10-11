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
