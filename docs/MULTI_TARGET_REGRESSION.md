# Multi-Target Regression in mlframe

This document describes the design and rollout plan for
`MULTI_TARGET_REGRESSION` support: K independent continuous targets
trained jointly through a shared model trunk.

Target shape: `y` of shape `(N, K)` float, `K >= 2`.
Output shape: `(N, K)` float — column `k` is the point prediction for
target `k`.

## Status (2026-05-31)

- **MLP estimator** — ✅ **Native, landed in commit `2d300944`**.
  `PytorchLightningRegressor` auto-detects `(N, K>=2)` float `y` in
  `_fit_common` and routes `num_classes = K` to `generate_mlp`. MSE
  between `(N, K)` preds and `(N, K)` labels works without loss-shape
  gymnastics. The `_is_multi_target_regression` flag is set on the
  estimator. Verified at `R^2 >= 0.985` per column on K=3 synthetic data.

- **`TargetTypes.MULTI_TARGET_REGRESSION`** enum value — ✅ landed in
  commit `f03a86ef`. New helpers `is_multi_target_regression` and
  `is_any_regression` expose the predicate. `is_multi_output` now
  includes MTR alongside QR and multiclass / multilabel.

- **Strategy dispatch foundation** — ✅ **Landed in commit `d73bb763`** + smoke tests in `2241468f`.
  `ModelPipelineStrategy` gained:
  * `supports_native_multi_target: bool` (default False)
  * `get_multi_target_objective_kwargs() -> dict` (default `{}`)
  * `wrap_multi_target(estimator)` — identity for native, MultiOutputRegressor wrap for non-native
  Per-strategy overrides:
  * **CatBoostStrategy**: native, returns `{"loss_function": "MultiRMSE"}`
  * **XGBoostStrategy**: native, returns `{"multi_strategy": "multi_output_tree", "tree_method": "hist"}` (XGBoost ≥ 2.0)
  * **NeuralNetStrategy**: native (MLP-side already done in F-24)
  * **LinearModelStrategy**: native (sklearn handles `(N, K)` natively)
  * **TreeModelStrategy (LightGBM) / HGBStrategy**: non-native → `MultiOutputRegressor` wrap
  All 5 backends fit + predict `(N, K)` correctly per `test_multi_target_regression_smoke.py`.

- **Auto-route helper** — ✅ Landed in `d73bb763`.
  New `multilabel_strategy="multi_target_regression"` option on
  `CompositeTargetDiscoveryConfig`: instead of expanding `(N, K)`
  regression targets to K independent 1-D targets (the legacy
  `"per_target"` default), keeps them joint under
  `TargetTypes.MULTI_TARGET_REGRESSION`. Pick this for correlated
  targets that benefit from a shared trunk / boosting ensemble.

- **Per-target body wiring** — ✅ **Landed in commit `d48245de` (D1)**.
  `_phase_train_one_target_body.py` around line 651: when target_type
  is MULTI_TARGET_REGRESSION, the cloned model has
  `get_multi_target_objective_kwargs()` injected via `set_params`
  (CatBoost `loss_function="MultiRMSE"`, XGBoost
  `multi_strategy="multi_output_tree"`), then non-native strategies
  are wrapped via `wrap_multi_target()` (LightGBM / HGB →
  MultiOutputRegressor). MLP / Linear / Ridge auto-handle (N, K) at
  fit-time so the block is a no-op for them. Fallback to setattr if
  set_params rejects a kwarg.

- **Metrics registry** — ✅ **Landed in commit `d48245de` (D3)**.
  Seven MTR metrics registered: `rmse_macro` / `rmse_micro` /
  `rmse_max` / `mae_macro` / `mae_max` (lower-is-better), `r2_macro`
  / `r2_min` (higher-is-better). macro = mean per-target column,
  micro = pooled across (N*K), max/min = worst-case per-target
  (catches degenerate target columns the mean masks).
  `metric_name_higher_is_better()` resolves these via the
  registry-fallback path automatically.

- **Regression reporting** — ✅ **Fully landed in commits `d48245de` (D4) + `86504f08` (E1)**.
  `report_regression_model_perf` detects (N, K≥2) targets/preds and:
  * stamps aggregated MTR metrics (rmse_macro / r2_macro / ...)
    into the metrics dict via `iter_extra_metrics(MULTI_TARGET_REGRESSION, ...)`
  * when `plot_outputs` + `plot_file` are set, renders ONE chart file
    per target column via the existing `build_regression_panel_spec`
    → `render_and_save` pipeline. Output names:
    `{plot_file_base}_target{k}` (the renderer appends the format
    extension from the DSL). Each per-target chart shows the same
    scatter + histogram + residual-audit overlay as the single-target
    report.
  Still skipped on the MTR path: fairness subgroup, MASE,
  prediction-envelope clip — all 1-D-only.

- **CT_ENSEMBLE for MTR** — ✅ **Per-column equal-mean ensemble landed in commit `86504f08` (E2)**.
  `_build_cross_target_ensemble_for_target` dispatcher routes MTR
  targets to a new `_build_mtr_per_column_ensemble` helper that:
  * collects components from `models[target_type][target_name]`
  * builds `MTRPerColumnEqualMeanEnsemble` (a thin wrapper that
    stacks K trained component models, averages their (N, K)
    predictions across the component axis)
  * appends the ensemble as a new entry with
    `ct_ensemble=True + mtr_ensemble=True + ensemble_strategy="per_column_equal_mean"`
    so downstream save / report layers treat it like any other
    ensemble.
  Single-component pools skip with INFO. Future PR: swap equal-weight
  averaging for per-column honest-OOF optimal weights without changing
  the public predict() contract.

- **Composite targets (`CompositeTargetEstimator` / discovery)** —
  ⏳ NATURALLY SKIPPED. Composite discovery at
  `_phase_composite_discovery.py:223` already filters to
  `TargetTypes.REGRESSION` only — MTR targets are not iterated, so
  no composite generation runs for them. Users mixing the auto-route
  with composite discovery on the same target name should NOT see
  duplicate work; the auto-route happens BEFORE discovery and moves
  the target out of the REGRESSION bucket.

## Per-strategy target dispatch (post-rollout)

| Strategy | Native MTR? | Mechanism | Notes |
|---|---|---|---|
| **CatBoost** | ✅ | `loss_function="MultiRMSE"` (or `"MultiRMSEWithMissingValues"`) | Single ensemble, native (N, K) output. Preferred path. |
| **XGBoost** | ✅ (≥2.0) | `multi_strategy="multi_output_tree"` + `tree_method="hist"` | Treats targets jointly; faster than wrapper for K small. |
| **LightGBM** | ❌ | `sklearn.multioutput.MultiOutputRegressor(LGBMRegressor)` | LGB has no native MTR; K independent fits. |
| **HistGradientBoosting** | ❌ | `MultiOutputRegressor` wrapper | sklearn HGB doesn't support MTR natively. |
| **Linear (Ridge / LinearRegression / Lasso)** | ✅ | sklearn handles `(N, K)` natively | `MultiTaskLasso` / `MultiTaskElasticNet` add joint L1 across targets. |
| **Random Forest / Extra Trees** | ✅ | sklearn handles `(N, K)` natively | Per-target MSE summed. |
| **MLP (ours)** | ✅ | `PytorchLightningRegressor` (F-24) | K output heads sharing trunk; MSE per column. |
| **Recurrent (LSTM / GRU / Transformer)** | ⏳ | TBD — likely K-head extension of `MLPHead` | Lower priority; separate session. |
| **NGB (NGBoost)** | ❌ | `MultiOutputRegressor` wrapper | NGBoost is single-output by design (probabilistic forecast). |

## Suite-integration roadmap

Dependency-ordered outline of the work to wire multi-target regression (MTR) end-to-end:

- **Detection & target-type wiring** — auto-detect MTR (`y.ndim == 2 and y.shape[1] >= 2 and y.dtype.kind == "f"`) into a `MULTI_TARGET_REGRESSION` target type, propagate it through the per-model dispatch, and add per-strategy allow/skip-lists (unsupported strategies raise a clear `NotImplementedError`, same pattern LTR uses).
- **Per-strategy native paths** — CatBoost `loss_function="MultiRMSE"`, XGBoost `multi_strategy="multi_output_tree"`; linear / sklearn-native / MLP pass `(N, K)` y through unchanged.
- **Wrappers for non-native strategies** — wrap LightGBM / HistGradientBoosting / NGB in `sklearn.multioutput.MultiOutputRegressor`, mirroring the `MultilabelDispatchConfig` n_jobs convention.
- **Metrics & reporting** — per-target metric variants (`R2_mean`, `R2_min`, `RMSE_mean`, `RMSE_max`, ...) and per-target reporting tables / feature importances. Composite targets stay 1-D (explicit `NotImplementedError` for MTR).
- **Ensembling, TTR/scaling, dummy baselines** — per-target ensembling stacked across K (joint multi-target average as an opt-in), verify `TransformedTargetRegressor` passes `(N, K)` through, per-target `DummyRegressor(strategy="mean")` baselines.
- **Docs & tests** — MTR example in the suite docstring plus an end-to-end biz_value test (synthetic K=3 → all strategies → per-target metrics → ensemble aggregation).

## Cross-cutting risks

- **Target naming**: `target_name` is currently a single string. For MTR with K=3 named ["price", "volume", "spread"], reporting must reference each by name. Pass `target_names: List[str]` alongside `target_name` or use a delimiter convention.
- **Per-target sample weights**: most strategies accept a single 1-D `sample_weight` of shape `(N,)`. Per-target weighting (e.g. K-column `(N, K)` weight matrix) is NOT supported by sklearn's `MultiOutputRegressor` or by native CatBoost MultiRMSE; document the limitation.
- **Mixed dtype targets**: MTR assumes all K targets are continuous (float). Mixed int + float (e.g. one regression + one classification) is NOT this target type — that's joint multi-task learning, a separate feature.

## References

- F-24 audit entry in `project_mlp_audit_progress.md`
- Commit `2d300944` (MLP-side native MTR)
- Sibling doc: `MULTI_OUTPUT.md` (multilabel + multiclass classification)
- sklearn `MultiOutputRegressor`: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
- CatBoost `MultiRMSE`: https://catboost.ai/en/docs/concepts/loss-functions-multiregression
- XGBoost multi-output trees: https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html
