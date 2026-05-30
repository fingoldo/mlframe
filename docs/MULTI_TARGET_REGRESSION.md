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
  `_configs_base.py`. New helpers `is_multi_target_regression` and
  `is_any_regression` expose the predicate. `is_multi_output` now
  includes MTR alongside QR and multiclass / multilabel.

- **Suite-side dispatch** — ⏳ PENDING. The training suite (`trainer.py`,
  `_helpers_training_configs.py`, `_phase_train_one_target_*.py`, etc.)
  does NOT yet auto-detect MTR; it currently raises or degenerates
  per-model. Plan below.

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

Ordered by dependency. Each step is a separate PR.

### Phase 1 — Detection & target-type wiring

1. Auto-detect MTR in `trainer.py::_resolve_target_type`: when `y.ndim == 2 and y.shape[1] >= 2 and y.dtype.kind == "f"`, return `TargetTypes.MULTI_TARGET_REGRESSION`.
2. Confirm `train_mlframe_models_suite` propagates the new target type through `target_type_dispatch` to every per-model factory.
3. Add `MULTI_TARGET_REGRESSION` to `_strategies_base.py::is_supported_target_type` allow/skip-lists per strategy (any strategy NOT in the matrix above raises `NotImplementedError` with a clear message — same pattern LTR uses).

### Phase 2 — Per-strategy native paths

4. **CatBoostStrategy**: when `target_type.is_multi_target_regression`, override `loss_function="MultiRMSE"` in `get_native_kwargs`.
5. **XGBoostStrategy**: override `multi_strategy="multi_output_tree"` + `tree_method="hist"`.
6. **LinearStrategy / SklearnNativeStrategies**: no override needed; sklearn passes `(N, K)` y through.
7. **MLPStrategy**: no override needed; F-24 covers it.

### Phase 3 — Wrappers for non-native strategies

8. **LightGBMStrategy / HistGradientBoostingStrategy / NGBStrategy**: wrap the per-target regressor in `sklearn.multioutput.MultiOutputRegressor(base, n_jobs="auto")`. Mirror the `MultilabelDispatchConfig.wrapper_n_jobs` convention.
9. Add `MultiTargetDispatchConfig` (parallel to `MultilabelDispatchConfig`): `strategy="auto" | "wrapper" | "native"`, `n_jobs="auto"`, `force_native_xgb=False`.

### Phase 4 — Metrics

10. Add per-target metric variants: `MultiTargetMetricSpec(per_target_fn, aggregator="mean" | "max" | "min")` that loops a single-target metric over `K` columns. Default aggregator `mean` returns one scalar for ranking; `max` for worst-case monitoring.
11. Register in `metrics_registry`: `val_R2_mean`, `val_R2_min`, `val_RMSE_mean`, `val_RMSE_max` etc. — all min-direction except R2_mean / R2_min which are max-direction.

### Phase 5 — Reporting

12. `_reporting_regression.py`: add per-target tables (one row per K target) alongside the aggregated summary. Reuse the per-class table layout from multilabel reporting.
13. `feature_importance` per target: SHAP / built-in importances per target column.
14. `composite_target_*`: composite targets (linear residual / per-group residual) are 1-D; explicit `NotImplementedError` for MTR (sane default — composite needs per-target design).

### Phase 6 — Ensembling

15. `CT_ENSEMBLE`: the cross-target ensemble logic assumes single y per fit. Either:
    - **a)** Per-target ensembling: K independent CT_ENSEMBLE calls, stacked.
    - **b)** Joint multi-target ensemble: average per-model (N, K) preds, then per-target R² for the ensemble score.
    Default to (a); (b) as opt-in via `MultiTargetDispatchConfig.ensemble_mode`.
16. `composite_post_xt_ensemble`: similarly per-target, then stack.

### Phase 7 — TTR-on-y / scaling

17. `_ttr_eval_set_scaling`: sklearn `TransformedTargetRegressor(transformer=StandardScaler())` already supports `(N, K)` natively. Verify our `_TTRWithEvalSetScaling` subclass passes through. The bench-attempt-rejected note from base.py around output_activation skips MTR explicitly — good.

### Phase 8 — Dummy baselines

18. `dummy_baselines.py`: per-target `DummyRegressor(strategy="mean")` over K targets. Mean-per-target baseline R² is what users compare against.

### Phase 9 — Documentation & tests

19. Update `train_mlframe_models_suite` docstring with MTR example.
20. End-to-end biz_value test: synthetic K=3 MTR → suite trains all strategies → reports per-target metrics → CT_ENSEMBLE aggregates correctly.

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
