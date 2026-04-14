# Changelog

## 2026-04-14 — CatBoost Text & Embedding Features + Memory Optimizations

### Added

- **`text_features` and `embedding_features` support**: CatBoost now receives `text_features` (free-text string columns) and `embedding_features` (list-of-float vector columns) via `fit()` params. Models that don't support them (Ridge, XGB, LGB, HGB, MLP, etc.) automatically have these columns dropped before training.
- **`FeatureTypesConfig`** Pydantic class in `configs.py`: `text_features`, `embedding_features`, `auto_detect_feature_types`, `cat_text_cardinality_threshold` (default 50).
- **Auto-detection of feature types**:
  - Embedding columns: auto-detected from `pl.List(pl.Float32)` / `pl.List(pl.Float64)` dtype.
  - Text vs categorical: string columns with `n_unique > cat_text_cardinality_threshold` → text; `<= threshold` → categorical. User-specified lists always take priority.
- **Feature-tier model grouping**: models sorted by `strategy.feature_tier()` — `(True, True)` (CatBoost) trains first with all columns, then text/embedding columns are dropped once per tier for remaining models. Tier DFs cached via `_build_tier_dfs()` using `.select()` (not `.drop()`).
- `supports_text_features` and `supports_embedding_features` properties on `ModelPipelineStrategy` (default `False`). `CatBoostStrategy` overrides both to `True`.
- `feature_tier()` method on `ModelPipelineStrategy` — returns `(supports_text, supports_embedding)` tuple for grouping.
- Mutual exclusivity validation: `text ∩ cat`, `emb ∩ cat`, `text ∩ emb` → `ValueError`.
- Pipeline exclusion: text and embedding columns excluded from encoding/scaling in `fit_and_transform_pipeline()`.
- CatBoost text columns auto-filled with `""` for nulls (CatBoost requirement).
- 18 CPU integration tests + 2 GPU tests in `TestTextAndEmbeddingFeatures`.

### Memory Optimizations (for 100GB+ DataFrames)

- **B1: Conditional clone** — `train_df.clone()` only when pipeline will modify categoricals (`skip_categorical_encoding=False`). Saves 100GB+ when all models are Polars-native.
- **B2: Aggressive cleanup** — post-pipeline Polars DFs released after pandas conversion when no longer needed.
- **B3: `prepare_polars_dataframe()` cache** — moved outside weight schema loop, called once per model instead of once per weight schema.
- **B4: `.select()` over `.drop()`** — tier column trimming uses `.select(cols_to_keep)` for better Polars optimization.
- **B5: Release Polars originals after tier transition** — pre-pipeline Polars DFs freed after all Polars-native models finish training.

### Changed

- Model training loop now sorts models by `feature_tier()` (most features first) instead of using the user-provided order.
- `select_target()` and `configure_training_params()` accept `text_features` and `embedding_features` params.
- `fit_and_transform_pipeline()` accepts `text_features` and `embedding_features` params to exclude from encoding/scaling.

## 2026-04-14 — Typed Training Parameters Refactor

### Breaking Changes

- `train_mlframe_models_suite` signature changed: removed `config_params`, `control_params`, `config_params_override`, `control_params_override`, and `**kwargs`.
- New parameters: `hyperparams_config` (`ModelHyperparamsConfig` or dict) and `behavior_config` (`TrainingBehaviorConfig` or dict).
- `select_target()` signature changed accordingly.

### Added

- **`ModelHyperparamsConfig`** Pydantic class in `configs.py`: typed replacement for `config_params`/`config_params_override` dicts. Fields: `iterations`, `learning_rate`, `early_stopping_rounds`, `has_time`, `rfecv_kwargs`, per-model kwargs (`cb_kwargs`, `lgb_kwargs`, `xgb_kwargs`, `hgb_kwargs`, `mlp_kwargs`, `ngb_kwargs`).
- **`TrainingBehaviorConfig`** Pydantic class in `configs.py`: typed replacement for `control_params`/`control_params_override` dicts. Fields: `prefer_gpu_configs`, `prefer_cpu_for_lightgbm`, `prefer_cpu_for_xgboost`, `prefer_calibrated_classifiers`, `use_robust_eval_metric`, `nbins`, `fairness_features`, `fairness_min_pop_cat_thresh`, `cont_nbins`, `metamodel_func`, `callback_params`, `cb_fit_params`, `use_flaml_zeroshot`, scoring configs.
- Both classes exported from `mlframe.training` and added to `__init__.py` lazy imports.
- Constants `DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH`, `DEFAULT_RFECV_*` moved to `configs.py` (canonical location).

### Changed

- `_initialize_training_defaults()` simplified: no longer normalizes 4 dict params.
- `_build_common_params_for_target()` accepts `TrainingBehaviorConfig` instead of dict.
- `_should_skip_catboost_metamodel()` accepts `TrainingBehaviorConfig` instead of dict.
- `_compute_fairness_subgroups()` accepts `TrainingBehaviorConfig` instead of dict.
- `TrainingConfig` class updated: `config_params_override`/`control_params_override` fields replaced with `hyperparams: ModelHyperparamsConfig` and `behavior: TrainingBehaviorConfig`.

### Fixed

- **Bug**: Tests used `models["target"][TargetTypes.X]` but actual structure is `models[TargetTypes.X]["target"]` — fixed across all test files.

### Migration

```python
# Before:
train_mlframe_models_suite(
    ...,
    config_params_override={"iterations": 10, "cb_kwargs": {"task_type": "CPU"}},
    control_params_override={"prefer_calibrated_classifiers": False},
)

# After:
train_mlframe_models_suite(
    ...,
    hyperparams_config={"iterations": 10, "cb_kwargs": {"task_type": "CPU"}},
    behavior_config={"prefer_calibrated_classifiers": False},
)
```

## 2026-04-14 — Auto-skip Categorical Encoding + Verbose Logging

### Added

- **`skip_categorical_encoding`** flag on `PolarsPipelineConfig`: when `True`, the polars-ds pipeline and sklearn pandas path skip ordinal/onehot encoding of categorical features. **Auto-detected** by `train_mlframe_models_suite` — when all requested `mlframe_models` support Polars natively (cb, xgb, hgb), the flag is set automatically, avoiding wasted encoding work and preserving original categorical dtypes.
- **Verbose timing & shape logging** across the training pipeline (`verbose=True`):
  - `core.py`: Phase 1 (data loading, FTE, preprocessing), Phase 2 (splitting with shapes), Phase 3 (pipeline with dtypes), per-model `process_model()` timing, Polars fastpath activation logging
  - `trainer.py`: `model.fit()` timing with shape, `_apply_pre_pipeline_transforms` timing with shape, metrics computation timing
  - `pipeline.py`: Polars-ds pipeline creation timing (scaler/encoding config), transform timing with shape, sklearn categorical encoding timing with shape
- Helper functions `_df_shape_str(df)` and `_elapsed_str(start)` in `core.py`
- 6 parametrized tests for `skip_categorical_encoding` auto-detection (all-native, mixed, non-native model lists)

### Changed

- `CatBoostStrategy.cache_key` = `"catboost"` (was inherited `"tree"`). `XGBoostStrategy.cache_key` = `"xgboost"` (was inherited `"tree"`). Each Polars-native model now gets its own pipeline cache slot, preventing cross-contamination when running multiple models together (e.g. `["cb", "xgb", "hgb"]`).

### Fixed

- **Bug**: XGBoost Polars fastpath passed `cat_features` as a `fit()` parameter, causing `TypeError: XGBClassifier.fit() got an unexpected keyword argument 'cat_features'`. Only CatBoost accepts `cat_features` in `fit()` — XGBoost/HGB auto-detect `pl.Categorical` columns via `enable_categorical=True`.
- **Bug**: When running multiple Polars-native models together (e.g. `["cb", "xgb"]`), the pipeline cache shared the `"tree"` key, causing the second model to receive cached pandas DFs from the first — overriding the Polars fastpath and causing `KeyError: DataType(large_string)` in XGBoost.

## 2026-04-14 — XGBoost Polars Fastpath + Unified Categorical Handling

### Added

- **XGBoost Polars fastpath**: XGBoost (>= 3.1) now receives Polars DataFrames directly via `train_mlframe_models_suite`. String columns are cast to `pl.Categorical` (XGBoost auto-detects via `enable_categorical=True`). No cardinality limit unlike HGB.
- `XGBoostStrategy` in `training/strategies.py` — inherits `TreeModelStrategy`, adds `supports_polars = True` and `prepare_polars_dataframe` (casts `pl.String` → `pl.Categorical`).
- **Unified categorical type constants** in `training/strategies.py`:
  - `PANDAS_CATEGORICAL_DTYPES` — `frozenset({"category", "object", "string", "string[pyarrow]", "large_string[pyarrow]"})`
  - `get_polars_cat_columns(df)` — detects `pl.Categorical`, `pl.Utf8`, `pl.String` columns
  - `is_polars_categorical(dtype)` — type check helper
- 5 unit tests for `XGBoostStrategy.prepare_polars_dataframe` (string→categorical, high-cardinality, passthrough)
- `TestXGBoostPolarsClassification` — XGBoost trained directly on Polars with categorical features
- Parametrized integration tests extended: `test_polars_fastpath_parametrized` and `test_polars_fastpath_regression_target` now cover `["cb", "xgb", "hgb"]`

### Changed

- `training/strategies.py`: XGBoost (`"xgb"`) now uses `XGBoostStrategy` instead of shared `TreeModelStrategy`.
- **Refactored categorical detection** across codebase to use unified constants:
  - `pipeline.py`: uses `PANDAS_CATEGORICAL_DTYPES` and `get_polars_cat_columns()`
  - `trainer.py:_filter_categorical_features`: uses unified constants, **fixed missing `pl.Utf8` bug** and missing pandas string types
  - `utils.py:get_categorical_columns`: uses unified constants
  - `core.py`: uses `get_polars_cat_columns()` for pre-pipeline detection

### Fixed

- **Bug**: `_filter_categorical_features` in `trainer.py` did not include `pl.Utf8` in Polars detection, silently filtering out valid categorical columns.
- **Bug**: `_filter_categorical_features` pandas path only checked `["category", "object"]`, missing `"string"`, `"string[pyarrow]"`, `"large_string[pyarrow]"`.

### Polars support matrix (updated)

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| XGBoost (`xgb`) | Yes (>= 3.1) | Yes (auto-casts strings → pl.Categorical) |
| HGB | Yes (numeric + Categorical) | Yes (auto-casts strings, handles cardinality > 255) |
| LightGBM (`lgb`) | No (broken in 4.6) | No |
| Linear models | No (internal NumPy conversion) | No |
| MLP / NGBoost | No | No |

## 2026-04-14 — HGB Polars Native Fastpath

### Added

- **HGB Polars fastpath** in `train_mlframe_models_suite`: when input is a Polars DataFrame, HGB models now receive it directly without intermediate pandas conversion. String categorical columns are automatically cast to `pl.Categorical` (cardinality ≤ 255) or ordinal-encoded to `pl.UInt32` (cardinality > 255, treated as continuous by HGB).
- `supports_polars = True` on `HGBStrategy`.
- `prepare_polars_dataframe(df, cat_features)` method on `ModelPipelineStrategy` base class (no-op default). `HGBStrategy` overrides it to handle cardinality-aware categorical casting.
- Pre-pipeline Polars originals are now saved before `fit_and_transform_pipeline()` to preserve string/categorical dtypes that polars-ds may convert to float.
- `cat_features_polars` list detected from pre-pipeline schema, used in Polars fastpath to ensure categorical columns are passed to models correctly.
- Polars fastpath now overrides `fit_params["cat_features"]` with pre-pipeline categorical columns when they differ from post-pipeline ones.
- 8 unit tests for `HGBStrategy.prepare_polars_dataframe` in `test_catboost_polars.py` (low/high cardinality, boundary 255/256, passthrough, missing columns).
- 2 integration tests in `test_core.py::TestPolarsNativeFastpath`: `test_hgb_receives_polars_dataframe`, `test_hgb_polars_categorical_is_cast`.

### Changed

- `training/strategies.py`: `HGBStrategy` now sets `supports_polars = True` and overrides `prepare_polars_dataframe` with cardinality-aware casting logic.
- `training/core.py`: Polars fastpath block now calls `strategy.prepare_polars_dataframe()` and sets `skip_preprocessing=True` for models that normally require encoding (HGB). Pre-pipeline Polars originals are saved before `fit_and_transform_pipeline()`.

### Polars support matrix (updated)

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| HGB | Yes (numeric + Categorical) | Yes (auto-casts strings, handles cardinality > 255) |
| LightGBM (`lgb`) | No | No |
| XGBoost (`xgb`) | No | No |
| Linear models | No | No |
| MLP / NGBoost | No | No |

## 2026-04-14 — Polars Native Fastpath for CatBoost

### Added

- **CatBoost Polars fastpath** in `train_mlframe_models_suite`: when input is a Polars DataFrame, CatBoost models now receive it directly without intermediate pandas conversion. This eliminates zero-copy overhead and allows CatBoost (>= 1.2.7) to use its native Polars ingestion path.
- `supports_polars` property on `ModelPipelineStrategy` (default `False`). New `CatBoostStrategy` subclass sets it to `True`.
- `CatBoostStrategy` in `training/strategies.py` — inherits `TreeModelStrategy`, adds `supports_polars = True`.
- Test file `tests/training/test_catboost_polars.py`: 11 tests covering CatBoost and HGB training directly on Polars DataFrames with categorical, numeric, text, and embedding features, plus early stopping on a Polars validation set.
- Integration tests in `tests/training/test_core.py` (`TestPolarsNativeFastpath`):
  - `test_catboost_receives_polars_dataframe` — monkeypatches `_train_model_with_fallback` to verify CatBoost `.fit()` receives a Polars DataFrame.
  - `test_non_catboost_still_gets_pandas` — verifies Ridge still receives pandas when input is Polars.

### Changed

- `training/core.py`: original Polars DataFrames are preserved before `_convert_dfs_to_pandas()` and substituted into `common_params` for models with `supports_polars`.
- `training/trainer.py`:
  - `train_df.columns.to_list()` replaced with `list(train_df.columns)` for Polars compatibility.
  - `_filter_categorical_features` now detects `pl.String` columns in addition to `pl.Categorical` when input is Polars.
- `training/strategies.py`: CatBoost (`"cb"`) now uses `CatBoostStrategy` instead of the shared `TreeModelStrategy`.

### Polars support matrix

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| HGB | Yes (numeric only) | No (requires category encoding) |
| LightGBM (`lgb`) | No | No |
| XGBoost (`xgb`) | No | No |
| Linear models | No | No |
| MLP / NGBoost | No | No |
