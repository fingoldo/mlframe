# mlframe

A modular ML training framework built on top of scikit-learn, CatBoost, LightGBM, XGBoost, and PyTorch. Provides a unified API (`train_mlframe_models_suite`) for training, evaluating, and ensembling multiple model types on the same dataset with minimal boilerplate.

## Architecture

```
mlframe/
  training/
    core.py              # train_mlframe_models_suite — main entry point
    trainer.py           # configure_training_params, train_and_evaluate_model
    train_eval.py        # select_target, process_model, _call_train_evaluate_with_configs
    strategies.py        # ModelPipelineStrategy — per-model preprocessing logic
    helpers.py           # get_training_configs — config factory for CB/LGB/XGB/HGB
    configs.py           # Pydantic config models (DataConfig, TrainingControlConfig, etc.)
    pipeline.py          # Polars-ds pipeline, prepare_df_for_catboost
    evaluation.py        # report_model_perf, metrics computation
    automl.py            # AutoGluon / TPOT wrappers
    neural/              # LSTM, GRU, Transformer wrappers (PyTorch Lightning)
  feature_selection/     # MRMR, RFECV wrappers
  metrics.py             # ICE, calibration metrics, custom scorers
  tests/
    training/
      test_core.py       # Integration tests for train_mlframe_models_suite
      test_core_coverage.py  # Comprehensive coverage tests (48 tests, all code paths)
      test_catboost_polars.py  # CatBoost & HGB native Polars support tests
      ...
```

## Data flow

```
Polars/pandas DataFrame
  |
  v
fit_and_transform_pipeline()        # optional polars-ds pipeline
  |
  +-- save Polars originals          # for models with supports_polars=True
  |
  v
_convert_dfs_to_pandas()            # zero-copy Arrow view
  |
  v
select_target()                     # builds common_params + models_params
  |
  v
MODEL LOOP (per model type):
  |
  +-- get_strategy(model_name)       # TreeModel / CatBoost / HGB / Neural / Linear
  |
  +-- if strategy.supports_polars:   # substitute Polars DFs into common_params
  |       use original Polars DFs
  |   else:
  |       use pandas DFs
  |
  +-- build_pipeline()               # encoding / imputation / scaling as needed
  |
  v
process_model() -> train_and_evaluate_model() -> model.fit(df, target)
```

## Strategy pattern (per-model preprocessing)

Each model type has a `ModelPipelineStrategy` that declares its preprocessing needs:

| Strategy | Models | Encoding | Imputation | Scaling | Polars native | Text features | Embedding features |
|----------|--------|:--------:|:----------:|:-------:|:-------------:|:-------------:|:------------------:|
| `CatBoostStrategy` | cb | No | No | No | Yes | Yes | Yes |
| `XGBoostStrategy` | xgb | No | No | No | Yes (auto-casts) | No | No |
| `TreeModelStrategy` | lgb | No | No | No | No | No | No |
| `HGBStrategy` | hgb | Yes | No | No | Yes (auto-casts) | No | No |
| `NeuralNetStrategy` | mlp, ngb | Yes | Yes | Yes | No | No | No |
| `LinearModelStrategy` | ridge, lasso, ... | Yes | Yes | Yes | No | No | No |

## Key parameters

`train_mlframe_models_suite` accepts:

- **`hyperparams_config`** — model hyperparameters (`ModelHyperparamsConfig` or dict): iterations, learning_rate, early_stopping_rounds, per-model kwargs (cb_kwargs, lgb_kwargs, etc.)
- **`behavior_config`** — training behavior flags (`TrainingBehaviorConfig` or dict): prefer_calibrated_classifiers, prefer_gpu_configs, fairness_features, etc.
- **`mlframe_models`** — list of model types to train: `["cb", "lgb", "xgb", "hgb", "ridge", "mlp", ...]`
- **`pipeline_config`** — Polars-ds pipeline configuration (see `PolarsPipelineConfig`)
  - `skip_categorical_encoding` — auto-set to `True` when all `mlframe_models` support Polars natively (cb, xgb, hgb), skipping unnecessary ordinal/onehot encoding in the pipeline. Can also be set manually.
- **`feature_types_config`** — feature type configuration (`FeatureTypesConfig` or dict):
  - `text_features` — list of free-text string columns (passed to CatBoost via `fit(text_features=[...])`, dropped for other models)
  - `embedding_features` — list of list-of-float vector columns (passed to CatBoost via `fit(embedding_features=[...])`, dropped for other models)
  - `auto_detect_feature_types` — when `True` (default), auto-detects embeddings from `pl.List(pl.Float*)` and splits string columns into text vs categorical by cardinality
  - `cat_text_cardinality_threshold` — unique value count threshold (default 50): `<= threshold` → categorical, `> threshold` → text
- **`custom_pre_pipelines`** — dict of custom sklearn transformers (e.g., PCA)
- **`verbose`** — when `True`, logs timing and shape info for every major phase (data loading, splitting, pipeline, per-model training, metrics)

## Feature-tier model grouping

When `text_features` or `embedding_features` are present, models are sorted by feature support level (most features first). CatBoost trains first with all columns, then text/embedding columns are dropped once per tier for remaining models. This avoids redundant column operations and enables aggressive memory cleanup between tiers.

## Running tests

```bash
python -m pytest mlframe/tests/training/test_catboost_polars.py -v
python -m pytest mlframe/tests/training/test_core.py::TestPolarsNativeFastpath -v
python -m pytest mlframe/tests/training/test_core.py::TestTextAndEmbeddingFeatures -v -m "not gpu"
```
