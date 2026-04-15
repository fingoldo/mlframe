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
- **`preprocessing_extensions`** — optional `PreprocessingExtensionsConfig` (or dict). Shared sklearn stack applied once after the Polars-ds pipeline; every model reuses the transformed frame. Covers scaler override (10 variants), `Binarizer`/`KBinsDiscretizer` (mutually exclusive), `PolynomialFeatures` with `memory_safety_max_features` guard, non-linear maps (`RBFSampler`/`Nystroem`/`AdditiveChi2Sampler`/`SkewedChi2Sampler`), TF-IDF, and dim reducers (PCA / KernelPCA / LDA / NMF / TruncatedSVD / FastICA / Isomap / UMAP / random projections / RandomTreesEmbedding / BernoulliRBM). `None` (default) is a byte-for-byte noop — the Polars-native fastpath is preserved. UMAP is gated via `importlib.util.find_spec` with an install-hint `ImportError`.
- **`custom_pre_pipelines`** — dict of custom sklearn transformers (e.g., PCA)
- **`verbose`** — when `True`, logs timing and shape info for every major phase (data loading, splitting, pipeline, per-model training, metrics)

### Sweeping variants — `run_grid`

When you need to compare multiple configs of the same suite, use
`mlframe.training.run_grid(base_kwargs, grid)` — it calls
`train_mlframe_models_suite` once per entry and collects results in a dict.
Grid entries may be raw dicts (auto-labelled `variant_0`, `variant_1`, …)
or `(label, dict)` tuples. With `stop_on_error=False` (default) a failing
variant is logged and stored as `{"error": repr(exc)}` while the sweep
continues.

## Feature-tier model grouping

When `text_features` or `embedding_features` are present, models are sorted by feature support level (most features first). CatBoost trains first with all columns, then text/embedding columns are dropped once per tier for remaining models. This avoids redundant column operations and enables aggressive memory cleanup between tiers.

## Fast test mode

Run the full test surface with one representative variant per parametrized group (scalers, dim reducers, optimizers, …):

```bash
pytest --fast                      # CLI flag
MLFRAME_FAST=1 pytest              # env var (equivalent)
```

Parametrized tests opt in by wrapping their argument list with `fast_subset`:

```python
from tests.conftest import fast_subset

@pytest.mark.parametrize("scaler", fast_subset(ALL_SCALERS, representative="StandardScaler"))
def test_scaler_round_trip(scaler): ...
```

Tests marked `@pytest.mark.slow` (or `slow_only`) are auto-skipped in fast mode.

## Running tests

```bash
python -m pytest mlframe/tests/training/test_catboost_polars.py -v
python -m pytest mlframe/tests/training/test_core.py::TestPolarsNativeFastpath -v
python -m pytest mlframe/tests/training/test_core.py::TestTextAndEmbeddingFeatures -v -m "not gpu"
```

Run the whole suite in parallel (falls back to rerunning last-failed verbosely):

```bash
pytest tests/ -n auto --maxprocesses=16 --dist loadscope && exit 0 || pytest tests/ -vv --lf
```

## Troubleshooting

### Windows fatal exception / access violation in numba kernels

After changing `NUMBA_NJIT_PARAMS` flags (e.g. `cache=True`, `nogil=True`), stale on-disk numba caches (`.nbi`/`.nbc` in `__pycache__`) from a prior build can trigger `Windows fatal exception: access violation` inside `compute_numerical_aggregates_numba` / similar kernels. The flags themselves are correct — a cold rebuild is required. Clear and retry:

```bash
find . -name "*.nbi" -delete; find . -name "*.nbc" -delete
```

## Security notes

- `joblib.load` / `dill.load` in `inference.py`, `pipelines.py`, `training/io.py` are gated by a `trusted_root` path check and (for dill) a `_SafeUnpickler` allowlist. Never load pickle/joblib artifacts from untrusted sources.
- `torch.load` is always called with `weights_only=True`.
- SQL field names in `experiments.py` are validated against an allowlist.

See `CHANGELOG.md` (2026-04-14 entry) for the full audit/fix history.
