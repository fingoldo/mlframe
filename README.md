# mlframe

[![CI](https://github.com/fingoldo/mlframe/workflows/CI/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml)
[![sklearn-matrix](https://github.com/fingoldo/mlframe/workflows/sklearn-matrix/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml)
[![codecov](https://codecov.io/gh/fingoldo/mlframe/branch/master/graph/badge.svg)](https://codecov.io/gh/fingoldo/mlframe)
[![PyPI](https://img.shields.io/pypi/v/mlframe.svg)](https://pypi.org/project/mlframe/)
[![Python](https://img.shields.io/pypi/pyversions/mlframe.svg)](https://pypi.org/project/mlframe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-grade machine-learning framework for tabular data. One uniform entry point (`train_mlframe_models_suite`) trains, evaluates, calibrates, ensembles, and reports across scikit-learn, CatBoost, LightGBM, XGBoost, HistGradientBoosting, and PyTorch Lightning models on the same dataset, with proper handling of polars/pandas frames, mixed dtypes, text features, ranking and quantile targets, and composite-target stacking.

## Installation

```bash
pip install mlframe                       # core: numpy + pandas + scipy + sklearn + pyutilz
pip install mlframe[boosting]             # catboost + lightgbm + xgboost
pip install mlframe[calibration]          # shap + venn-abers + netcal + betacal + pycalib
pip install mlframe[neural]               # torch + pytorch-lightning
pip install mlframe[automl]                # flaml (HPO)
pip install mlframe[feature_engineering]    # pysr (symbolic regression) + optbinning (WoE / monotonic binning)
pip install mlframe[sampling]             # imbalanced-learn + category-encoders + iterative-stratification
pip install mlframe[polars_ext]            # polars-talib + polars-ds (polars itself is a core dep)
pip install mlframe[viz]                  # matplotlib + plotly + seaborn + altair + hvplot
pip install mlframe[mlflow]               # mlflow experiment tracking
pip install mlframe[db]                   # sqlalchemy + psycopg2 + duckdb + pymongo + zstandard
pip install mlframe[signal]               # antropy + astropy + pywavelets + ruptures
pip install mlframe[unsupervised]         # hdbscan + umap-learn
pip install mlframe[stats]                # statsmodels
pip install mlframe[gpu]                  # cupy + gpu-info (match your CUDA build)
pip install mlframe[all]                  # all runtime extras above
pip install mlframe[dev]                  # pytest + coverage + ruff + black + mypy + bandit + pre-commit
```

Requires Python 3.9+. Tested on 3.9 through 3.13.

For development from source:

```bash
git clone https://github.com/fingoldo/mlframe.git
cd mlframe
pip install -e ".[boosting,calibration,viz,dev]"
pre-commit install
pytest
```

`mlframe` depends on [`pyutilz`](https://github.com/fingoldo/pyutilz) (sibling utility library: parallel execution, pandas/polars helpers, hardware introspection).

## Modules

| Sub-package                       | Purpose                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| `mlframe.training`                | End-to-end training pipeline: `train_mlframe_models_suite`, per-model strategies, configs, feature handling, dummy baselines, AutoML, neural nets |
| `mlframe.feature_engineering`     | Numerical / time-series / financial / categorical / Hurst / MPS features, brute-force PySR search |
| `mlframe.feature_selection`       | MRMR (multiple variants), RFECV, Boruta-SHAP, mutual-information, optbinning filters and wrappers |
| `mlframe.metrics`                 | ICE, ECE, Brier decomposition (REL/RES/UNC), CMAEW, calibration plots, quantile and ranking metrics |
| `mlframe.evaluation`              | Performance reporting across cv folds + holdout sets                                     |
| `mlframe.calibration`             | Calibration-quality diagnostics, isotonic / Platt / beta / Venn-Abers post-hoc calibrators |
| `mlframe.models`                  | Ensembling (stack / blend / vote), hyperparameter optimisation, splitting strategies     |
| `mlframe.estimators`              | sklearn-compatible custom estimators, early-stopping aware wrappers, pipelines           |
| `mlframe.preprocessing`           | NaN cleaning, scaling, outlier handling, clustering for preprocessing                    |
| `mlframe.inference`               | Batch and streaming prediction, SHAP / permutation explainability                        |
| `mlframe.reporting`               | Matplotlib + plotly chart backends; spec-driven panel rendering                          |
| `mlframe.core`                    | numba-accelerated array ops, statistical helpers, EWMA                                   |
| `mlframe.data`                    | Built-in / synthetic dataset generators                                                  |
| `mlframe.testing`                 | Parametric frame generation for property-based tests                                     |
| `mlframe.integrations`            | Optional third-party integrations (MLflow)                                               |
| `mlframe.utils`                   | EDA, experiments, text, miscellaneous helpers                                            |

## Quick examples

**One-call multi-model training and evaluation.** The suite trains every model with its native preprocessing strategy (CatBoost / LightGBM / XGBoost run on raw frames; linear and neural pipelines get encoded + imputed + scaled). It returns per-model predictions, fit-time metrics, calibration diagnostics, and a ready-to-render reporting spec:

```python
from mlframe.training import train_mlframe_models_suite

result = train_mlframe_models_suite(
    df=df,
    target="y",
    models=["cb", "lgb", "xgb", "hgb", "mlp"],
    regression=False,
    cv_folds=5,
    early_stopping_rounds=50,
    use_polars=True,
)

result.models["cb"].metrics["holdout_brier"]
result.models["lgb"].calibration_plot()
result.ensemble("stack").predict_proba(X_new)
```

**Composite-target stacking.** Train K base learners on `y`, fit a meta-learner on out-of-fold residuals to capture targeted error structure. sklearn-compatible (clone, get_params, sample_weight, runtime stats callback), with cross-sklearn-version behaviour pinned by CI:

```python
from mlframe.estimators.custom import CompositeTargetEstimator

est = CompositeTargetEstimator(
    base_estimator="lgb",
    meta_estimator="ridge",
    cv=5,
    target_transform="residual",
)
est.fit(X_train, y_train)
est.feature_importances_       # aggregated across folds
```

**Per-target metric panel.** Computes Brier reliability/resolution/uncertainty decomposition, ICE bands, ECE, CMAEW, with proper handling of class imbalance and small-bin variance:

```python
from mlframe.metrics import compute_calibration_report

report = compute_calibration_report(
    y_true=y_test,
    y_proba=clf.predict_proba(X_test)[:, 1],
    n_bins=15,
    method="quantile",
)
print(report.brier_rel, report.brier_res, report.brier_unc, report.ece)
```

**MRMR / RFECV feature selection.** Several MRMR variants (FCQ, MID, FCD, plus n-way interaction extensions) plus an RFECV wrapper that handles LightGBM / XGBoost / CatBoost feature-importance access correctly:

```python
from mlframe.feature_selection.filters.mrmr import mrmr_classif
from mlframe.feature_selection.wrappers import RFECVCustom

# Filter method (MRMR FCQ variant)
selected = mrmr_classif(X, y, K=30, scheme="fcq")

# Wrapper (RFECV with non-sklearn-native importance fallback)
rfe = RFECVCustom(estimator=LGBMClassifier(), step=0.1, cv=5)
rfe.fit(X, y)
```

**Time-series and financial feature engineering on Polars.** Windowed aggregation, ACF, Hurst exponent, TA-Lib indicators, market-wide rolling features. Most extraction paths run Polars-native without copying to pandas:

```python
from mlframe.feature_engineering.timeseries import create_aggregated_features
from mlframe.feature_engineering.financial import compute_market_features

agg = create_aggregated_features(
    df, value_col="price", time_col="ts",
    windows=[5, 15, 60],
    aggs=["mean", "std", "min", "max", "acf1"],
)

market = compute_market_features(df, ohlc_cols=("open", "high", "low", "close"))
```

**Post-hoc probability calibration.** Compare Venn-Abers, isotonic, Platt, beta, and per-class isotonic on the same out-of-fold preds; pick the calibrator that minimises Brier reliability without inflating reduction:

```python
from mlframe.calibration.post import select_best_calibrator

best, scores = select_best_calibrator(
    y_true=y_val,
    y_proba=oof_proba,
    candidates=["isotonic", "platt", "beta", "venn_abers"],
    objective="brier_rel",
)
calibrated = best.transform(test_proba)
```

**Batch inference with adaptive worker count.** Worker count adapts to available RAM; results stream back with proper exception propagation:

```python
from mlframe.inference.predict import batch_predict

preds = batch_predict(
    model=clf,
    X=large_frame,
    batch_size=10_000,
    n_workers="auto",
)
```

## Design notes

- **Modular, opt-in extras.** Core install pulls `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow`, `joblib`, `tqdm`, `pyutilz`, `pydantic`. Heavy stacks (CatBoost, PyTorch, MLflow, SHAP, plotly) ship as extras; nothing ImportError-s on `import mlframe` because optional deps are lazy-imported at call site.
- **Polars-native where it matters.** Tree models that accept Arrow-backed frames (CatBoost, HGB, XGBoost auto-cast) skip the polars-to-pandas round-trip; non-native models get a zero-copy Arrow view via `pyarrow.Table.to_pandas(zero_copy_only=True)`.
- **sklearn-version pinning.** A dedicated [sklearn-matrix CI workflow](.github/workflows/sklearn-matrix-ci.yml) tests the composite-target wrapper surface against scikit-learn 1.5, 1.6, 1.7, and 1.8 on every PR; attribute-delegation breakage shows up before users hit it.
- **Fuzz-tested.** ~150 pairwise + ~400 3-wise (IPOG-covering) parameter combos run per release, hitting axes the unit tests don't reach. Combo regressions get permanent sensors so they don't recur.

## Testing

```bash
pytest                                    # full suite
pytest -m fast                            # representative-subset run (<15s)
pytest -m "not slow and not gpu"          # CI default
pytest --cov=src/mlframe --cov-report=html
```

Markers: `slow`, `integration`, `gpu`, `multigpu`, `benchmark`, `windows_only`, `linux_only`, `fast`.

## Contributing

Pull requests welcome. Code style: `black` + `ruff`, line length 160. Every new feature ships with: a unit test, a quantitative business-value test, a representative `@pytest.mark.fast` subset, and a `cProfile` hotspot check. See [CLAUDE.md](CLAUDE.md) for project conventions and the fuzz / combo test philosophy.

## License

MIT, see [LICENSE](LICENSE).
