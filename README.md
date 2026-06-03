# mlframe

[![CI](https://github.com/fingoldo/mlframe/workflows/CI/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml)
[![sklearn-matrix](https://github.com/fingoldo/mlframe/workflows/sklearn-matrix/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml)
[![codecov](https://codecov.io/gh/fingoldo/mlframe/branch/master/graph/badge.svg)](https://codecov.io/gh/fingoldo/mlframe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A machine-learning framework for tabular data with a single entry point,
`train_mlframe_models_suite`, that trains, evaluates, calibrates, ensembles, and
reports across scikit-learn, CatBoost, LightGBM, XGBoost, HistGradientBoosting,
and PyTorch Lightning models on one dataset. It handles polars and pandas frames,
mixed dtypes, text features, ranking and quantile targets, and composite-target
stacking through a uniform API.

The changelog lives in [CHANGELOG.md](CHANGELOG.md).

## Installation

`mlframe` depends on [`pyutilz`](https://github.com/fingoldo/pyutilz), a sibling
utility library (parallel execution, pandas/polars helpers, hardware
introspection). Neither package is published to PyPI yet, so install both from
source — **`pyutilz` first**, then `mlframe`:

```bash
git clone https://github.com/fingoldo/pyutilz.git
git clone https://github.com/fingoldo/mlframe.git

pip install -e ./pyutilz
pip install -e ./mlframe
```

The core install pulls only the lightweight stack (numpy, pandas, polars, scipy,
scikit-learn, pyarrow, joblib, tqdm, pydantic, numba). Heavier stacks ship as
optional extras:

```bash
pip install -e "./mlframe[boosting]"             # catboost + lightgbm + xgboost
pip install -e "./mlframe[calibration]"          # shap + venn-abers + netcal + betacal + pycalib
pip install -e "./mlframe[neural]"               # torch + pytorch-lightning
pip install -e "./mlframe[automl]"               # flaml (HPO)
pip install -e "./mlframe[feature_engineering]"  # pysr (symbolic regression) + optbinning
pip install -e "./mlframe[sampling]"             # imbalanced-learn + category-encoders + iterative-stratification
pip install -e "./mlframe[polars_ext]"           # polars-talib + polars-ds
pip install -e "./mlframe[viz]"                  # matplotlib + plotly + seaborn + altair + hvplot
pip install -e "./mlframe[mlflow]"               # mlflow experiment tracking
pip install -e "./mlframe[db]"                   # sqlalchemy + psycopg2 + duckdb + pymongo + zstandard
pip install -e "./mlframe[signal]"               # antropy + astropy + pywavelets + ruptures
pip install -e "./mlframe[unsupervised]"         # hdbscan + umap-learn
pip install -e "./mlframe[stats]"                # statsmodels
pip install -e "./mlframe[gpu]"                  # cupy + gpu-info (match your CUDA build)
pip install -e "./mlframe[transformer]"          # transformer-style FE, numba-only CPU path
pip install -e "./mlframe[transformer_ann]"      # adds hnswlib for approximate-NN at N >= 500k
pip install -e "./mlframe[transformer_gpu]"      # adds cupy-cuda12x for the GPU stages
pip install -e "./mlframe[transformer_full]"     # transformer + ann + gpu
pip install -e "./mlframe[all]"                  # all runtime extras above
pip install -e "./mlframe[dev]"                  # pytest + coverage + ruff + black + mypy + bandit + pre-commit
```

Requires Python 3.9 or newer; tested on 3.9 through 3.14. The full core stack
(numpy, numba/llvmlite, polars, scikit-learn, pyarrow, pydantic) ships `cp314`
wheels and the numba JIT kernels compile and run on 3.14.

For a development checkout, add the dev extras and the pre-commit hooks:

```bash
pip install -e "./mlframe[boosting,calibration,viz,dev]"
cd mlframe
pre-commit install
pytest
```

## Modules

| Sub-package                   | Purpose |
| ----------------------------- | ------- |
| `mlframe.training`            | End-to-end training pipeline: `train_mlframe_models_suite`, per-model strategies, configs, feature handling, dummy baselines, AutoML, neural nets |
| `mlframe.feature_engineering` | Numerical / time-series / financial / categorical / Hurst / MPS features, brute-force PySR search |
| `mlframe.feature_selection`   | MRMR (multiple variants), RFECV, Boruta-SHAP, mutual-information and optbinning filters and wrappers |
| `mlframe.metrics`             | Calibration (ICE, ECE, Brier REL/RES/UNC, CMAEW), classification (KS, MCC, Cohen kappa, balanced accuracy, G-mean, BSS, Gini, F-beta, Lift@k, Hosmer-Lemeshow, top-k, RPS), regression (RMSLE, MAPE/SMAPE/MASE, MBE, NSE, Poisson/Gamma/Tweedie deviance, rank correlations), multilabel and LTR metrics, CRPS-from-quantiles, drift (PSI/KL/JS/Wasserstein), and plotting |
| `mlframe.evaluation`          | Performance reporting across CV folds and holdout sets |
| `mlframe.calibration`         | Calibration diagnostics and isotonic / Platt / beta / Venn-Abers post-hoc calibrators |
| `mlframe.models`              | Ensembling (stack / blend / vote), hyperparameter optimisation, splitting strategies |
| `mlframe.estimators`          | scikit-learn-compatible custom estimators, early-stopping-aware wrappers, pipelines |
| `mlframe.preprocessing`       | NaN cleaning, scaling, outlier handling, clustering |
| `mlframe.inference`           | Batch and streaming prediction, SHAP / permutation explainability |
| `mlframe.reporting`           | Matplotlib and plotly chart backends, spec-driven panel rendering |
| `mlframe.core`                | numba-accelerated array ops, statistical helpers, EWMA |
| `mlframe.data`                | Built-in and synthetic dataset generators |
| `mlframe.testing`             | Parametric frame generation for property-based tests |
| `mlframe.integrations`        | Optional third-party integrations (MLflow) |
| `mlframe.utils`               | EDA, experiments, text, and miscellaneous helpers |

## Quick examples

**One-call multi-model training and evaluation.** Each model is trained with its
native preprocessing strategy (CatBoost / LightGBM / XGBoost on raw frames; linear
and neural pipelines get encoded, imputed, and scaled). The suite returns per-model
predictions, fit-time metrics, calibration diagnostics, and a reporting spec.

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

**Composite-target stacking.** Train K base learners on `y`, then fit a
meta-learner on out-of-fold residuals to capture targeted error structure. The
estimator is scikit-learn-compatible (clone, `get_params`, `sample_weight`, runtime
stats callback), with cross-version behaviour pinned by CI.

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

**Per-target metric panel.** Computes the Brier reliability / resolution /
uncertainty decomposition, ICE bands, ECE, and CMAEW, with handling for class
imbalance and small-bin variance.

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

**MRMR / RFECV feature selection.** Several MRMR variants (FCQ, MID, FCD, plus
n-way interaction extensions) and an RFECV wrapper that reads LightGBM / XGBoost /
CatBoost feature importances correctly.

```python
from mlframe.feature_selection.filters.mrmr import mrmr_classif
from mlframe.feature_selection.wrappers import RFECVCustom

selected = mrmr_classif(X, y, K=30, scheme="fcq")

rfe = RFECVCustom(estimator=LGBMClassifier(), step=0.1, cv=5)
rfe.fit(X, y)
```

**SHAP-proxied feature selection (`ShapProxiedFS`).** Trains one model on all
features, computes SHAP values once, then approximates the prediction of a model
trained on any feature subset `S` by the coalition value `base + sum_{j in S} phi_j`.
Subsets can therefore be ranked without retraining (roughly 450x faster per subset
than an honest retrain in-repo). The cheap ranking is re-validated on a disjoint
holdout to pick the final subset. Backends: exact numba/CUDA brute force for
`n <= ~22`, otherwise beam / greedy / genetic / annealing / gradient search. A
proxy-trust guard measures proxy-vs-honest rank fidelity on the data and surfaces
the known limitation (the proxy under-credits subsets that drop a feature whose
signal correlated survivors could recover).

```python
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

sel = ShapProxiedFS(classification=True, metric="brier", optimizer="auto")
sel.fit(X, y)
print(sel.selected_features_)
report = sel.shap_proxy_report_
print(report["trust"]["spearman"])
print(report["importance_ablation"]["proxy_wins"])
X_sel = sel.transform(X)
```

For wide data (hundreds to tens of thousands of features) it scales via a
native-importance pre-filter, correlated-feature clustering, and a SHAP-importance
pre-screen, so the search runs on a reduced set of representatives that are then
expanded and pruned back to real columns. `prefilter_method` trades speed for
interaction-awareness (`"model"`, `"univariate"`, `"fast_model"`, `"gpu_model"`,
`"two_stage"`, default `"auto"`). Optional opt-in levers include
`interaction_aware`, `config_jitter` + `uncertainty_penalty`, and `active_learning`.
`ShapProxiedFS.preflight(X, y)` returns a run / caution / fallback recommendation
before a full fit.

```python
sel = ShapProxiedFS(classification=True, cluster_features=True, prefilter_top=2000,
                    interaction_aware=True, config_jitter=True, uncertainty_penalty=0.3)
print(ShapProxiedFS.preflight(X, y, classification=True)["recommendation"])
```

**Friend-graph post-analysis.** After screening, the `MRMR` estimator can build a
graph of the selected features (node = feature sized by entropy, edge = pairwise
mutual information, arrow = asymmetric dependency, colour = unique / suspected-sink /
middling). It flags a feature correlated with many genuine predictors but carrying
no unique target information, which greedy MRMR can otherwise promote early. The
graph is exposed on the fitted estimator and rendered through the reporting
backends. Diagnostic by default; pruning is opt-in.

```python
from mlframe.feature_selection.filters.mrmr import MRMR

sel = MRMR(build_friend_graph=True, friend_graph_prune=False).fit(X, y)
g = sel.friend_graph_
print(g.suspected_garbage)
print(g.to_meta()["class_counts"])

pruned = MRMR(friend_graph_prune=True).fit(X, y)
```

**Automatic feature engineering (`fe_auto`).** `MRMR` exposes around 50 opt-in
feature-engineering generators, each useful only on a specific data shape. Rather
than asking the caller to flip flags by hand, `MRMR(fe_auto=True)` fingerprints
`(X, y)` before the FE stages run and enables only the generators whose data-shape
precondition is met. It only adds generators (a flag set `True` by the caller is
never turned off), and restores the original constructor values after `fit` so
semantics stay stable across `fit` / `clone` / pickle. The default `fe_auto=False`
keeps the legacy path byte-identical. `MRMR.recommend_enabled_fe(X, y)` returns the
same recommendation without running a fit.

```python
from mlframe.feature_selection.filters.mrmr import MRMR

sel = MRMR(fe_auto=True).fit(X, y)
print(MRMR.recommend_enabled_fe(X, y)["recommended_enable"])
```

**Param-Oracle (`mlframe.utils._param_oracle.ParamOracle`).** Many hot decisions in
the codebase (MI scorer, CUDA kernel variant, FE recipe) have a data-dependent
optimum rather than a constant. `ParamOracle` learns the fingerprint-to-best-param
mapping: it stores only scalar fingerprint statistics, the parameter combo, and the
scalar objective in an on-disk parquet store (never raw arrays), then resolves a new
fingerprint by exact bucket match, k-NN, or global best. It reuses the per-host
layout conventions of `pyutilz.system.kernel_tuning_cache` without modifying it.
Modes: `"benchmark"` (sweep and record every combo), `"inference"` (recommend only),
`"hybrid"` (epsilon-greedy explore/exploit).

```python
from mlframe.utils._param_oracle import ParamOracle

oracle = ParamOracle(
    "my_kernel.parquet",
    param_space={"variant": ["njit", "cuda"], "block": [128, 256]},
    mode="hybrid",
    minimize="elapsed_s",
    epsilon=0.1,
)

@oracle
def my_kernel(X, variant="njit", block=128):
    ...
    return result
```

**Time-series and financial feature engineering on Polars.** Windowed aggregation,
ACF, Hurst exponent, TA-Lib indicators, and market-wide rolling features. Most
extraction paths run Polars-native without copying to pandas.

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

**Post-hoc probability calibration.** Compare Venn-Abers, isotonic, Platt, beta, and
per-class isotonic on the same out-of-fold predictions and pick the calibrator that
minimises Brier reliability without inflating resolution.

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

**Batch inference with adaptive worker count.** Worker count adapts to available
RAM; results stream back with exception propagation.

```python
from mlframe.inference.predict import batch_predict

preds = batch_predict(
    model=clf,
    X=large_frame,
    batch_size=10_000,
    n_workers="auto",
)
```

## Caching strategy

The training suite runs two caching layers with different key strategies, because
the lifetime of the cached value, the cost of computing the key, and the failure
mode of a stale hit differ between them.

**Layer 1: `_PRE_PIPELINE_CACHE` (content-keyed).** Caches the output of
`(SimpleImputer + StandardScaler + feature selectors).fit_transform(train_df, val_df)`
so consecutive models in one suite call that share the same pre-pipeline structure
reuse the fitted transforms. Keys come from a content fingerprint of `train_df`,
`val_df`, the pipeline signature, the target, the target name, and optionally sample
weights. Consecutive lookups see the same Python objects (same `id()`) but the value
differs across targets, so id-keying would alias entries and content-keying is the
only safe option. The hash cost is amortised across the pre-pipeline fit a hit skips.

**Layer 2: `FeatureCache.InMemoryKey` (id-keyed).** Caches per-column intermediate
stats (MRMR scores, target-encoder folds, RFECV ranks) within a single suite call.
Keys are tuples of `(session_id, id(train_df), id(train_idx), column,
params_canonical_hash, provider_signature)`. `id()` is safe here because the suite
holds strong references to `train_df` and `train_idx` for its whole lifetime, and the
per-call `session_id` prevents cross-call collisions. These hits land in inner
per-column loops where a content hash per lookup would dominate the work the cache is
meant to skip. Cross-session reuse is provided separately by a content-keyed `DiskKey`.

In short: layer 1 spans model boundaries inside one suite call (content-keying
mandatory because intermediate frames are deleted between fits and `id()`s recycle);
layer 2 spans only inner-loop boundaries where suite-level strong references keep
`id()` stable, so id-keying is cheap and safe.

## Design notes

- **Modular, opt-in extras.** The core install pulls only the lightweight stack;
  heavy dependencies (CatBoost, PyTorch, MLflow, SHAP, plotly) are extras and are
  lazy-imported at call site, so nothing fails on `import mlframe`.
- **Polars-native where it matters.** Tree models that accept Arrow-backed frames
  (CatBoost, HGB, XGBoost) skip the polars-to-pandas round-trip; non-native models
  get a zero-copy Arrow view via `pyarrow.Table.to_pandas(zero_copy_only=True)`.
- **scikit-learn version pinning.** A dedicated
  [sklearn-matrix CI workflow](.github/workflows/sklearn-matrix-ci.yml) tests the
  composite-target wrapper surface against scikit-learn 1.5 through 1.8 on every PR,
  catching attribute-delegation breakage before users hit it.
- **Fuzz-tested.** Roughly 150 pairwise and 400 three-wise (IPOG-covering) parameter
  combos run per release. Combo regressions become permanent sensors so they do not
  recur.

## Roadmap

Operations that do not fit the current pipeline-slot abstractions are parked here
pending a refactor:

- **Row-wise transformations** (per-sample normalisation). `Normalizer` projects each
  sample onto a unit hypersphere, which suits text/embedding similarity but silently
  breaks tree models that rely on absolute feature magnitudes. A dedicated
  `row_transform` pipeline slot is planned so row-wise operations have an unambiguous
  home that cannot be confused with column scalers.

## Testing

```bash
pytest                                    # full suite
pytest -m fast                            # representative subset (<15s)
pytest -m "not slow and not gpu"          # CI default
pytest --cov=src/mlframe --cov-report=html
```

Markers: `slow`, `integration`, `gpu`, `multigpu`, `benchmark`, `windows_only`,
`linux_only`, `fast`.

## Contributing

Pull requests are welcome. Code style is `black` + `ruff` with a line length of 160.
Every new feature ships with a unit test, a quantitative business-value test, a
representative `@pytest.mark.fast` subset, and a `cProfile` hotspot check. See
[CLAUDE.md](CLAUDE.md) for project conventions and the fuzz / combo test philosophy.

## License

MIT, see [LICENSE](LICENSE).
