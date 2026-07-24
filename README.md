# mlframe

[![CI](https://github.com/fingoldo/mlframe/workflows/CI/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml)
[![MyPy](https://github.com/fingoldo/mlframe/actions/workflows/mypy-full.yml/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/mypy-full.yml)
[![Black](https://github.com/fingoldo/mlframe/workflows/Black/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/black-filtered.yml)
[![sklearn-matrix](https://github.com/fingoldo/mlframe/workflows/sklearn-matrix/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml)
[![numba coverage](https://github.com/fingoldo/mlframe/actions/workflows/numba-coverage.yml/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/numba-coverage.yml)
[![codecov](https://codecov.io/gh/fingoldo/mlframe/branch/master/graph/badge.svg)](https://codecov.io/gh/fingoldo/mlframe)
[![codecov-numba](https://codecov.io/gh/fingoldo/mlframe/branch/master/graph/badge.svg?flag=numba-disabled)](https://codecov.io/gh/fingoldo/mlframe/flags)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://fingoldo.github.io/mlframe/)

A machine-learning framework for tabular data with a single entry point,
`train_mlframe_models_suite`, that trains, evaluates, calibrates, ensembles, and
reports across scikit-learn, CatBoost, LightGBM, XGBoost, HistGradientBoosting,
and PyTorch Lightning models on one dataset. It handles polars and pandas frames,
mixed dtypes, text features, ranking and quantile targets, and composite-target
stacking through a uniform API.

The changelog lives in [CHANGELOG.md](CHANGELOG.md). Full guide index, including
baseline diagnostics, honest-diagnostics, calibration policy, composite-target
config reference, and error-decoding guides: [docs/README.md](docs/README.md).

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

The core install pulls a compact stack (numpy, pandas, polars, scipy, scikit-learn,
pyarrow, joblib, tqdm, pydantic, numba), plus matplotlib and several other packages
that are imported unconditionally at module load time -- see `pyproject.toml`'s
`[project.dependencies]` for the exact list. Heavier stacks ship as optional extras:

```bash
pip install mlframe[all,dev]                     # full install (recommended)

pip install -e "./mlframe[boosting]"             # catboost + lightgbm + xgboost
pip install -e "./mlframe[calibration]"          # shap + venn-abers + netcal + betacal + pycalib
pip install -e "./mlframe[neural]"               # torch + lightning + captum + transformers
pip install -e "./mlframe[automl]"               # flaml (HPO)
pip install -e "./mlframe[feature_engineering]"  # pysr (symbolic regression) + optbinning
pip install -e "./mlframe[sampling]"             # imbalanced-learn + iterative-stratification
pip install -e "./mlframe[polars_ext]"           # polars-talib + polars-ds
pip install -e "./mlframe[viz]"                  # matplotlib + plotly + seaborn + altair + hvplot
pip install -e "./mlframe[mlflow]"               # mlflow experiment tracking
pip install -e "./mlframe[db]"                   # sqlalchemy + psycopg2 + duckdb + pymongo + zstandard
pip install -e "./mlframe[signal]"               # antropy + astropy + pywavelets + ruptures
pip install -e "./mlframe[unsupervised]"         # hdbscan + umap-learn
pip install -e "./mlframe[stats]"                # statsmodels
pip install -e "./mlframe[gpu,transformer_gpu]"  # cupy + gpu-info + cupy-cuda12x for the GPU stages (match your CUDA build)
pip install -e "./mlframe[transformer,transformer_ann]"  # transformer-style FE (numba-only CPU path) + hnswlib for approximate-NN at N >= 500k
pip install -e "./mlframe[all]"                  # runtime extras EXCEPT the CUDA-build-specific gpu / transformer_gpu (install those explicitly on a CUDA host)
pip install -e "./mlframe[dev]"                  # pytest + coverage + ruff + black + mypy + bandit + pre-commit
```

Requires Python 3.9 or newer; tested on 3.9 through 3.14. The full core stack
(numpy, numba/llvmlite, polars, scikit-learn, pyarrow, pydantic) ships `cp314`
wheels and the numba JIT kernels compile and run on 3.14.

For development:

```bash
git clone https://github.com/fingoldo/mlframe.git
cd mlframe
pip install -e ".[all,dev]"
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
| `mlframe.core`                | numba-accelerated array ops, statistical helpers, EWMA, spectral matrix seriation, robust location estimators |
| `mlframe.data`                | Built-in and synthetic dataset generators |
| `mlframe.testing`             | Parametric frame generation for property-based tests |
| `mlframe.integrations`        | Optional third-party integrations (MLflow) |
| `mlframe.utils`               | EDA, experiments, text, `ParamOracle` data-dependent parameter learning, and miscellaneous helpers |
| `mlframe.inspection`          | Model-agnostic interpretation primitives absent from `sklearn.inspection` (Friedman-Popescu H-statistic interaction detection) |
| `mlframe.signal`               | DTW alignment and Gaussian-Process smoothing/confidence features for irregularly-sampled series |
| `mlframe.system`              | GPU import guards and kernel-tuning-cache integration shared across subpackages |
| `mlframe.votenrank`           | Ensemble-blending strategies beyond `mlframe.models` (confidence-gated, adversarial-stochastic, rank-splice, KNN-fallback blends) |

## Quick examples

**One-call multi-model training and evaluation.** Each model is trained with its
native preprocessing strategy (CatBoost / LightGBM / XGBoost on raw frames; linear
and neural pipelines get encoded, imputed, and scaled). The suite returns a
`(models, metadata)` tuple: `models` is a per-target-type dict of trained model
entries, `metadata` carries fit-time metrics, calibration diagnostics, baseline
diagnostics, and the reporting spec. The features and targets are pulled from the
frame by a caller-supplied extractor (see `SimpleFeaturesAndTargetsExtractor`).

```python
import numpy as np, pandas as pd
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

rng = np.random.default_rng(0)
df = pd.DataFrame({"x1": rng.normal(size=500), "x2": rng.normal(size=500), "x3": rng.integers(0, 5, size=500)})
df["y"] = df["x1"] * 2 - df["x2"] + rng.normal(scale=0.1, size=500)
X_new = df[["x1", "x2", "x3"]].iloc[:5]

fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

models, metadata = train_mlframe_models_suite(
    df=df,
    target_name="y",
    model_name="exp_quickstart",
    features_and_targets_extractor=fte,
    mlframe_models=["lgb"],
)

# `models` is keyed by target-type, then target name; the value is a list with one
# entry per requested model (plus ensembles, if enabled), each exposing the fitted model.
entry = models["regression"]["y"][0]
y_pred = entry.model.predict(X_new)

# Diagnostics live in `metadata` (per target-type, per target).
print(metadata["baseline_diagnostics"]["regression"]["y"])
```

**Composite-target regression.** A scikit-learn-compatible wrapper that fits a
single inner regressor on a *transformed* target (e.g. `T = y - alpha*base`) and
inverts the transform at predict time, exposing residual structure the raw target
buries. Pick the transform by name (`list_transforms()` enumerates them) and name
the base-feature column it residualises against. The wrapper clones the inner
estimator, delegates `feature_importances_` / `get_booster()` / other attributes
transparently, and pins cross-version behaviour in CI.

```python
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mlframe.training.composite import CompositeTargetEstimator

rng = np.random.default_rng(0)
y_prev = rng.normal(size=300)
X_train = pd.DataFrame({"y_prev": y_prev, "x1": rng.normal(size=300)})
y_train = y_prev + X_train["x1"] * 0.5 + rng.normal(scale=0.1, size=300)
X_new = X_train.iloc[:5]

est = CompositeTargetEstimator(
    base_estimator=RandomForestRegressor(),
    transform_name="linear_residual",
    base_column="y_prev",
)
est.fit(X_train, y_train)
est.predict(X_new)
est.feature_importances_       # delegated from the fitted inner estimator
```

The wrapper above is the manual entry point. `train_mlframe_models_suite` also runs automatic composite-target discovery (`CompositeTargetDiscovery` / `CompositeTargetDiscoveryConfig`), which screens transforms, ensembles the survivors, and caches results; set `MLFRAME_DISABLE_COMPOSITE=1` to turn it off. See [docs/examples/composite_targets.md](docs/examples/composite_targets.md) and the [tutorial notebook](docs/composite_targets_tutorial.ipynb).

**Per-target metric panel.** `fast_calibration_report` computes the Brier
reliability / resolution / uncertainty decomposition, ICE bands, ECE, ROC/PR AUC,
and the classification scores in one numba-accelerated pass, returning them as a
`CalibrationReport` `NamedTuple` (and, optionally, the calibration figure). The
result unpacks positionally and indexes exactly like the historical flat tuple, and
also exposes every element as a named attribute (`report.brier_loss`, `report.ece`, ...).

```python
from mlframe.metrics import fast_calibration_report

# Positional unpacking still works (back-compatible with the flat 17-tuple):
(brier_loss, cal_mae, cal_std, cal_coverage,
 ece, brier_rel, brier_res, brier_unc,
 roc_auc, pr_auc, ice, ll, precision, recall, f1,
 metrics_string, fig) = fast_calibration_report(
    y_true=y_test,
    y_pred=clf.predict_proba(X_test)[:, 1],
    nbins=15,
    show_plots=False,
)
print(brier_rel, brier_res, brier_unc, ece)

# Or keep the result and use named access (clearer, index-safe):
report = fast_calibration_report(
    y_true=y_test,
    y_pred=clf.predict_proba(X_test)[:, 1],
    nbins=15,
    show_plots=False,
)
print(report.brier_reliability, report.brier_resolution, report.ece, report.roc_auc)
print(report[0] == report.brier_loss)  # True — positional indexing preserved
```

**MRMR / RFECV feature selection.** Several MRMR variants (FCQ, MID, FCD, plus
n-way interaction extensions) and an RFECV wrapper that reads LightGBM / XGBoost /
CatBoost feature importances correctly. Both are scikit-learn `fit` / `transform`
estimators; the selected columns land on the fitted estimator after `fit(X, y)`.

```python
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.wrappers import RFECV

# X, y are reused by every snippet below in this section.
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(400, 8)), columns=[f"f{i}" for i in range(8)])
y = (X["f0"] + X["f1"] * 0.5 + rng.normal(scale=0.2, size=400) > 0).astype(int)

mrmr = MRMR(max_runtime_mins=1.0).fit(X, y)
X_mrmr = mrmr.transform(X)

rfe = RFECV(estimator=LGBMClassifier()).fit(X, y)
X_rfe = rfe.transform(X)
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
`proxy_mode="interaction"` (opt-in; default `"additive"`) re-scores subsets with the
off-diagonal TreeSHAP interaction values `base + sum phi_j + 2*sum_{i<j} Phi_ij`,
gated to the top-`interaction_proxy_top_k` features by `|phi|` (O(k^2), not O(P^2)), so
a non-additive pair (XOR / multiplicative) earns the joint credit the additive proxy
misses; it stays opt-in because the win does not generalise to additive-only beds.
`ShapProxiedFS.preflight(X, y)` returns a run / caution / fallback recommendation
before a full fit.

```python
# reuses X, y from the MRMR / RFECV example above
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
# reuses X, y from the MRMR / RFECV example above
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
# reuses X, y from the MRMR / RFECV example above
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
import time
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
    time.sleep(0.01)  # stand-in for the real kernel call
    return variant, block

for _ in range(5):
    print(my_kernel(None))  # each call explores or exploits, recording elapsed_s
```

**Time-series and financial feature engineering on Polars.** Windowed aggregation,
ACF, Hurst exponent, TA-Lib indicators, and market-wide rolling features. Most
extraction paths run Polars-native without copying to pandas.

`create_aggregated_features` is the per-window worker: it appends numeric / categorical
aggregates (raw, diffs, ratios, robust, EWMA, rolling, wavelets, ...) computed over the
rows of one window into the caller-supplied `row_features` list (and, when
`create_features_names=True`, the parallel `features_names` list). It mutates those
lists in place and returns `None`.

```python
import numpy as np, pandas as pd
from mlframe.feature_engineering.timeseries import create_aggregated_features

rng = np.random.default_rng(0)
window_df = pd.DataFrame({"price": rng.normal(100, 5, size=20), "volume": rng.integers(100, 1000, size=20)})

row_features: list = []
features_names: list = []
create_aggregated_features(
    window_df=window_df,           # one rolling window of rows
    row_features=row_features,     # appended to in place
    create_features_names=True,
    features_names=features_names, # appended to in place
    dataset_name="prices",
    differences_features=True,
    ratios_features=True,
    ewma_alphas=(0.1, 0.5),
)
print(len(row_features), features_names[:3])
```

`create_ohlcv_wholemarket_features` builds cross-ticker market-wide aggregates
(min / max / std / mean / quantiles plus value-weighted variants) per timestamp on a
Polars OHLCV frame:

```python
import numpy as np, polars as pl
from mlframe.feature_engineering.financial import create_ohlcv_wholemarket_features

rng = np.random.default_rng(0)
dates = np.repeat(np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-06")), 3)
ohlcv = pl.DataFrame({
    "date": dates,
    "ticker": ["AAA", "BBB", "CCC"] * 5,
    "close": rng.normal(100, 5, size=15),
    "volume": rng.integers(1000, 5000, size=15),
})

# The default weighting_columns=("volume", "qty") requires both to be present; pass an
# explicit subset when the frame only has a subset (most OHLCV feeds lack "qty").
market = create_ohlcv_wholemarket_features(ohlcv, timestamp_column="date", weighting_columns=["volume"])
```

**Post-hoc probability calibration.** Compare Venn-Abers, isotonic, Platt, beta, and
per-class isotonic on out-of-fold predictions and pick the calibrator that minimises
OOF ECE (with a bootstrap-CI tiebreak). Selection is OOF-only to keep the estimate
honest; the returned dict carries the chosen calibrator name, its fitted object, and
the per-candidate ECE scores.

```python
import numpy as np
from mlframe.calibration.policy import pick_best_calibrator

rng = np.random.default_rng(0)
oof_proba = np.clip(rng.normal(0.5, 0.25, size=400), 0.001, 0.999)
y_val = (oof_proba + rng.normal(scale=0.15, size=400) > 0.5).astype(int)

result = pick_best_calibrator(
    probs=None, y=None,                 # optional diagnostic-only held-out probs/labels
    oof_probs=oof_proba, oof_y=y_val,   # OOF probs/labels drive the selection
    candidates=["Sigmoid", "Isotonic", "Beta", "Spline"],
    n_bins=15,
)
print(result["chosen"], result["ece_mean"], result["alternatives"])
```

**Inference from saved models.** `read_trained_models` loads a featureset's saved
models from an inference folder (with optional `trusted_root` path-traversal guard and
SHA-256 sidecar verification), returning `(models, X)` aligned to the required feature
order; `get_models_raw_predictions` then evaluates each loaded model on `X`.

> **What's a "sidecar"?** A tiny companion file (`<model file>.sha256`) sitting right next to a saved
> model, holding the SHA-256 hash of that model file's bytes. Before loading a pickled model,
> `mlframe.utils.safe_pickle.safe_load` recomputes the hash of the file on disk and compares it against
> the sidecar — if they don't match (or the sidecar is missing), the load is refused instead of silently
> unpickling a corrupted or unexpectedly-swapped file. Loading arbitrary pickles executes arbitrary code,
> so this catches accidental corruption (truncated copy, crashed mid-write, wrong file dropped in the
> folder) before it turns into a confusing runtime error or a bad prediction. **It is not a defense
> against a malicious attacker**: anyone who can write to the model folder can rewrite the model file
> *and* its sidecar together, and the check will pass. Use `write_sidecar(path)` any time you save a new
> model file so `read_trained_models` / `safe_load` can verify it later.

```python
import os, json, joblib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from mlframe.utils.safe_pickle import write_sidecar
from mlframe.inference.predict import read_trained_models, get_models_raw_predictions

# Build a minimal on-disk featureset (this is what a training run's OutputConfig
# would populate for you): infer/<featureset>/<model>.dump(+.sha256) and features.dump.json.
rng = np.random.default_rng(0)
X = pd.DataFrame({"f0": rng.normal(size=200), "f1": rng.normal(size=200)})
y = (X["f0"] + rng.normal(scale=0.2, size=200) > 0).astype(int)
model = LogisticRegression().fit(X, y)

os.makedirs("infer/my_featureset", exist_ok=True)
joblib.dump(model, "infer/my_featureset/lgb.dump")
write_sidecar("infer/my_featureset/lgb.dump")
json.dump(["f0", "f1"], open("infer/my_featureset/features.dump.json", "w"))

X_new = X.iloc[:5]
models, X_aligned = read_trained_models(
    featureset="my_featureset",
    X=X_new,
    inference_folder="infer",
)
preds = get_models_raw_predictions(models, X_aligned, Y=None)
print(preds)
```

## Suite-level composite feature engineering (opt-in)

`train_mlframe_models_suite` can run eight composite feature-engineering tricks
directly as part of its own preprocessing pass, before categorical encoding —
each is off by default and enabled by setting the relevant fields on
`PreprocessingExtensionsConfig` (passed as `preprocessing_config=`):

- **Categorical composite concat** — `categorical_powerset_concat_enabled` /
  `categorical_group_concat_auto_enabled`: concatenate categorical columns
  (all pairs/subsets, or MI-selected groups) into new joint categorical features.
- **Entity/time state duration & recency aggregation** — `state_duration_columns`,
  `recency_aggregation_columns`: how long an entity has held its current
  state, and poly/exp/power recency-weighted aggregates over its history.
- **Cross-sectional neighbor aggregates** — `cross_sectional_neighbors_snapshot_col`
  / `cross_sectional_neighbors_feature_cols`: per-row summary stats (mean/std/...)
  over the k nearest peers sharing a snapshot key.
- **Two-step target encoding** — `two_step_target_encode_columns`: a
  leakage-safe (train-fit, predict-replay) target encoder with recency-decayed
  smoothing toward a global prior.
- **Moving-average crossover** — `ma_crossover_columns` / `ma_crossover_windows`:
  short/long window moving averages and their crossover signal per entity.
- **Latent interaction SVD** — `latent_interaction_svd_row_entity` /
  `latent_interaction_svd_col_entity`: dense embeddings from the SVD of an
  entity-by-entity co-occurrence/interaction matrix built from a separate
  `auxiliary_events_df`.
- **Nearest-past join** — `nearest_past_join_on` / `nearest_past_join_by`:
  as-of (leakage-safe, most-recent-past-only) enrichment from `auxiliary_events_df`.
- **Event-proximity decay** — `event_proximity_decay_event_dates`: distance-to-nearest-event
  decay features from a fixed list of event dates.

The two auxiliary-table tricks (latent interaction SVD, nearest-past join) read
from a new top-level `auxiliary_events_df: Optional[Union[pd.DataFrame, pl.DataFrame]]`
parameter on `train_mlframe_models_suite` and the predict entry points — a
separate events/entities table with its own row identity, distinct from the
main training frame. Each trick persists what it needs (config, a fitted
entity-lookup, or a fitted SVD embedding object) onto the trained bundle's
`metadata` and replays identically at predict time; pass a fresh
`auxiliary_events_df` at predict time to pick up new entities/events without
refitting.

## Also worth knowing about

Smaller, well-tested primitives that don't need a full walkthrough but are worth
knowing exist — each has a docstring, unit tests, and (where relevant) a
quantitative business-value test under `tests/`:

- **`mlframe.models.rf_proximity.rf_proximity_matrix` / `rf_outlier_measure`** —
  Breiman's random-forest proximity (fraction of trees where two rows share a
  leaf) as a reusable N×N similarity/distance metric plus an outlier score,
  computed from any fitted forest's leaf indices — numba-accelerated, memory-guarded.
- **`mlframe.core.matrix_seriation.seriate`** — reorders a correlation/similarity
  matrix by spectral score (Fiedler vector or leading SVD vector) so correlated
  feature blocks become visually contiguous instead of scattered across an
  unreadable `df.corr()` heatmap; doubles as a feature-clustering primitive.
- **`mlframe.core.composite_similarity.fit_composite_similarity`** — Dyakonov's
  LENKOR technique (1st place, ECML-PKDD 2011 Discovery Challenge): coordinate-descent-tunes
  weights to blend several precomputed per-attribute-block similarities (authors,
  category, co-view counts, ...) into one learned metric, for tasks where how to
  compare the whole is unclear but how to compare its parts is.
- **`mlframe.evaluation.group_leakage_guard.assert_no_group_leakage`** — a
  runtime assertion that no group/entity ID appears on both sides of a CV
  split, plus near-duplicate-feature detection across fold boundaries for the
  implicit leaks an explicit group column can't catch.
- **`mlframe.evaluation.AdversarialValidator`** — unifies adversarial
  train/test-shift AUC, per-feature drift importance, and test-like
  validation-fold selection (picking train rows most similar to the true test
  distribution) into one object.
- **`mlframe.votenrank`** — ensemble-blending strategies beyond stack/blend/vote:
  confidence-gated blending (mix in an auxiliary model only where it's
  confident, not by a fixed weight), adversarial-stochastic blend, rank-splice,
  KNN-fallback, and geometric/correlation-diversity-aware blends.
- **`mlframe.signal.gp_smoothing.compute_gp_smoothed_features`** — a Gaussian-Process
  front end for irregularly-sampled time series (the PLAsTiCC-winning technique):
  fits a Matern-kernel GP per series and extracts both the smoothed value and the
  posterior standard deviation as a built-in local-data-density feature.
- **`mlframe.core.robust_location`** — redescending M-estimator mean (Meshalkin /
  Huber / Tukey-biweight weighting) and the geometric median (Weiszfeld
  iteration) for outlier-robust aggregation where the plain mean is too
  sensitive and the coordinate-wise median is ill-defined in >1D.
- **`mlframe.testing.parametric`** — a thin, mlframe-tuned wrapper around
  `polars.testing.parametric` that generates test frames hitting the dtype /
  nullability shapes that actually crash CatBoost/XGBoost/LightGBM in
  production (nulls inside `pl.Categorical`, all-null high-cardinality text
  columns, constant/inf/NaN numeric columns), rather than hand-picked happy-path fixtures.

## Visualization & Diagnostics

`train_mlframe_models_suite` emits a task-appropriate set of diagnostic charts
whenever `output_config.data_dir` is set (charts land under
`<data_dir>/charts/...`; a run with no `data_dir` computes metrics but saves no
figures and logs a one-line hint so the absence is never silent). Everything
below is **default-ON** — you remove tokens / flip flags to opt *out*. All of it
is configured through `ReportingConfig`. Full reference:
[docs/visualization.md](docs/visualization.md).

**What renders per task type** (the default panel templates):

| Task type | Default panels (`ReportingConfig` knob) |
| --- | --- |
| Binary | `ROC PR SCORE_DIST KS THRESHOLD GAIN PIT` (`binary_panels`) |
| Multiclass | `CONFUSION CONFUSION_MARGINS CONFUSED_PAIRS PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC` (`multiclass_panels`) |
| Multilabel | `PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST THRESHOLD_SWEEP` (`multilabel_panels`) |
| LTR | `NDCG_K NDCG_DIST NDCG_BY_QSIZE LIFT MRR_DIST SCORE_BY_REL` (`ltr_panels`) |
| Quantile | `RELIABILITY COVERAGE PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST QUANTILE_RELIABILITY PINBALL_DECOMP QUANTILE_CROSSING FAN_CHART` (`quantile_panels`) |
| Regression | `SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE WORM RESID_ACF` (`regression_panels`) |

A sample of the rendered panels (full gallery: [docs/gallery](docs/gallery/index.md), regenerated with `python scripts/render_gallery.py`):

| | |
| --- | --- |
| Binary classification (`ROC PR SCORE_DIST KS THRESHOLD GAIN PIT`) | Regression (scatter, residuals, error-by-decile, worm, ACF) |
| ![binary_full](docs/gallery/binary/binary_full.png) | ![regression_full](docs/gallery/regression/regression_full.png) |
| PSI drift heatmap (feature × time) | Calibration reliability (Venn-Abers / isotonic / Platt / beta) |
| ![psi_heatmap](docs/gallery/drift/psi_heatmap.png) | ![calibration_reliability](docs/gallery/binary/calibration_reliability.png) |
| SHAP beeswarm | Model comparison across metrics |
| ![shap_beeswarm](docs/gallery/shap_panels/shap_shap_beeswarm.png) | ![model_comparison](docs/gallery/model_comparison/model_comparison.png) |

**New diagnostics this brings.** Binary classification gained the full curve set
it previously lacked (ROC / PR / score-distribution / KS / threshold-sweep /
cumulative-gain / PIT). Beyond the per-task panels the suite also renders, when
charts are being saved: a target/prediction **distribution overlay** per split
(incl. OOF-vs-test), a tree-guided **weak-segment error heatmap**,
**error-bias-per-feature** (OVER/UNDER/MAJORITY tails), a **worst-K errors**
table with the same points red-highlighted on the scatter, a **PSI drift
heatmap** (feature × time, 0.10/0.25 triage), **adversarial validation**
(train-vs-test/val LightGBM AUC + drifting-feature bars — "will my CV
transfer?"), **residual / metric over time**, per-model **training curves**
(train vs val metric per boosting iteration with the early-stop point marked),
and the quantile **reliability / coverage / pinball-by-alpha / PIT / crossing**
diagnostics including a CORP pinball decomposition.

**Large-n behavior.** The charts stay cheap on multi-million-row frames with no
pre-subsampling: the regression scatter switches to a log-density hexbin/hist2d
above 50k points (raw scatter with an extremes-preserving subsample below it, so
the MaxError point is always drawn); plotly scatters use `Scattergl` (WebGL)
above 10k and decimate above 50k; histograms are numpy-prebinned at ≥50k
(2M raw values → ~37MB HTML drops to ~14KB); curves are vertex-decimated to
~2000 points; violins/KDE subsample to 5000; PSI / overlays / over-time panels
are aggregate-first (one O(n) pass per feature).

**Output DSL.** `ReportingConfig.plot_outputs` is a backend×format DSL, default
`"plotly[html] + matplotlib[png]"` — interactive HTML for sharing plus a static
PNG from matplotlib (plotly PNG via kaleido spends ~12-15s/figure on a Chromium
reload, so the fast matplotlib path is the default; use `"plotly[html,png]"` to
force kaleido). Grammar: `<backend>[<fmt>,...] + <backend>[<fmt>,...]`.

**Panel templates** are space-separated token strings validated at config
construction against each chart module's allowed-token set, so a typo fails
before training starts. Set any subset (or `""` to skip a task's panels).

**Key knobs:** `binary_panels` / `multiclass_panels` / `multilabel_panels` /
`ltr_panels` / `quantile_panels` / `regression_panels` (panel templates),
`regression_scatter_sample_size` (5000), `calibration_binning`
(`auto`/`uniform`/`quantile`), `reliability_show_ci` (Wilson CI band, on),
`training_curves` (on), `keep_figure_handles` (retain pure-data `FigureSpec`s
in `metrics["figure_specs"]` for programmatic re-render; chart paths are always
in `metrics["charts"]`).

**Discovery.** `from mlframe.reporting import describe_available_panels;
describe_available_panels()` prints every token per task type with a one-line
description (and returns the same mapping for programmatic use).

## Caching strategy

The training suite runs two caching layers with different key strategies, because
the lifetime of the cached value, the cost of computing the key, and the failure
mode of a stale hit differ between them.

**Layer 1: `_PRE_PIPELINE_CACHE` (content-keyed).** Caches the output of
`(SimpleImputer + StandardScaler + feature selectors).fit_transform(train_df, val_df)`
so consecutive models in one suite call that share the same pre-pipeline structure
reuse the fitted transforms. Keys come from content fingerprints of `train_df`,
`val_df`, and the target array, the pipeline signature, the target name, and
optionally sample weights. Consecutive lookups see the same Python objects (same `id()`) but the value
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

## Environment variables

Every environment variable read anywhere in `src/mlframe/` via `os.environ.get(...)` / `os.getenv(...)`, generated from the source (name, first read site, and its literal default if one is passed inline). This is a mechanically-generated inventory, not a hand-written guide -- it documents *that* a var is read and its default, not *why* it exists or what it tunes; see the linked file for that.

| Variable | Default | First read at |
|---|---|---|
| `BENCH_DEBUG_CACHE` | — | [src/mlframe/training/_benchmarks/bench_per_target_hoist.py](src/mlframe/training/_benchmarks/bench_per_target_hoist.py#L121) |
| `COMPUTERNAME` | `os.environ.get('HOSTNAME', '?')` | [src/mlframe/training/composite/discovery/_benchmarks/bench_mi_from_binned_pair_njit.py](src/mlframe/training/composite/discovery/_benchmarks/bench_mi_from_binned_pair_njit.py#L79) |
| `CUDA_HOME` | — | [src/mlframe/__init__.py](src/mlframe/__init__.py#L82) |
| `CUDA_PATH` | — | [src/mlframe/__init__.py](src/mlframe/__init__.py#L84) |
| `CUDA_VISIBLE_DEVICES` | — | [src/mlframe/training/cb/_cb_pool.py](src/mlframe/training/cb/_cb_pool.py#L601) |
| `FE_ACCEPT_SKIP_MADELON` | `'0'` | [src/mlframe/feature_selection/_benchmarks/fs_hybrid/round4_fe_accept_bench.py](src/mlframe/feature_selection/_benchmarks/fs_hybrid/round4_fe_accept_bench.py#L264) |
| `FS` | `'rfecv'` | [src/mlframe/feature_selection/_benchmarks/fs_hybrid/round4_fs_campaign_profile.py](src/mlframe/feature_selection/_benchmarks/fs_hybrid/round4_fs_campaign_profile.py#L16) |
| `FS_HYBRID_SCENARIOS` | `'default'` | [src/mlframe/feature_selection/_benchmarks/fs_hybrid/run_experiment.py](src/mlframe/feature_selection/_benchmarks/fs_hybrid/run_experiment.py#L49) |
| `FULL_P_SWEEP` | — | [src/mlframe/feature_selection/_benchmarks/fs_quality/mrmr_largeN_campaign.py](src/mlframe/feature_selection/_benchmarks/fs_quality/mrmr_largeN_campaign.py#L280) |
| `HF_HOME` | — | [src/mlframe/training/feature_handling/hf_provider.py](src/mlframe/training/feature_handling/hf_provider.py#L84) |
| `HOSTNAME` | `'?'` | [src/mlframe/training/composite/discovery/_benchmarks/bench_mi_from_binned_pair_njit.py](src/mlframe/training/composite/discovery/_benchmarks/bench_mi_from_binned_pair_njit.py#L79) |
| `HYB_N` | `'1500'` | [src/mlframe/feature_selection/_benchmarks/fs_hybrid/hybrid_opt_baseline.py](src/mlframe/feature_selection/_benchmarks/fs_hybrid/hybrid_opt_baseline.py#L25) |
| `HYB_SEED` | `'0'` | [src/mlframe/feature_selection/_benchmarks/fs_hybrid/hybrid_opt_baseline.py](src/mlframe/feature_selection/_benchmarks/fs_hybrid/hybrid_opt_baseline.py#L26) |
| `ITER100_OUT_DIR` | `'D:/Temp'` | [src/mlframe/feature_selection/_benchmarks/bench_iter100_stratified_anchors_contour.py](src/mlframe/feature_selection/_benchmarks/bench_iter100_stratified_anchors_contour.py#L137) |
| `ITER101_OUT_DIR` | `'D:/Temp'` | [src/mlframe/feature_selection/_benchmarks/bench_iter101_stratified_anchors_2d.py](src/mlframe/feature_selection/_benchmarks/bench_iter101_stratified_anchors_2d.py#L157) |
| `ITER87_BASELINE_WT` | `'D:/Temp/iter87_baseline_wt'` | [src/mlframe/feature_selection/_benchmarks/bench_iter87_cumulative.py](src/mlframe/feature_selection/_benchmarks/bench_iter87_cumulative.py#L159) |
| `ITER87_RESULTS` | `'D:/Temp/iter87_results.json'` | [src/mlframe/feature_selection/_benchmarks/bench_iter87_cumulative.py](src/mlframe/feature_selection/_benchmarks/bench_iter87_cumulative.py#L209) |
| `JULIA_NUM_THREADS` | `'?'` | [src/mlframe/training/_benchmarks/bench_pysr_fe.py](src/mlframe/training/_benchmarks/bench_pysr_fe.py#L184) |
| `LOKY_MAX_CPU_COUNT` | — | [src/mlframe/training/__init__.py](src/mlframe/training/__init__.py#L93) |
| `MDL_OLD_BASELINE` | `''` | [src/mlframe/feature_engineering/_benchmarks/bench_mdl_binning_split_iter81.py](src/mlframe/feature_engineering/_benchmarks/bench_mdl_binning_split_iter81.py#L30) |
| `MLFRAME_BATCH_MI_KERNEL` | `''` | [src/mlframe/feature_selection/filters/info_theory/_batch_kernels.py](src/mlframe/feature_selection/filters/info_theory/_batch_kernels.py#L861) |
| `MLFRAME_BOOTSTRAP_BACKEND` | `'threading'` | [src/mlframe/evaluation/bootstrap.py](src/mlframe/evaluation/bootstrap.py#L475) |
| `MLFRAME_BORUTA_AUTO_NP_RATIO` | `'30.0'` | [src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py](src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py#L47) |
| `MLFRAME_BORUTA_AUTO_OOB_GAP` | `'0.25'` | [src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py](src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py#L48) |
| `MLFRAME_BORUTA_AUTO_PROBE_ROWS` | `'2000'` | [src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py](src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py#L51) |
| `MLFRAME_BORUTA_AUTO_PROBE_TREES` | `'80'` | [src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py](src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py#L50) |
| `MLFRAME_BORUTA_SHADOW_TIE_GATE` | `'0.20'` | [src/mlframe/feature_selection/boruta_shap/_shadow_stats.py](src/mlframe/feature_selection/boruta_shap/_shadow_stats.py#L86) |
| `MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES` | — | [src/mlframe/calibration/policy.py](src/mlframe/calibration/policy.py#L56) |
| `MLFRAME_CALIB_BINNING_PRANGE_THRESHOLD` | `'2000000'` | [src/mlframe/metrics/calibration/_calibration_plot.py](src/mlframe/metrics/calibration/_calibration_plot.py#L188) |
| `MLFRAME_CAT_DIAG` | — | [src/mlframe/training/_eval_helpers.py](src/mlframe/training/_eval_helpers.py#L74) |
| `MLFRAME_CAT_FE_BENCH_PROD` | — | [src/mlframe/feature_selection/_benchmarks/bench_categorical_fe.py](src/mlframe/feature_selection/_benchmarks/bench_categorical_fe.py#L136) |
| `MLFRAME_CMI_ANALYTIC_NULL_MIN_N` | `''` | [src/mlframe/feature_selection/filters/_fe_cmi_redundancy_null.py](src/mlframe/feature_selection/filters/_fe_cmi_redundancy_null.py#L27) |
| `MLFRAME_CMI_FORDER` | `'1'` | [src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py](src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py#L89) |
| `MLFRAME_CMI_FORDER_MAX_MB` | `'4096'` | [src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py](src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py#L82) |
| `MLFRAME_CMI_GPU` | `''` | [src/mlframe/feature_selection/filters/_mi_greedy_cmi_fe.py](src/mlframe/feature_selection/filters/_mi_greedy_cmi_fe.py#L967) |
| `MLFRAME_CMI_NULL_MAX_ROWS` | `'100000'` | [src/mlframe/feature_selection/filters/_fe_raw_redundancy_helpers.py](src/mlframe/feature_selection/filters/_fe_raw_redundancy_helpers.py#L30) |
| `MLFRAME_CMI_PARALLEL_MIN_CANDS` | `'8'` | [src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py](src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py#L63) |
| `MLFRAME_CMI_RESIDENT_CACHE` | `'1'` | [src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py](src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py#L361) |
| `MLFRAME_CMI_XC_RESIDENT` | `'1'` | [src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py](src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py#L420) |
| `MLFRAME_CMI_YZ_HOIST` | `'1'` | [src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py](src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py#L713) |
| `MLFRAME_CONFIDENCE_BLEND_BACKEND` | `''` | [src/mlframe/votenrank/confidence_gated_blend.py](src/mlframe/votenrank/confidence_gated_blend.py#L176) |
| `MLFRAME_CRIT_DTYPE_RELAXED` | `'1'` | [src/mlframe/feature_selection/filters/_fe_usability_signal.py](src/mlframe/feature_selection/filters/_fe_usability_signal.py#L150) |
| `MLFRAME_CTX_STRICT` | `''` | [src/mlframe/training/core/_misc_helpers.py](src/mlframe/training/core/_misc_helpers.py#L949) |
| `MLFRAME_CUDA_GRAPH_PREDICT` | `'0'` | [src/mlframe/training/neural/_flat_torch_module/_flat_torch_predict_accel.py](src/mlframe/training/neural/_flat_torch_module/_flat_torch_predict_accel.py#L276) |
| `MLFRAME_CUDA_GRAPH_PREDICT_CACHE_MAX` | `'16'` | [src/mlframe/training/neural/_flat_torch_module/_flat_torch_predict_accel.py](src/mlframe/training/neural/_flat_torch_module/_flat_torch_predict_accel.py#L31) |
| `MLFRAME_CYCLICAL_PAR_THRESHOLD` | `'1000000'` | [src/mlframe/feature_engineering/basic.py](src/mlframe/feature_engineering/basic.py#L39) |
| `MLFRAME_DETECT_HEAVY_TAIL_NJIT_MAX_N` | `'3000'` | [src/mlframe/feature_selection/filters/hermite_fe/_hermite_robust.py](src/mlframe/feature_selection/filters/hermite_fe/_hermite_robust.py#L142) |
| `MLFRAME_DISABLE_BATCHED_PREPROCESS_SCAN` | — | [src/mlframe/training/preprocessing.py](src/mlframe/training/preprocessing.py#L373) |
| `MLFRAME_DISABLE_COMPOSITE` | `''` | [src/mlframe/training/core/_phase_config_setup.py](src/mlframe/training/core/_phase_config_setup.py#L241) |
| `MLFRAME_DISABLE_GPU` | `''` | [src/mlframe/feature_selection/filters/_confirm_predictor.py](src/mlframe/feature_selection/filters/_confirm_predictor.py#L450) |
| `MLFRAME_DISABLE_HNSW` | `''` | [src/mlframe/feature_engineering/transformer/_knn_helper.py](src/mlframe/feature_engineering/transformer/_knn_helper.py#L49) |
| `MLFRAME_DISCOVERY_CACHE_MAX_BYTES` | — | [src/mlframe/training/composite/cache_store.py](src/mlframe/training/composite/cache_store.py#L316) |
| `MLFRAME_DISCOVERY_CACHE_STRICT` | `''` | [src/mlframe/training/composite/cache_store.py](src/mlframe/training/composite/cache_store.py#L340) |
| `MLFRAME_DISCOVERY_CACHE_TMP_AGE_S` | — | [src/mlframe/training/composite/cache_store.py](src/mlframe/training/composite/cache_store.py#L443) |
| `MLFRAME_DISCOVERY_LAZY_PREBIN` | `''` | [src/mlframe/training/composite/discovery/_fit.py](src/mlframe/training/composite/discovery/_fit.py#L313) |
| `MLFRAME_DISCOVERY_LAZY_PREBIN_MIN_N` | `'50000'` | [src/mlframe/training/composite/discovery/_fit.py](src/mlframe/training/composite/discovery/_fit.py#L314) |
| `MLFRAME_DISCOVERY_RAM_PROFILER` | `'1'` | [src/mlframe/training/composite/discovery/_fit.py](src/mlframe/training/composite/discovery/_fit.py#L169) |
| `MLFRAME_DISCOVERY_SKIP_TINY_RERANK` | `''` | [src/mlframe/training/composite/discovery/_tiny_rerank.py](src/mlframe/training/composite/discovery/_tiny_rerank.py#L89) |
| `MLFRAME_DISCRETIZE_COL_CACHE` | `'1'` | [src/mlframe/feature_selection/filters/discretization/_discretization_dataset.py](src/mlframe/feature_selection/filters/discretization/_discretization_dataset.py#L102) |
| `MLFRAME_DISCRETIZE_COL_CACHE_MAX_BYTES` | `str(512 * 1024 * 1024)` | [src/mlframe/feature_selection/filters/discretization/_discretization_dataset.py](src/mlframe/feature_selection/filters/discretization/_discretization_dataset.py#L67) |
| `MLFRAME_DISCRETIZE_FLOAT32` | `''` | [src/mlframe/feature_selection/filters/discretization/_discretization_dataset.py](src/mlframe/feature_selection/filters/discretization/_discretization_dataset.py#L46) |
| `MLFRAME_DISCRETIZE_UNIFORM_PAR_THRESHOLD` | `'50000'` | [src/mlframe/feature_selection/filters/discretization/__init__.py](src/mlframe/feature_selection/filters/discretization/__init__.py#L556) |
| `MLFRAME_DTW_AUTOTUNE` | `'1'` | [src/mlframe/signal/dtw.py](src/mlframe/signal/dtw.py#L535) |
| `MLFRAME_EWMA_BACKEND` | `''` | [src/mlframe/training/composite/transforms/nonlinear.py](src/mlframe/training/composite/transforms/nonlinear.py#L618) |
| `MLFRAME_EXTENSIONS_SOFT_FAIL` | `''` | [src/mlframe/training/core/_predict_pre_pipeline.py](src/mlframe/training/core/_predict_pre_pipeline.py#L116) |
| `MLFRAME_FDR_NULL_INT32` | `''` | [src/mlframe/feature_selection/filters/_permutation_null.py](src/mlframe/feature_selection/filters/_permutation_null.py#L362) |
| `MLFRAME_FDR_SHUFFLEGEN` | `''` | [src/mlframe/feature_selection/filters/_permutation_null_shufflegen_ktc.py](src/mlframe/feature_selection/filters/_permutation_null_shufflegen_ktc.py#L45) |
| `MLFRAME_FEATURE_CACHE_STRICT` | `''` | [src/mlframe/training/feature_handling/cache.py](src/mlframe/training/feature_handling/cache.py#L567) |
| `MLFRAME_FE_BUDGET_CACHE_DIR` | `str(Path.home() / '.cache' / 'mlframe' / 'fe_family_budget')` | [src/mlframe/feature_selection/filters/_fe_family_budget.py](src/mlframe/feature_selection/filters/_fe_family_budget.py#L40) |
| `MLFRAME_FE_BUFFER_MAX_GB` | — | [src/mlframe/feature_selection/filters/_feature_engineering_mem_budget.py](src/mlframe/feature_selection/filters/_feature_engineering_mem_budget.py#L235) |
| `MLFRAME_FE_CHEAP_MI_GPU` | `'1'` | [src/mlframe/feature_selection/filters/_binned_numeric_agg_fe.py](src/mlframe/feature_selection/filters/_binned_numeric_agg_fe.py#L334) |
| `MLFRAME_FE_CMI_PERM_NULL_GPU` | `'1'` | [src/mlframe/feature_selection/filters/_fe_cmi_perm_null_gpu.py](src/mlframe/feature_selection/filters/_fe_cmi_perm_null_gpu.py#L90) |
| `MLFRAME_FE_CMI_PERM_NULL_SPARSE` | `'0'` | [src/mlframe/feature_selection/filters/_fe_cmi_perm_null_gpu.py](src/mlframe/feature_selection/filters/_fe_cmi_perm_null_gpu.py#L293) |
| `MLFRAME_FE_DEDUP_CORR_BACKEND` | `''` | [src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_dedup.py](src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_dedup.py#L208) |
| `MLFRAME_FE_DEDUP_MAX_CORR_ROWS` | `'100000'` | [src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_dedup.py](src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_dedup.py#L26) |
| `MLFRAME_FE_DROP_NO_HARM` | `'1'` | [src/mlframe/feature_selection/filters/_fe_raw_redundancy_drop.py](src/mlframe/feature_selection/filters/_fe_raw_redundancy_drop.py#L622) |
| `MLFRAME_FE_EDGE_BINNING` | `''` | [src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_mi_backends.py](src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_mi_backends.py#L73) |
| `MLFRAME_FE_FUSION_AB` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_resident_materialise.py](src/mlframe/feature_selection/filters/_gpu_resident_materialise.py#L810) |
| `MLFRAME_FE_FUSION_MAX_ROWS` | `'250000'` | [src/mlframe/feature_selection/filters/_fe_additive_fusion_gpu_resident.py](src/mlframe/feature_selection/filters/_fe_additive_fusion_gpu_resident.py#L199) |
| `MLFRAME_FE_GATE_MAX_ROWS` | `'250000'` | [src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_score.py](src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_score.py#L127) |
| `MLFRAME_FE_GATE_RESIDENT_CANDS` | `'1'` | [src/mlframe/feature_selection/filters/_conditional_gate_fe.py](src/mlframe/feature_selection/filters/_conditional_gate_fe.py#L270) |
| `MLFRAME_FE_GPU_BINNING` | `'auto'` | [src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py](src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py#L214) |
| `MLFRAME_FE_GPU_BINNING_DTYPE` | `''` | [src/mlframe/feature_selection/filters/_gpu_resident_basis.py](src/mlframe/feature_selection/filters/_gpu_resident_basis.py#L1443) |
| `MLFRAME_FE_GPU_BINNING_MIN_NK` | `'1000000'` | [src/mlframe/feature_selection/filters/_gpu_resident_basis.py](src/mlframe/feature_selection/filters/_gpu_resident_basis.py#L1093) |
| `MLFRAME_FE_GPU_DEFER_FLOAT` | `'1'` | [src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py](src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py#L1031) |
| `MLFRAME_FE_GPU_DEFER_HOST_CODES` | `''` | [src/mlframe/feature_selection/filters/_gpu_resident_fe.py](src/mlframe/feature_selection/filters/_gpu_resident_fe.py#L274) |
| `MLFRAME_FE_GPU_DEVICE_BORN_BINAGG` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L95) |
| `MLFRAME_FE_GPU_DEVICE_BORN_CROSSBASIS` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L141) |
| `MLFRAME_FE_GPU_DEVICE_BORN_DISPERSION` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L117) |
| `MLFRAME_FE_GPU_DEVICE_BORN_DUAL_UPLIFT` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L168) |
| `MLFRAME_FE_GPU_DEVICE_BORN_EXTRA_BASIS` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L253) |
| `MLFRAME_FE_GPU_DEVICE_BORN_GATE` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L75) |
| `MLFRAME_FE_GPU_DEVICE_BORN_MODULAR` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L272) |
| `MLFRAME_FE_GPU_DEVICE_BORN_UPLIFT_UNIVARIATE` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L231) |
| `MLFRAME_FE_GPU_DEVICE_BORN_WAVELET` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L190) |
| `MLFRAME_FE_GPU_DISCRETIZE` | `'auto'` | [src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py](src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py#L159) |
| `MLFRAME_FE_GPU_DISCRETIZE_MIN_NK` | `'2000000'` | [src/mlframe/feature_selection/filters/_gpu_resident_basis.py](src/mlframe/feature_selection/filters/_gpu_resident_basis.py#L956) |
| `MLFRAME_FE_GPU_FUSE_CMI_ENTROPY` | `'0'` | [src/mlframe/feature_selection/filters/_fe_batched_mi_cmi.py](src/mlframe/feature_selection/filters/_fe_batched_mi_cmi.py#L444) |
| `MLFRAME_FE_GPU_FUSE_MI` | `'1'` | [src/mlframe/feature_selection/filters/_hermite_fe_mi.py](src/mlframe/feature_selection/filters/_hermite_fe_mi.py#L254) |
| `MLFRAME_FE_GPU_GRAND_FUSION` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_resident_fe.py](src/mlframe/feature_selection/filters/_gpu_resident_fe.py#L852) |
| `MLFRAME_FE_GPU_HISTGATE_CM` | `'1'` | [src/mlframe/feature_selection/filters/batch_mi_noise_gate_gpu.py](src/mlframe/feature_selection/filters/batch_mi_noise_gate_gpu.py#L630) |
| `MLFRAME_FE_GPU_MATERIALISE` | `'1'` | [src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py](src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py#L1138) |
| `MLFRAME_FE_GPU_MATERIALISE_CM` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_resident_materialise.py](src/mlframe/feature_selection/filters/_gpu_resident_materialise.py#L188) |
| `MLFRAME_FE_GPU_MIN_FREE_MB` | `_DEFAULT_MIN_FREE_MB` | [src/mlframe/feature_selection/filters/_fe_gpu_vram.py](src/mlframe/feature_selection/filters/_fe_gpu_vram.py#L55) |
| `MLFRAME_FE_GPU_MIN_FREE_VRAM_MB` | `'1024'` | [src/mlframe/feature_selection/filters/_gpu_resident_fe.py](src/mlframe/feature_selection/filters/_gpu_resident_fe.py#L156) |
| `MLFRAME_FE_GPU_POOL_FRACTION` | `_DEFAULT_POOL_FRACTION` | [src/mlframe/feature_selection/filters/_fe_gpu_vram.py](src/mlframe/feature_selection/filters/_fe_gpu_vram.py#L63) |
| `MLFRAME_FE_GPU_RADIX_EDGES` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_resident_select_kernels.py](src/mlframe/feature_selection/filters/_gpu_resident_select_kernels.py#L841) |
| `MLFRAME_FE_GPU_RESIDENT` | `''` | [src/mlframe/feature_selection/filters/_gpu_resident_fe.py](src/mlframe/feature_selection/filters/_gpu_resident_fe.py#L133) |
| `MLFRAME_FE_GPU_RESIDENT_GATE` | `'1'` | [src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_dispatch.py](src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_dispatch.py#L29) |
| `MLFRAME_FE_GPU_RESIDENT_OPERANDS` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_resident_materialise.py](src/mlframe/feature_selection/filters/_gpu_resident_materialise.py#L375) |
| `MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L209) |
| `MLFRAME_FE_GPU_STRICT` | `''` | [src/mlframe/feature_selection/filters/_fe_batch_dispatch.py](src/mlframe/feature_selection/filters/_fe_batch_dispatch.py#L61) |
| `MLFRAME_FE_GPU_STRICT_AUTO_MIN_N` | `_DEFAULT_AUTO_MIN_N` | [src/mlframe/feature_selection/filters/_fe_gpu_strict.py](src/mlframe/feature_selection/filters/_fe_gpu_strict.py#L142) |
| `MLFRAME_FE_GPU_STRICT_BYTEMATCH` | `''` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L58) |
| `MLFRAME_FE_GPU_STRICT_RESIDENT` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py](src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py#L40) |
| `MLFRAME_FE_GPU_USABILITY` | `''` | [src/mlframe/feature_selection/filters/_usability_gpu.py](src/mlframe/feature_selection/filters/_usability_gpu.py#L57) |
| `MLFRAME_FE_HINGE_GPU` | `'1'` | [src/mlframe/feature_selection/filters/_hinge_detect_gpu_resident.py](src/mlframe/feature_selection/filters/_hinge_detect_gpu_resident.py#L38) |
| `MLFRAME_FE_HOIST_HEADROOM_OVERHEAD` | — | [src/mlframe/feature_selection/filters/_feature_engineering_mem_budget.py](src/mlframe/feature_selection/filters/_feature_engineering_mem_budget.py#L166) |
| `MLFRAME_FE_IMBALANCE_MI` | `''` | [src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_imbalance_mi.py](src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_imbalance_mi.py#L102) |
| `MLFRAME_FE_IMBALANCE_N_RARE` | `'150'` | [src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_imbalance_mi.py](src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_imbalance_mi.py#L90) |
| `MLFRAME_FE_IMBALANCE_PRIOR` | `'0.30'` | [src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_imbalance_mi.py](src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_imbalance_mi.py#L89) |
| `MLFRAME_FE_MATRIX_P0` | `''` | [src/mlframe/feature_selection/filters/_fe_matrix_io.py](src/mlframe/feature_selection/filters/_fe_matrix_io.py#L52) |
| `MLFRAME_FE_MIN_FREE_RAM_GB` | — | [src/mlframe/feature_selection/filters/_feature_engineering_mem_budget.py](src/mlframe/feature_selection/filters/_feature_engineering_mem_budget.py#L201) |
| `MLFRAME_FE_MI_SPLIT` | `''` | [src/mlframe/feature_selection/_benchmarks/kernel_tuning_cache/dispatch.py](src/mlframe/feature_selection/_benchmarks/kernel_tuning_cache/dispatch.py#L330) |
| `MLFRAME_FE_NOISE_FLOOR_MAX_ROWS` | `'30000'` | [src/mlframe/feature_selection/filters/_hermite_fe_optimise_pair.py](src/mlframe/feature_selection/filters/_hermite_fe_optimise_pair.py#L786) |
| `MLFRAME_FE_PAIR_MAXT_MAX_ROWS` | `''` | [src/mlframe/feature_selection/filters/_permutation_null.py](src/mlframe/feature_selection/filters/_permutation_null.py#L66) |
| `MLFRAME_FE_PAIR_MAXT_PERM_NULL_GPU` | `'1'` | [src/mlframe/feature_selection/filters/_permutation_null_pair_resident.py](src/mlframe/feature_selection/filters/_permutation_null_pair_resident.py#L235) |
| `MLFRAME_FE_PIPELINE_CHUNKS` | `'1'` | [src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py](src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py#L1094) |
| `MLFRAME_FE_RECURSION_BACKEND` | `''` | [src/mlframe/feature_engineering/_recursion_dispatch.py](src/mlframe/feature_engineering/_recursion_dispatch.py#L53) |
| `MLFRAME_FE_RESIDENT_OPERANDS` | `'1'` | [src/mlframe/feature_selection/filters/_fe_resident_operands.py](src/mlframe/feature_selection/filters/_fe_resident_operands.py#L123) |
| `MLFRAME_FE_RUNG_KEEP_FRAC` | `''` | [src/mlframe/feature_selection/filters/_fe_rung_schedule.py](src/mlframe/feature_selection/filters/_fe_rung_schedule.py#L144) |
| `MLFRAME_FE_VRAM_BACKEND` | `''` | [src/mlframe/feature_selection/filters/_fe_batch_dispatch.py](src/mlframe/feature_selection/filters/_fe_batch_dispatch.py#L58) |
| `MLFRAME_FE_VRAM_BLOCKS_PER_DEVICE` | `'4'` | [src/mlframe/feature_selection/filters/_fe_gpu_batch/_executor.py](src/mlframe/feature_selection/filters/_fe_gpu_batch/_executor.py#L30) |
| `MLFRAME_FE_VRAM_CPSAT_TIME_LIMIT_S` | `'1.0'` | [src/mlframe/feature_selection/filters/_fe_gpu_batch/_packer.py](src/mlframe/feature_selection/filters/_fe_gpu_batch/_packer.py#L19) |
| `MLFRAME_FE_VRAM_CUSHION_FLOOR_BYTES` | `str(128 * 1024 * 1024)` | [src/mlframe/feature_selection/filters/_fe_gpu_batch/_devices.py](src/mlframe/feature_selection/filters/_fe_gpu_batch/_devices.py#L29) |
| `MLFRAME_FE_VRAM_CUSHION_FRAC` | `'0.10'` | [src/mlframe/feature_selection/filters/_fe_gpu_batch/_devices.py](src/mlframe/feature_selection/filters/_fe_gpu_batch/_devices.py#L28) |
| `MLFRAME_FE_VRAM_DEVICES` | `''` | [src/mlframe/feature_selection/filters/_fe_gpu_batch/_devices.py](src/mlframe/feature_selection/filters/_fe_gpu_batch/_devices.py#L110) |
| `MLFRAME_FE_VRAM_F32` | `''` | [src/mlframe/feature_selection/filters/_fe_gpu_batch/_devices.py](src/mlframe/feature_selection/filters/_fe_gpu_batch/_devices.py#L42) |
| `MLFRAME_FE_VRAM_PACKER` | `''` | [src/mlframe/feature_selection/filters/_fe_gpu_batch/_packer.py](src/mlframe/feature_selection/filters/_fe_gpu_batch/_packer.py#L85) |
| `MLFRAME_FORCED_GC_MIN_DF_MB` | `'256'` | [src/mlframe/training/core/_phase_helpers.py](src/mlframe/training/core/_phase_helpers.py#L43) |
| `MLFRAME_FORCE_HELPER_PD_VIEW` | — | [src/mlframe/training/utils.py](src/mlframe/training/utils.py#L545) |
| `MLFRAME_FORCE_REPROBE` | — | [src/mlframe/training/mlp_runtime_defaults.py](src/mlframe/training/mlp_runtime_defaults.py#L350) |
| `MLFRAME_FOURIER_COARSE_BASIS_EXACT` | `''` | [src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_extra_basis_fe.py](src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_extra_basis_fe.py#L618) |
| `MLFRAME_FOURIER_DETECT_MAX_N` | `str(_DEFAULT_MAX_N)` | [src/mlframe/feature_selection/filters/_fourier_detect_cap.py](src/mlframe/feature_selection/filters/_fourier_detect_cap.py#L36) |
| `MLFRAME_FP_CACHE_MAX` | — | [src/mlframe/training/feature_handling/fingerprint.py](src/mlframe/training/feature_handling/fingerprint.py#L62) |
| `MLFRAME_FRAC_DIFF_INV_BACKEND` | `''` | [src/mlframe/training/composite/transforms/nonlinear.py](src/mlframe/training/composite/transforms/nonlinear.py#L625) |
| `MLFRAME_GATE_BUILD_NJIT_MIN_N` | `'20000'` | [src/mlframe/feature_selection/filters/_conditional_gate_fe.py](src/mlframe/feature_selection/filters/_conditional_gate_fe.py#L139) |
| `MLFRAME_GROUPED_COUNT_VECTORIZE_MAX_AVG` | `'64'` | [src/mlframe/feature_engineering/grouped.py](src/mlframe/feature_engineering/grouped.py#L61) |
| `MLFRAME_HINGE_BATCH_MIN_K` | `''` | [src/mlframe/feature_selection/filters/_hinge_basis_fe.py](src/mlframe/feature_selection/filters/_hinge_basis_fe.py#L396) |
| `MLFRAME_HINGE_MAX_ROWS` | `'250000'` | [src/mlframe/feature_selection/filters/_hinge_detect_gpu_resident.py](src/mlframe/feature_selection/filters/_hinge_detect_gpu_resident.py#L147) |
| `MLFRAME_HYP_PROFILE` | `'mlframe-fast'` | [src/mlframe/testing/parametric.py](src/mlframe/testing/parametric.py#L85) |
| `MLFRAME_INFONET_CACHE` | `str(Path.home() / '.cache' / 'mlframe' / 'infonet')` | [src/mlframe/feature_selection/filters/_neural_mi.py](src/mlframe/feature_selection/filters/_neural_mi.py#L222) |
| `MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY` | `'0'` | [src/mlframe/feature_selection/filters/evaluation.py](src/mlframe/feature_selection/filters/evaluation.py#L49) |
| `MLFRAME_KEEP_BROKEN_CUPY` | `''` | [src/mlframe/__init__.py](src/mlframe/__init__.py#L157) |
| `MLFRAME_KEEP_COLORAMA_REINIT` | `''` | [src/mlframe/__init__.py](src/mlframe/__init__.py#L253) |
| `MLFRAME_KEEP_PREDICTION_DATAMODULE` | — | [src/mlframe/training/neural/base/_base_fit.py](src/mlframe/training/neural/base/_base_fit.py#L911) |
| `MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS` | — | [src/mlframe/training/reporting/_reporting_regression/__init__.py](src/mlframe/training/reporting/_reporting_regression/__init__.py#L463) |
| `MLFRAME_KSG_GPU_N` | `'50000'` | [src/mlframe/feature_selection/filters/_ksg.py](src/mlframe/feature_selection/filters/_ksg.py#L54) |
| `MLFRAME_KTC_ONLINE_LEARN` | `''` | [src/mlframe/feature_selection/_benchmarks/kernel_tuning_cache/dispatch.py](src/mlframe/feature_selection/_benchmarks/kernel_tuning_cache/dispatch.py#L178) |
| `MLFRAME_LGB_CACHE_DISABLE` | — | [src/mlframe/training/lgb_shim.py](src/mlframe/training/lgb_shim.py#L277) |
| `MLFRAME_LOAD_MODEL_CACHE_MAX` | `'32'` | [src/mlframe/training/io.py](src/mlframe/training/io.py#L733) |
| `MLFRAME_LOAD_MODEL_CACHE_MAX_MB` | `'2048'` | [src/mlframe/training/io.py](src/mlframe/training/io.py#L681) |
| `MLFRAME_MAX_ERROR_PAR_THRESHOLD` | `'5000000'` | [src/mlframe/metrics/regression/_regression_metrics.py](src/mlframe/metrics/regression/_regression_metrics.py#L55) |
| `MLFRAME_MEMORY_BUDGET_GB` | — | [src/mlframe/training/feature_handling/system.py](src/mlframe/training/feature_handling/system.py#L89) |
| `MLFRAME_METRICS_ARGSORT_GPU_MIN_N` | `'50000'` | [src/mlframe/metrics/_core_auc_brier.py](src/mlframe/metrics/_core_auc_brier.py#L36) |
| `MLFRAME_METRICS_ARGSORT_PAR_MIN_N` | `'200000'` | [src/mlframe/metrics/_core_auc_brier.py](src/mlframe/metrics/_core_auc_brier.py#L47) |
| `MLFRAME_METRICS_STABLE_SORT` | — | [src/mlframe/metrics/_core_auc_brier.py](src/mlframe/metrics/_core_auc_brier.py#L137) |
| `MLFRAME_MI_ANALYTIC_NULL` | `'1'` | [src/mlframe/feature_selection/filters/_analytic_mi_null.py](src/mlframe/feature_selection/filters/_analytic_mi_null.py#L74) |
| `MLFRAME_MI_ANALYTIC_NULL_MIN_CELL` | `''` | [src/mlframe/feature_selection/filters/_analytic_mi_null.py](src/mlframe/feature_selection/filters/_analytic_mi_null.py#L106) |
| `MLFRAME_MI_ANALYTIC_NULL_MIN_N` | `''` | [src/mlframe/feature_selection/filters/_analytic_mi_null.py](src/mlframe/feature_selection/filters/_analytic_mi_null.py#L83) |
| `MLFRAME_MI_BACKEND` | `''` | [src/mlframe/feature_selection/filters/_hermite_fe_mi.py](src/mlframe/feature_selection/filters/_hermite_fe_mi.py#L415) |
| `MLFRAME_MI_FROM_CODES_V2` | `'0'` | [src/mlframe/feature_selection/filters/_fe_batched_mi.py](src/mlframe/feature_selection/filters/_fe_batched_mi.py#L620) |
| `MLFRAME_MLP_PIN_MEMORY` | — | [src/mlframe/training/mlp_runtime_defaults.py](src/mlframe/training/mlp_runtime_defaults.py#L34) |
| `MLFRAME_MRMR_ADDONE_PVALUE` | `'1'` | [src/mlframe/feature_selection/filters/permutation.py](src/mlframe/feature_selection/filters/permutation.py#L152) |
| `MLFRAME_MRMR_BATCH_PAIR_MI` | `'1'` | [src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairmi.py](src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairmi.py#L159) |
| `MLFRAME_MRMR_COMPACT_CODES` | `'1'` | [src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py](src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py#L5554) |
| `MLFRAME_MRMR_FALLBACK_REDUNDANCY_FRAC` | `'0.5'` | [src/mlframe/feature_selection/filters/_mrmr_fit_impl/_finalise.py](src/mlframe/feature_selection/filters/_mrmr_fit_impl/_finalise.py#L164) |
| `MLFRAME_MRMR_FIT_CACHE_MAX_MB` | `'1024'` | [src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py](src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py#L9381) |
| `MLFRAME_MRMR_GPU_CMI` | `'1'` | [src/mlframe/feature_selection/filters/_evaluation_driver.py](src/mlframe/feature_selection/filters/_evaluation_driver.py#L51) |
| `MLFRAME_MRMR_NULL_PERMS` | `'32'` | [src/mlframe/feature_selection/filters/permutation.py](src/mlframe/feature_selection/filters/permutation.py#L75) |
| `MLFRAME_MRMR_NULL_SIGNIF_ALPHA` | `'0.05'` | [src/mlframe/feature_selection/filters/evaluation.py](src/mlframe/feature_selection/filters/evaluation.py#L73) |
| `MLFRAME_NEURAL_MI_DEVICE` | `'auto'` | [src/mlframe/feature_selection/filters/_neural_mi.py](src/mlframe/feature_selection/filters/_neural_mi.py#L59) |
| `MLFRAME_NONFINITE_PAR_THRESHOLD` | `'1000000'` | [src/mlframe/feature_engineering/transformer/_utils.py](src/mlframe/feature_engineering/transformer/_utils.py#L40) |
| `MLFRAME_NO_CUDA_AUTOCONFIG` | `''` | [src/mlframe/__init__.py](src/mlframe/__init__.py#L51) |
| `MLFRAME_NO_GPU_INFO_CACHE` | — | [src/mlframe/training/cb/_cb_pool.py](src/mlframe/training/cb/_cb_pool.py#L550) |
| `MLFRAME_NUMBA_MI` | `''` | [src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_mi_backends.py](src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_orth_mi_backends.py#L203) |
| `MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL` | — | [src/mlframe/metrics/_core_numba_warmup.py](src/mlframe/metrics/_core_numba_warmup.py#L295) |
| `MLFRAME_NW_PARALLEL_MIN_QUERIES` | `'2000'` | [src/mlframe/feature_engineering/nadaraya_watson.py](src/mlframe/feature_engineering/nadaraya_watson.py#L42) |
| `MLFRAME_OOF_BATCH_BINNING_MAX_TRAIN_ROWS` | `'16000'` | [src/mlframe/feature_selection/filters/_orthogonal_three_gate_mi_fe.py](src/mlframe/feature_selection/filters/_orthogonal_three_gate_mi_fe.py#L91) |
| `MLFRAME_PAIR_NULL_MAX_ROWS` | `'250000'` | [src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairs_rank.py](src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairs_rank.py#L320) |
| `MLFRAME_PANDAS_VIEW_CACHE_MAX_MB` | — | [src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py](src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py#L62) |
| `MLFRAME_PANDAS_VIEW_CACHE_SIZE` | `''` | [src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py](src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py#L64) |
| `MLFRAME_PANDAS_VIEW_CACHE_TYPE` | `''` | [src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py](src/mlframe/training/core/_phase_train_one_target_polars_fastpath.py#L63) |
| `MLFRAME_PER_MEMBER_AUTOTUNE` | `'1'` | [src/mlframe/models/ensembling/member_metrics.py](src/mlframe/models/ensembling/member_metrics.py#L68) |
| `MLFRAME_PER_MEMBER_BACKEND` | `''` | [src/mlframe/models/ensembling/member_metrics.py](src/mlframe/models/ensembling/member_metrics.py#L62) |
| `MLFRAME_PIPELINE_CACHE_BYTES_LIMIT` | — | [src/mlframe/training/core/_phase_config_setup.py](src/mlframe/training/core/_phase_config_setup.py#L169) |
| `MLFRAME_PIPELINE_CACHE_RAM_FRACTION` | — | [src/mlframe/training/core/_phase_config_setup.py](src/mlframe/training/core/_phase_config_setup.py#L169) |
| `MLFRAME_PLOT_INLINE_DISPLAY` | — | [src/mlframe/reporting/renderers/save.py](src/mlframe/reporting/renderers/save.py#L102) |
| `MLFRAME_POLYEVAL_BACKEND` | `''` | [src/mlframe/feature_selection/filters/hermite_fe/__init__.py](src/mlframe/feature_selection/filters/hermite_fe/__init__.py#L388) |
| `MLFRAME_POLYEVAL_CUDA_THRESHOLD` | `'500000'` | [src/mlframe/feature_selection/filters/hermite_fe/_hermite_oracle.py](src/mlframe/feature_selection/filters/hermite_fe/_hermite_oracle.py#L18) |
| `MLFRAME_POLYEVAL_ORACLE` | `'0'` | [src/mlframe/feature_selection/filters/hermite_fe/_hermite_oracle.py](src/mlframe/feature_selection/filters/hermite_fe/_hermite_oracle.py#L57) |
| `MLFRAME_POLYEVAL_PAR_THRESHOLD` | `'50000'` | [src/mlframe/feature_selection/filters/hermite_fe/_hermite_oracle.py](src/mlframe/feature_selection/filters/hermite_fe/_hermite_oracle.py#L17) |
| `MLFRAME_PREBIN_CACHE` | `'1'` | [src/mlframe/training/composite/discovery/_fit.py](src/mlframe/training/composite/discovery/_fit.py#L322) |
| `MLFRAME_PREBIN_CACHE_MAX_BYTES` | — | [src/mlframe/training/composite/cache.py](src/mlframe/training/composite/cache.py#L559) |
| `MLFRAME_PREWARM_HEAVY_LIBS` | `''` | [src/mlframe/metrics/_core_numba_warmup.py](src/mlframe/metrics/_core_numba_warmup.py#L451) |
| `MLFRAME_PYSR_INPUT_BYTES_LIMIT` | `_DEFAULT_PYSR_INPUT_BYTES_LIMIT` | [src/mlframe/feature_engineering/bruteforce.py](src/mlframe/feature_engineering/bruteforce.py#L144) |
| `MLFRAME_RADIX_F64_V2` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_resident_select_kernels.py](src/mlframe/feature_selection/filters/_gpu_resident_select_kernels.py#L754) |
| `MLFRAME_RADIX_F64_V3` | `'1'` | [src/mlframe/feature_selection/filters/_gpu_resident_select_kernels.py](src/mlframe/feature_selection/filters/_gpu_resident_select_kernels.py#L737) |
| `MLFRAME_RECENCY_KDE_PARALLEL_MIN_GROUPS` | `'512'` | [src/mlframe/feature_engineering/recency_density.py](src/mlframe/feature_engineering/recency_density.py#L39) |
| `MLFRAME_RECURRENT_PREDICTION_CACHE_MAX` | `'16'` | [src/mlframe/training/neural/recurrent_dataset_helpers.py](src/mlframe/training/neural/recurrent_dataset_helpers.py#L61) |
| `MLFRAME_REGRESSION_RECALIBRATION` | `''` | [src/mlframe/training/core/_phase_finalize_calibration.py](src/mlframe/training/core/_phase_finalize_calibration.py#L275) |
| `MLFRAME_RETENTION_NULL_MAX_ROWS` | `'250000'` | [src/mlframe/feature_selection/filters/_fe_retention_subsumption.py](src/mlframe/feature_selection/filters/_fe_retention_subsumption.py#L104) |
| `MLFRAME_ROBUST_AXIS` | `''` | [src/mlframe/feature_selection/filters/hermite_fe/_hermite_robust.py](src/mlframe/feature_selection/filters/hermite_fe/_hermite_robust.py#L60) |
| `MLFRAME_ROBUST_MEAN_PARALLEL_MIN_N` | `'50000'` | [src/mlframe/core/robust_location.py](src/mlframe/core/robust_location.py#L40) |
| `MLFRAME_ROBUST_WARP_FIT` | `''` | [src/mlframe/feature_selection/filters/hermite_fe/_hermite_robust.py](src/mlframe/feature_selection/filters/hermite_fe/_hermite_robust.py#L278) |
| `MLFRAME_SETUP_TIMING` | `'1'` | [src/mlframe/training/core/_phase_config_setup.py](src/mlframe/training/core/_phase_config_setup.py#L116) |
| `MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS` | `''` | [src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_subsetrank.py](src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_subsetrank.py#L155) |
| `MLFRAME_SKIP_NUMBA_WARMUP` | — | [src/mlframe/feature_selection/filters/_numba_warmup.py](src/mlframe/feature_selection/filters/_numba_warmup.py#L33) |
| `MLFRAME_STABILITY_CLUSTER_MAX_FEATURES` | `'4000'` | [src/mlframe/feature_selection/filters/_stability_cluster.py](src/mlframe/feature_selection/filters/_stability_cluster.py#L57) |
| `MLFRAME_TORCH_COMPILE_DEBUG` | `'0'` | [src/mlframe/training/neural/_flat_torch_module/_flat_torch_predict_accel.py](src/mlframe/training/neural/_flat_torch_module/_flat_torch_predict_accel.py#L105) |
| `MLFRAME_TORCH_COMPILE_PREDICT` | `'0'` | [src/mlframe/training/neural/_flat_torch_module/_flat_torch_predict_accel.py](src/mlframe/training/neural/_flat_torch_module/_flat_torch_predict_accel.py#L164) |
| `MLFRAME_TORCH_PROFILE` | `'0'` | [src/mlframe/training/neural/base/_base_fit.py](src/mlframe/training/neural/base/_base_fit.py#L596) |
| `MLFRAME_TORCH_PROFILE_DIR` | `os.path.join(_ckpt_root, 'torch_traces')` | [src/mlframe/training/neural/base/_base_fit.py](src/mlframe/training/neural/base/_base_fit.py#L602) |
| `MLFRAME_TRUST_LGB_CUDA` | — | [src/mlframe/training/_gpu_probe.py](src/mlframe/training/_gpu_probe.py#L50) |
| `MLFRAME_TTR_DISABLE_PREDICT_CLIP` | — | [src/mlframe/training/targets/_ttr_eval_set_scaling.py](src/mlframe/training/targets/_ttr_eval_set_scaling.py#L107) |
| `MLFRAME_USABILITY_CORR_MAX_ROWS` | `'30000'` | [src/mlframe/feature_selection/filters/_fe_usability_signal.py](src/mlframe/feature_selection/filters/_fe_usability_signal.py#L138) |
| `MLFRAME_USABILITY_POOL_BACKEND` | `''` | [src/mlframe/feature_selection/filters/_usability_njit_pool.py](src/mlframe/feature_selection/filters/_usability_njit_pool.py#L715) |
| `MLFRAME_USE_SKLEARNEX` | `'1'` | [src/mlframe/feature_engineering/transformer/_intel_patch.py](src/mlframe/feature_engineering/transformer/_intel_patch.py#L45) |
| `MLFRAME_XGB_CACHE_DISABLE` | — | [src/mlframe/training/xgb_shim.py](src/mlframe/training/xgb_shim.py#L200) |
| `MODE` | `'baseline'` | [src/mlframe/feature_selection/_benchmarks/fs_hybrid/hybrid_opt_baseline.py](src/mlframe/feature_selection/_benchmarks/fs_hybrid/hybrid_opt_baseline.py#L24) |
| `MRMR_CAMPAIGN_RAISE` | — | [src/mlframe/feature_selection/_benchmarks/fs_quality/mrmr_largeN_campaign.py](src/mlframe/feature_selection/_benchmarks/fs_quality/mrmr_largeN_campaign.py#L185) |
| `PYTHON_JULIACALL_THREADS` | `'?'` | [src/mlframe/training/_benchmarks/bench_pysr_fe.py](src/mlframe/training/_benchmarks/bench_pysr_fe.py#L185) |
| `PYUTILZ_KERNEL_CACHE_DIR` | `''` | [src/mlframe/utils/_param_oracle.py](src/mlframe/utils/_param_oracle.py#L130) |
| `SCENE_N` | `'700'` | [src/mlframe/feature_selection/_benchmarks/fs_hybrid/round4_fs_campaign_profile.py](src/mlframe/feature_selection/_benchmarks/fs_hybrid/round4_fs_campaign_profile.py#L17) |
| `TRANSFORMERS_CACHE` | — | [src/mlframe/training/feature_handling/hf_provider.py](src/mlframe/training/feature_handling/hf_provider.py#L84) |
| `WELLBORE_DUMP_AUDIT` | `'0'` | [src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py](src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py#L30) |
| `WELLBORE_FE_OPTIMIZER` | `'cupy_kernel'` | [src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py](src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py#L87) |
| `WELLBORE_MRMR_CPROFILE` | `'0'` | [src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py](src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py#L25) |
| `WELLBORE_MRMR_MODE` | `'gpu'` | [src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py](src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py#L22) |
| `WELLBORE_TARGET_ROWS` | `'100000'` | [src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py](src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py#L24) |
| `_SF3_ARM_SEEDS` | `'[0,1,2,3]'` | [src/mlframe/feature_selection/filters/_benchmarks/bench_sf3_jmim_exponent_selection.py](src/mlframe/feature_selection/filters/_benchmarks/bench_sf3_jmim_exponent_selection.py#L105) |
| `_USAB_FORCE_FULL_REFIT` | — | [src/mlframe/feature_selection/filters/_usability_aware_selection.py](src/mlframe/feature_selection/filters/_usability_aware_selection.py#L827) |

## Contributing

Pull requests are welcome. Code style is `black` + `ruff` with a line length of 160.
Every new feature ships with a unit test, a quantitative business-value test, a
representative `@pytest.mark.fast` subset, and a `cProfile` hotspot check. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow, test bar, and
the fuzz / combo test philosophy.

## License

MIT, see [LICENSE](LICENSE).
