# mlframe

[![CI](https://github.com/fingoldo/mlframe/workflows/CI/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml)
[![sklearn-matrix](https://github.com/fingoldo/mlframe/workflows/sklearn-matrix/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml)
[![codecov](https://codecov.io/gh/fingoldo/mlframe/branch/master/graph/badge.svg)](https://codecov.io/gh/fingoldo/mlframe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://fingoldo.github.io/mlframe/)

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
pip install -e "./mlframe[neural]"               # torch + lightning + captum + transformers
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
and neural pipelines get encoded, imputed, and scaled). The suite returns a
`(models, metadata)` tuple: `models` is a per-target-type dict of trained model
entries, `metadata` carries fit-time metrics, calibration diagnostics, baseline
diagnostics, and the reporting spec. The features and targets are pulled from the
frame by a caller-supplied extractor (see `SimpleFeaturesAndTargetsExtractor`).

```python
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

models, metadata = train_mlframe_models_suite(
    df=df,
    target_name="y",
    model_name="exp_quickstart",
    features_and_targets_extractor=fte,
    mlframe_models=["cb", "lgb", "xgb", "hgb", "mlp"],
)

# `models` is keyed by target-type, then model name; each entry exposes the fitted model.
entry = models["regression"]["lgb"][0]
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
from sklearn.ensemble import RandomForestRegressor
from mlframe.training.composite import CompositeTargetEstimator

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
flat tuple (and, optionally, the calibration figure).

```python
from mlframe.metrics import fast_calibration_report

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
```

**MRMR / RFECV feature selection.** Several MRMR variants (FCQ, MID, FCD, plus
n-way interaction extensions) and an RFECV wrapper that reads LightGBM / XGBoost /
CatBoost feature importances correctly. Both are scikit-learn `fit` / `transform`
estimators; the selected columns land on the fitted estimator after `fit(X, y)`.

```python
from lightgbm import LGBMClassifier
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.wrappers import RFECV

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

`create_aggregated_features` is the per-window worker: it appends numeric / categorical
aggregates (raw, diffs, ratios, robust, EWMA, rolling, wavelets, ...) computed over the
rows of one window into the caller-supplied `row_features` list (and, when
`create_features_names=True`, the parallel `features_names` list). It mutates those
lists in place and returns `None`.

```python
from mlframe.feature_engineering.timeseries import create_aggregated_features

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
```

`create_ohlcv_wholemarket_features` builds cross-ticker market-wide aggregates
(min / max / std / mean / quantiles plus value-weighted variants) per timestamp on a
Polars OHLCV frame:

```python
from mlframe.feature_engineering.financial import create_ohlcv_wholemarket_features

market = create_ohlcv_wholemarket_features(ohlcv, timestamp_column="date")
```

**Post-hoc probability calibration.** Compare Venn-Abers, isotonic, Platt, beta, and
per-class isotonic on out-of-fold predictions and pick the calibrator that minimises
OOF ECE (with a bootstrap-CI tiebreak). Selection is OOF-only to keep the estimate
honest; the returned dict carries the chosen calibrator name, its fitted object, and
the per-candidate ECE scores.

```python
from mlframe.calibration.policy import pick_best_calibrator

result = pick_best_calibrator(
    probs=None, y=None,                 # optional diagnostic-only held-out probs/labels
    oof_probs=oof_proba, oof_y=y_val,   # OOF probs/labels drive the selection
    candidates=["isotonic", "platt", "beta", "venn_abers"],
    n_bins=15,
)
print(result["chosen"], result["ece_mean"], result["alternatives"])
```

**Inference from saved models.** `read_trained_models` loads a featureset's saved
models from an inference folder (with optional `trusted_root` path-traversal guard and
SHA-256 sidecar verification), returning `(models, X)` aligned to the required feature
order; `get_models_raw_predictions` then evaluates each loaded model on `X`.

```python
from mlframe.inference.predict import read_trained_models, get_models_raw_predictions

models, X_aligned = read_trained_models(
    featureset="my_featureset",
    X=X_new,
    inference_folder="infer",
)
preds = get_models_raw_predictions(models, X_aligned, Y=None)
```

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
| Multiclass | `CONFUSION CONFUSED_PAIRS PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC` (`multiclass_panels`) |
| Multilabel | `PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST` (`multilabel_panels`) |
| LTR | `NDCG_K NDCG_DIST NDCG_BY_QSIZE LIFT MRR_DIST SCORE_BY_REL` (`ltr_panels`) |
| Quantile | `RELIABILITY COVERAGE PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST QUANTILE_RELIABILITY PINBALL_DECOMP QUANTILE_CROSSING` (`quantile_panels`) |
| Regression | `SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE` (`regression_panels`) |

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
[CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow, test bar, and
the fuzz / combo test philosophy. By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT, see [LICENSE](LICENSE).
