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
  metrics.py             # ICE, ECE, Brier decomposition (REL/RES/UNC), CMAEW, calibration plot
  tests/
    training/
      test_core.py       # Integration tests for train_mlframe_models_suite
      test_core_coverage.py  # Comprehensive coverage tests (48 tests, all code paths)
      test_suite_coverage_gaps.py  # Tier 1-4 + Group A-C invariant / edge-case tests (47)
      test_fuzz_suite.py  # 150-combo pairwise fuzz on 39 axes — per-combo
                          # caller-mutation, metadata, and Fix-C prediction
                          # invariants (finiteness, non-constant, shape)
      test_fuzz_3way_suite.py          # 400-combo 3-wise (IPOG) coverage on
                                       # 15 load-bearing axes (Fix A, nightly)
      test_fuzz_metamorphic.py         # Dual-run: column-rename + dup-row
                                       # stability (Fix D, 5 curated combos)
      test_fuzz_hypothesis.py          # Hypothesis continuous-leaf sampler
                                       # for n_rows/fillna/test_size/... (Fix B)
      test_fuzz_regression_sensors.py  # Permanent sensors per fuzz-caught bug
      _fuzz_combo.py      # FuzzCombo axes, frame builder, KNOWN_XFAIL_RULES,
                          # pairwise + 3-wise covering algorithms
      run_fuzz_10k.py                  # Driver: pytest per master_seed (~10k combos)
      run_fuzz_seed_rotation.py        # Nightly seed-rotation driver (Fix E)
      COMBO_FUZZ.md       # Design doc: combo approach, A-G roadmap
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

## Target types

`TargetTypes` — declared via `SimpleFeaturesAndTargetsExtractor(target_type=...)`
or auto-derived from `regression=True/False`. The suite plumbs the value into
[get_training_configs](training/helpers.py) so each strategy injects the right
native objective (CB `loss_function`, XGB `objective`+`num_class`, LGB
`objective`+`num_class`); non-native cases get auto-wrapped in
`MultiOutputClassifier` / `_ChainEnsemble`.

| target_type                  | y shape | CB              | XGB                            | LGB                            | HGB                            | Linear (LR)                    |
|------------------------------|:-------:|:---------------:|:------------------------------:|:------------------------------:|:------------------------------:|:------------------------------:|
| `REGRESSION`                 | `(N,)`  | regressor       | regressor                      | regressor                      | regressor                      | regressor                      |
| `BINARY_CLASSIFICATION`      | `(N,)`  | `Logloss`       | `binary:logistic`              | `binary`                       | classifier (auto)              | `LogisticRegression`           |
| `MULTICLASS_CLASSIFICATION`  | `(N,)`  | `MultiClass`    | `multi:softprob` + `num_class` | `multiclass` + `num_class`     | classifier (auto)              | `multi_class='multinomial'`    |
| `MULTILABEL_CLASSIFICATION`  | `(N,K)` | `MultiLogloss` (native) | `MultiOutputClassifier(...)`   | `MultiOutputClassifier(...)`   | `MultiOutputClassifier(...)`   | `MultiOutputClassifier(...)`   |

Multilabel notes:
- CB returns `(N, K)` directly via `MultiLogloss`; no wrapper.
- Other strategies are wrapped in `sklearn.multioutput.MultiOutputClassifier`
  by [training/trainer.py:_wrap_for_multilabel_if_needed](training/trainer.py).
  Inner-estimator `early_stopping_rounds`/`callbacks` are auto-disabled —
  `eval_set` does not safely propagate through the wrapper to per-label fits.
- Set `MultilabelDispatchConfig(strategy="chain", n_chains=...)` to use
  `_ChainEnsemble` (random-ordered `ClassifierChain` ensemble) when labels
  are correlated; +2-5% Jaccard on correlated synthetic data.
- Stratified splits for 2-D y require `iterative-stratification` (optional
  dep): `pip install iterative-stratification`.
- Numba multilabel metrics: `mlframe.metrics.{hamming_loss,
  subset_accuracy, jaccard_score_multilabel}` — auto-routed to a parallel
  variant when `N*K > 1_000_000`.

See [docs/MULTI_OUTPUT.md](docs/MULTI_OUTPUT.md) for the full design notes.

## Selection-bias / drift tools

For binary targets where train/val/test marginals diverge (selection
bias, temporal drift, label-shift), the suite ships two tools to
catch the problem early and remediate it.

### Drift detection (auto-emitted by the suite)

Every per-target run logs a `label_distribution_drift report` block
right after the train/val/test split materialises and BEFORE training
starts. It surfaces P(y=1) per split for binary targets (per-class /
per-label / mean+std for the other target types) and warns when any
split's prior diverges from train's by more than 5pp (binary/multi)
or 0.5σ (regression). The structured report is also stored on
`metadata["label_distribution_drift"]` for retrospective inspection.

```python
from mlframe.training import compute_label_distribution_drift, format_drift_report

report = compute_label_distribution_drift(y_train, y_val, y_test, "binary_classification")
print(format_drift_report(report, target_name="my_target"))
# WARN: TEST P(y=1)=0.83 vs train 0.74 (Δ=+8.7pp); selection-bias / prior-shift suspected ...
```

### PU-learning wrapper (selection-biased binary classification)

When the training data is "positives heavily over-observed because the
data source mostly only surfaces positives" (e.g. a marketplace listing
hired jobs but not unhired ones), a naive classifier learns the wrong
prior and gets blown out at TEST time. `PULearningWrapper` recovers
calibration via three strategies — pick via `strategy="..."` or let
`strategy="auto"` choose:

| strategy                  | when to use                                              |
|---------------------------|----------------------------------------------------------|
| `unbiased_only`           | ≥1k each-class fully-labeled samples; cleanest calibration |
| `prior_shift_correction`  | Saerens (2002); preserves AUC, recovers calibration       |
| `elkan_noto`              | Elkan & Noto (KDD 2008); classical PU                     |
| `auto` (default)          | picks `unbiased_only` if data large enough, else Saerens   |

```python
from mlframe.training import PULearningWrapper
from sklearn.ensemble import HistGradientBoostingClassifier

pu = PULearningWrapper(
    base_estimator=HistGradientBoostingClassifier(max_iter=200),
    strategy="prior_shift_correction",
    true_prior=0.40,  # known target population P(y=1); estimated from unbiased subset if None
)
pu.fit(X=X_train, y=y_observed, is_unbiased=is_full_labeled_period)
probs = pu.predict_proba(X_test)
```

On a synthetic temporal-bias benchmark (true TEST P(y=1)=0.46, train
observed P(y=1)=0.96): naive Brier 0.376 → `prior_shift_correction`
0.149 (−60%), AUC 0.864 unchanged.

See [docs/SELECTION_BIAS.md](docs/SELECTION_BIAS.md) for the decision
matrix, full benchmark, and what these tools do NOT fix
(feature-distribution drift, label noise).

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
  - `cat_text_cardinality_threshold` — unique value count threshold (default 300): `<= threshold` → categorical, `> threshold` → text. Raised from 50 → 300 on 2026-04-19 (round 12) after a prod incident where mid-cardinality columns (`job_post_source:71`, `_raw_countries:2196`) got promoted to text_features and crashed CatBoost's TF-IDF estimator. 50-300 unique values are usually enum-like (country codes, categories) — tree models handle these natively as cats, no text extraction needed.
- **`preprocessing_extensions`** — optional `PreprocessingExtensionsConfig` (or dict). Shared sklearn stack applied once after the Polars-ds pipeline; every model reuses the transformed frame. Covers scaler override (10 variants), `Binarizer`/`KBinsDiscretizer` (mutually exclusive), `PolynomialFeatures` with `memory_safety_max_features` guard, non-linear maps (`RBFSampler`/`Nystroem`/`AdditiveChi2Sampler`/`SkewedChi2Sampler`), TF-IDF, and dim reducers (PCA / KernelPCA / LDA / NMF / TruncatedSVD / FastICA / Isomap / UMAP / random projections / RandomTreesEmbedding / BernoulliRBM). `None` (default) is a byte-for-byte noop — the Polars-native fastpath is preserved. UMAP is gated via `importlib.util.find_spec` with an install-hint `ImportError`.
- **`feature_selection_config`** — feature selection configuration (`FeatureSelectionConfig` or dict): `use_mrmr_fs`, `mrmr_kwargs`, `rfecv_models`, `rfecv_kwargs`, and `custom_pre_pipelines` (dict of custom sklearn transformers like PCA).
- **`reporting_config`** — calibration / training-report look (`ReportingConfig` or dict): figsize, chart toggles, the **title-metrics template** (`"ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC"` by default - a single ordered string instead of 9 separate booleans, validated at config-construction time), **probability histogram subplot** under the reliability scatter (`show_prob_histogram`, `prob_histogram_yscale="auto"|"log"|"linear"`), inline per-bin population labels (`show_inline_population_labels`), and the typed `feature_importance_config`.
- **`output_config`** — filesystem destinations (`OutputConfig` or dict): `data_dir`, `models_dir`, `plot_file`, `save_charts` (when `False`, skips per-model chart file output - useful for CI).
- **`outlier_detection_config`** — outlier-detection settings (`OutlierDetectionConfig` or dict): `detector` (sklearn OutlierMixin), `apply_to_val`.
- **`verbose`** — when `True`, logs timing and shape info for every major phase (data loading, splitting, pipeline, per-model training, metrics)

### Calibration metrics emitted per class

The per-class `class_metrics` dict carries (alongside `roc_auc`, `pr_auc`, `precision`, `recall`, `f1`):
- `ice` — Integral Calibration Error (mlframe-native composite of CMAEW + ROC + PR + Brier)
- `ece` — standard Expected Calibration Error
- `calibration_mae`, `calibration_std` — CMAE/CMAEW (power-weighted by default; same bins as ECE)
- `calibration_coverage` — fraction of bins populated
- `brier_loss` — raw Brier score
- `brier_reliability` (REL), `brier_resolution` (RES), `brier_uncertainty` (UNC) — Murphy 1973 decomposition. Identity `BinnedBrier == REL - RES + UNC` holds exactly to fp precision (raw Brier differs from binned Brier by within-bin prediction variance). REL high ⇒ calibration problem (try Platt/isotonic/postcalibration); RES low ⇒ ranker can't separate.
- `log_loss` — None when single-class

### Hyperparameters notes

- `ModelHyperparamsConfig.early_stopping_rounds: Optional[int]` — set to `None` to disable early stopping across all strategies (CB/LGB/XGB/MLP/RFECV/HGB/NGB).
- `PreprocessingExtensionsConfig.tfidf_columns` — listed text columns are vectorized in `apply_preprocessing_extensions` and replaced with `<col>__tfidf_<i>` numeric features before any model sees the frame.

### Suite metadata

On return, `metadata` exposes (in addition to the training artefacts documented elsewhere):

- `metadata["fairness_report"]` — aggregated fairness metrics propagated from per-model runs.
- `metadata["outlier_detection"]` — dict with `applied`, `n_outliers_dropped_train`, `n_outliers_dropped_val`, `train_size_after_od`, `val_size_after_od`.

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

### Static meta-tests (`tests/test_meta/`)

Static parity checks that catch *classes* of bugs without exercising any
training code. Each runs in <1 s and is part of the regular pytest suite.

| Test | What it asserts |
|------|-----------------|
| `tests/test_meta/test_config_field_consumption.py` | Every Pydantic Field declared in `mlframe/training/configs.py` is referenced by at least one production consumer (anything in `mlframe/` outside `tests/`). Catches dead config fields — added but never threaded into the trainer/strategy/pipeline that should read them, leading to silent no-op flags ("I configured X=True but nothing changed"). Includes a self-test that refuses to pass on an empty corpus, plus a `model_dump()`-splat detector that auto-exempts whole-class fields consumed via `**cfg.model_dump(...)`. |
| `tests/test_meta/test_estimator_kwarg_parity.py` | Every per-estimator `<flavor>_kwargs` field on every config class (`cb_kwargs` / `lgb_kwargs` / `xgb_kwargs` / `hgb_kwargs` / `mlp_kwargs` / `ngb_kwargs` / `rfecv_kwargs`) must reach an estimator constructor via `**field`, `.update(field)`, `field.get(...)`, or extract-then-splat. Catches the failure mode where a typed-config refactor adds the field on the dataclass but the trainer keeps the old standalone-kwarg interface. |
| `tests/test_meta/test_subconfig_wiring_parity.py` | Every BaseModel-typed field on a parent config (e.g. `TrainingConfig.linear_config: LinearModelConfig`) must appear as a bare attribute access (`cfg.linear_config`) or kwarg pass-through somewhere in production. Catches sub-configs orphaned by trainer-side parameter renaming. |
| `tests/test_meta/test_dead_helpers.py` | Every public top-level `def` / `class` in `training/` and `feature_selection/` must be referenced ≥ 2 times in the production corpus (definition + ≥ 1 call site). Surfaces stale helpers left behind by refactors. Top-level `mlframe/*.py` modules are out of scope by design — those are the public-API surface for notebook users. |
| `tests/test_meta/test_metric_invariants.py` | Hypothesis-driven property checks on `mlframe.metrics`: Brier decomposition identity (`BinnedBrier == REL - RES + UNC`), all decomposition components in `[0, 1]`, AUC bounds + monotonic-affine invariance, perfect-prediction Brier == 0, log-loss ≥ 0, hamming/jaccard/subset-accuracy bounds, plus `_predict_from_probs` boundary cases. |
| `tests/test_meta/test_enum_exhaustiveness.py` | Every `Literal[...]` / `StrEnum` string value declared on a config field must appear quoted somewhere in the production corpus — i.e. is dispatched on, not just listed-but-ignored. |
| `tests/test_meta/test_utility_fuzz.py` | Targeted Hypothesis fuzz on the prepare-for-estimator transforms (`prepare_df_for_catboost` NaN handling, `_canonical_predict_proba_shape` shape adapter, `_predict_from_probs` boundary thresholds + NaN behaviour). Complements `test_fuzz_suite.py` (broad-axis integration fuzz) with unit-level boundary scrutiny. |
| `tests/test_meta/test_strategy_registration.py` | Every model-type alias accepted by `VALID_MODEL_TYPES` / `VALID_LINEAR_MODEL_TYPES` resolves to a real `ModelPipelineStrategy` instance via `MODEL_STRATEGIES`. Catches the silent-fall-through bug where a contributor adds an alias but forgets to register the strategy. Surfaced + closed the `hgb` mismatch on first run. |
| `tests/test_meta/test_config_docstring_drift.py` | Every parameter listed in a config-class docstring's numpydoc `Parameters ----------` section corresponds to a real field on that class. Catches stale documentation after a rename. |
| `tests/test_meta/test_config_round_trip.py` | `cls(**cls().model_dump()) == cls()` for every default-constructable Pydantic config. Catches mutable-default sharing, model_dump losing fields, non-deterministic `@model_validator(mode="after")`. |
| `tests/test_meta/test_field_bound_enforcement.py` | Every numeric-bounded field (`Field(ge=0, le=1)`) and every Literal-typed field actually rejects out-of-bounds inputs at construction time. |
| `tests/test_meta/test_mutual_exclusion_validators.py` | Every "X and Y are mutually exclusive" claim in a config docstring or `@model_validator` raise message is enforced empirically (instantiation with both fields set raises). |
| `tests/test_meta/test_validator_coverage.py` | Fields whose docstring promises a normalisation ("Case-insensitive, normalized to lowercase") must have a `@field_validator` that actually applies a normalising op (`.lower()` / `.upper()` / `.strip()`). |
| `tests/test_meta/test_todo_hygiene.py` | Every `TODO` / `FIXME` / `XXX` / `HACK` comment in production code must carry an attribution (assignee in parens or ISO date). Un-attributed markers fail unless explicitly grandfathered. |
| `tests/test_meta/test_api_stability.py` | Captures the public surface of `mlframe.training` (re-exported names + signature shapes + class MROs) into `_api_snapshot.json`. Renames / removals fail; additions are silent. Refresh after intentional API changes via `pytest tests/test_meta/test_api_stability.py --refresh-api-snapshot`. |
| `tests/test_meta/test_deferred_drift.py` | (A2) Counts entries in every `_USER_DEFERRED_*` / `_GRANDFATHERED` whitelist across every meta-test (via AST), compares against `_debt_baseline.json`. Fails when a whitelist GROWS. Refresh via `--refresh-debt-baseline`. Net counter visible per run. |
| `tests/test_meta/test_reproducibility.py` | (C1) Every linear model (ridge / lasso / elasticnet / sgd) yields bit-identical predictions when refit with the same `random_state`; `_predict_from_probs` is a pure function; `apply_preprocessing_extensions` is reproducible. Catches non-deterministic global-state pollution. |
| `tests/test_meta/test_public_api_contract.py` | (C2) Every value in `VALID_LINEAR_MODEL_TYPES` actually fits + predicts on a tiny synthetic dataset. Goes beyond MT-1's structural check by exercising the runtime path for every advertised alias. |
| `tests/test_meta/test_memory_budgets.py` | (C3) Linear-model fit + predict on a 200×8 dataset stays under 30 MB peak via `tracemalloc`; `_predict_from_probs` on 50K×10 stays under 5 MB. Catches accidental allocator regressions ≈5x without false-flagging micro-perf noise. |
| `tests/test_meta/test_calibration_monotonicity.py` | (C4) Hypothesis-driven: post-hoc isotonic calibration never increases per-label Brier on its own training set (Murphy 1973); calibrator falls back to identity on constant-label folds. |
| `tests/test_meta/test_version_consistency.py` | (E3) `mlframe.__version__` matches `version.py`'s `__version__` (and would also check `pyproject.toml::[project].version` if/when added). Catches "I bumped one version source but not the other". |
| `tests/test_meta/test_no_import_cycles.py` | (E4) Tarjan's SCC over the AST-built import graph; flags multi-node cycles. Surfaced 3 real cycles in mlframe held in `_USER_DEFERRED_CYCLES` for restructuring later. |
| `tests/test_meta/test_no_unicode_in_console_output.py` | (E5) Every `print(...)` / `logger.*(...)` call's first arg is ASCII-only. Snapshot-based: 74 existing offenders in baseline; new commits adding non-ASCII fail. Critical for Windows cp1251 stdout. |
| `tests/test_meta/test_public_docstrings.py` | (E1) Every public top-level `def` / `class` in production has a docstring. Snapshot-based; baseline currently 832 undocumented. Refresh: `--refresh-docstring-baseline`. |
| `tests/test_meta/test_public_annotations.py` | (E2) Every public function has return annotation + every non-self/cls param is typed. Snapshot-based; baseline 1356 unannotated. Refresh: `--refresh-annotation-baseline`. |
| `tests/test_meta/test_meta_meta.py` | (F1+F2+F3) Every `pytest.fail(...)` has actionable text; meta-tests don't reach into private internals without a whitelist entry; per-test perf-budget overrides match real test names. |

The shared building blocks (`consumer_corpus`, `public_top_level_symbols`, `capture_signature`, etc.) live in [pyutilz.dev.meta_test_utils](https://github.com/anatoly-ru/pyutilz/blob/main/src/pyutilz/dev/meta_test_utils.py) and are reused across both projects.

A `.pre-commit-config.yaml` hook (`mlframe-meta-tests`) runs the entire meta-test suite on every commit (≈ 1 minute total). Install with `pip install pre-commit && pre-commit install`. A `manual`-stage variant (`mlframe-meta-tests-static-only`) skips the Hypothesis tests for tight inner-loop work.

Each meta-test exposes two whitelists at file scope:

- `_KNOWN_INDIRECT_CONSUMERS` (or analogue) — fields/symbols consumed via routes a static grep can't see (`getattr(cfg, name)` loops, MCP wiring, etc.). Each entry MUST cite the consumer file:line. Keep short — every entry is technical debt that can mask future drift.
- `_USER_DEFERRED_DEAD` (or analogue) — fields/helpers the maintainer has surfaced and explicitly chose to defer cleanup on. Drain to zero over time. When wiring or deleting one of these, remove the corresponding line.

## Testing approach: reactive + proactive

Every non-trivial bug fixed in mlframe should land with two kinds of tests,
not just one. They catch different classes of regressions and together
approximate "this bug cannot come back nor spawn a sibling nearby".

### Reactive sensors — "don't break what's already fixed"

Classical regression tests: a concrete scenario that used to fail now
passes. Anchored to a specific commit / issue / production symptom.

Examples in this repo:
- `tests/training/test_cb_polars_fallback.py::test_fallback_decategorizes_text_columns_before_retry` — a pd.Categorical text column must not arrive at the CatBoost retry (otherwise CB raises "dtype 'category' but not in cat_features list"). The assertion message names the exact backend error.
- `tests/training/test_splitting_edges.py::test_test_size_1_with_timestamps_does_not_crash_on_empty_train` — `test_size=1.0` + timestamps used to hit `NaTType does not support strftime` on the empty-train date-range format.
- `tests/test_preprocessing.py::test_pandas_text_feature_skips_expensive_astype_rebuild` — perf budget sensor: prep_cb on a 50 k × 5 k-unique text column must finish in < 2 s (the pre-fix `astype(str).astype("category")` dance took minutes).

Rules of thumb for reactive sensors:
- **Name the production symptom in the docstring.** A future regression is easier to diagnose when the test title and error message name the exact user-visible error.
- **Keep the dataset small and deterministic.** Seed RNGs; avoid `tmp_path`-sensitive fixtures unless the bug is file-IO-specific.
- **Include a perf budget** when the bug was "this was slow / hung".

### Proactive probes — "what else looks wrong in this neighbourhood?"

After the reactive tests pass, spend ~10 minutes running what-if
experiments around the modified surface. This is where most of the
second-order bugs are caught.

Pattern: write a one-shot Python snippet that stresses edge cases
nobody wrote a test for yet:

```python
# adapt to the module under review
def check(name, fn):
    try: fn(); print(f"[{name}] OK")
    except Exception as e: print(f"[{name}] CRASH: {type(e).__name__}: {e}")

check("None arg",       lambda: f(None))
check("empty list",     lambda: f([]))
check("negative size",  lambda: f(-0.1))
check("threshold edge", lambda: f(threshold=10, n_unique=10))  # strict vs non-strict
check("overlap",        lambda: f(cat=['a'], text=['a']))
check("huge input",     lambda: f(n=1_000_000))  # perf sanity + memory
# ... etc.
```

Probe categories that have caught real bugs in mlframe so far:

| Category | What it surfaces | Example finding (2026-04-19) |
|---|---|---|
| None-guard | `if x in arg` / `for x in arg` crash on `None` | `_auto_detect_feature_types(cat_features=None)` → TypeError |
| Empty input | `.min()` on empty → `NaT`; `/ len(x)` → ZeroDiv | `make_train_test_split(test_size=1.0, timestamps=ts)` → NaT strftime |
| Boundary | `>` vs `>=`; sizes at 0 / 1 / threshold | `cat_text_cardinality_threshold` `>` is correct, regression to `>=` caught |
| Dtype edge | `pl.Enum` ≠ `pl.Categorical` for `dtype in (...)` | `pl.Enum` high-cardinality columns never promoted to text |
| State leak | in-place mutation of shared arg | `prepare_df_for_catboost(cat_features=list)` appends across calls |
| Silent overlap | same column in two feature-type lists | `_validate_feature_type_exclusivity(None, ...)` failed to validate |
| Orchestration | A must run before B | fallback `decategorize` after `prep_cb` → minutes-long hang |
| Retry propagation | errors in retry path swallowed | pandas-retry failure must propagate up |
| Catastrophic misconfig | detector/threshold discards ~100% of data | `_apply_outlier_detection_global`: contamination too high → 0-row train, 5 min later opaque crash. Now fails loud before fit. |
| NaN propagation | single-class eval → NaN metric → silent early-stop freeze | `integral_calibration_error_from_metrics(roc_auc=NaN)` poisoned ICE; early-stop compared NaN > best (always False), trainer stuck on iter-1. Guards added. |
| Strict vs lenient configs | typo silently absorbed by `extra='allow'` | `TrainingSplitConfig(trainset_agng_limit=0.5)` silently ignored. Hybrid Variant C: stable-surface configs switched to `extra='forbid'` (raises loud), research configs keep `allow` + warn. |

When a probe surfaces a real bug:

1. Fix it.
2. Add a reactive sensor under `tests/**/` with the production symptom in the docstring.
3. Keep the probe snippet in the commit message or in a `bench_*.py` file if it's reusable (e.g. perf benches in `bench_shared_dict_cache.py`, `bench_long_strings.py`).

### Why both matter

Reactive-only: comforting but narrow — the bugs they target have already
been fixed, so they pass on first run and give a false sense of coverage.
The string of production bugs on 2026-04-18/19 all slipped through
reactive-only testing.

Proactive-only: unbounded — the number of "what if" probes is infinite,
and without a concrete anchor you can't tell "done".

Together: reactive guards the fixed spots, proactive finds new spots, and
every finding from proactive graduates to a reactive sensor.

### Lessons from the 2026-04-18/19 campaign (13 commits, ~170 sensors)

Recurring bug patterns the campaign surfaced — probe for these explicitly
on any new code path:

| Pattern | Representative example |
|---|---|
| NaN-in-comparison silently breaks early-stop | `ICE metric`, `RFECV score`, `per-fold importances` all had NaN that made `x > best` always False, trainer stalled with no visible error |
| Stale/shared cache keys | `PipelineCache` used `cache_key="tree"` for CB/LGB/XGB; CB cached frame with text cols, LGB retrieved same cache → polars fastpath broke. Fix: include `feature_tier()` in key |
| pl.Enum is NOT pl.Categorical | Instance-level dtype; `dtype == pl.Categorical` and tuple-membership checks return False. Fixed in 3 places over the campaign |
| Magic-number sentinel in ratio features | `LARGE_CONST=1e3` when denominator=0 — domain-specific decision; document or accept |
| Global-pool Polars Categorical dictionary | `Categorical.astype(str)` materializes a `(n_categories × max_str_len × 4B)` Unicode array — 75 GiB OOM observed in prod |
| Div-by-zero in ratio features | Mitigated by `pllib.clean_numeric(…)` when wrapped; raw `a / b` returns inf/NaN silently |
| Silent column overwrite | `create_date_features` clobbered user-engineered `date_year`. Fix: collision WARN |
| Target-encoder leakage | `fit_transform` on full sample before CV split. Classic ML antipattern |
| Degenerate classification target | All-one-class → ROC AUC=NaN → NaN-in-comparison breaks downstream |
| Schema drift train→val/test | Missing/extra cols or dtype change at transform time. Add WARN before the transform |
| Concurrent file write | `joblib.dump` without atomic rename → corruption on parallel runs |
| Entry-point third-party bugs | `thinc.util.fix_random_seed` passed un-clamped seed to numpy via pytest-randomly → session cascades |

### Observability discipline

WARN must fire on the *pathological* case, not normal operation. We audited
the 17 WARN-sites added across the campaign and only one (the
`get_pandas_view_of_polars_df` nested-type WARN) was noise-prone — it
fires per bridge call with the same schema. Fix: module-level dedup
cache keyed on `(col, dtype)` tuple. When adding a new WARN:

1. Can it fire on a clean default-config run with representative data? If yes → downgrade to INFO or gate.
2. Does the same WARN fire N times per run? If yes → dedup by shape.
3. Does the WARN name the trigger column/value? If no → add it, else the operator can't act.

### Test-infrastructure fixes

Third-party `pytest_randomly.random_seeder` entry points can break the
test session when their seed-setters don't clamp to `2**32`. Known
offenders:

- `thinc.util.fix_random_seed` (spaCy/explosion.ai dep). Symptom: `4
  passed, 20 errors` with `previous item was not torn down properly`
  in `tests/training/`. Root cause: pytest-randomly passes
  `randomly_seed + crc32(test_nodeid)` un-clamped. Fix: session-scoped
  autouse shim in `tests/conftest.py::_patch_thinc_fix_random_seed_for_pytest_randomly_compat`
  that wraps the callable with `% 2**32`. Verified with known-bad
  `--randomly-seed=310986334`.

### Perf budgets

Sensors that assert `elapsed < X s` or `RSS < Y MB` catch whole classes
of regressions that functional tests can't:

- `get_pandas_view_of_polars_df` on 500 k × 1 Categorical with 500 k uniques must finish < 5 s (`tests/training/test_utils.py::TestPolarsSliceDictionaryDiffers::test_high_cardinality_conversion_perf_budget`).
- `prepare_df_for_catboost` on a 50 k × 5 k-unique text column declared as text_feature must finish < 2 s (the pre-fix path hit minutes).
- `_convert_dfs_to_pandas` per-split timing is logged; a future regression that silently doubles conversion time would show up in the suite log diff.

Budgets should be generous (3–5× realistic time on a dev box) so CI
machine variance doesn't flake, but tight enough that a return to
O(n·k) from O(n) trips them.

## Phase timing

`train_mlframe_models_suite` instruments its hot paths with a lightweight
`PhaseTimer` (see `training/phases.py`). Every wrapped phase emits a `START`
and `DONE in Xs` line and accumulates into a process-local registry; at the
end of a verbose suite run, a ranked table of the top phases is logged, e.g.:

```
[phases] Top phases by wall-clock time:
phase                           total       calls    avg
--------------------------------------------------------
model.fit                       523.41s       1   523.410s
predict_proba                   187.12s       2    93.560s
fast_calibration_report          42.03s       2    21.015s
plot_feature_importances          4.11s       2     2.055s
load_and_prepare_dataframe        2.88s       1     2.880s
split_data                        0.91s       1     0.910s
```

Currently instrumented phases:

- `load_and_prepare_dataframe`, `split_data`, `initialize_training_defaults`,
  `trainset_features_stats`, `process_model` (suite level)
- `model.fit` (with `retry=True` on fallback), `pre_pipeline_fit_transform`,
  `compute_split_metrics` (train/val/test)
- `report_probabilistic_model_perf`, `report_regression_model_perf`,
  `predict`, `predict_proba`, `fast_calibration_report`,
  `plot_feature_importances`, `compute_fairness_metrics`

To instrument a new hotspot:

```python
from mlframe.training.phases import phase

with phase("my_operation", split="val", n_rows=len(df)):
    result = expensive_call(...)
```

The summary is only printed when the suite is called with `verbose=True`,
but phases are always recorded — inspect them programmatically via
`phase_snapshot()` or `format_phase_summary()`.

### Logging in Jupyter

If the root logger has no handlers, `train_mlframe_models_suite` installs
a minimal stdout handler at INFO level when `verbose=True`, so phase logs
actually appear in notebooks. If you've already called `logging.basicConfig`
or configured handlers yourself, nothing is touched.

## Troubleshooting

### Windows fatal exception / access violation in numba kernels

After changing `NUMBA_NJIT_PARAMS` flags (e.g. `cache=True`, `nogil=True`), stale on-disk numba caches (`.nbi`/`.nbc` in `__pycache__`) from a prior build can trigger `Windows fatal exception: access violation` inside `compute_numerical_aggregates_numba` / similar kernels. The flags themselves are correct — a cold rebuild is required. Clear and retry:

```bash
find . -name "*.nbi" -delete; find . -name "*.nbc" -delete
```

### XGBoost silent kernel death / 50× slow MakeCuts on large Polars frames

On large Polars frames (observed at ~7M rows × ~15+ `pl.Categorical` cat_features on Windows), XGB 3.2 with `enable_categorical=True` either:

- silently kills the Jupyter kernel between train- and val-IterativeDMatrix construction, or
- at smaller scales just runs ~50× slower in MakeCuts (0.9s vs 0.018s without cats).

Mitigation is on by default: `TrainingBehaviorConfig.align_polars_categorical_dicts=True` casts every `pl.Categorical` / `pl.Enum` cat_feature across train/val/test to a shared `pl.Enum(sorted(union_of_categories))` before XGB sees the frames. Shared Enum dict → consistent physical codes across Series → XGB takes a fast numeric-like path for categoricals, and the silent kill disappears.

Mechanism not fully isolated yet (see TODO). Disable the default via `behavior_config.align_polars_categorical_dicts=False` to reproduce the original behavior.

## Security notes

- `joblib.load` / `dill.load` in `inference.py`, `pipelines.py`, `training/io.py` are gated by a `trusted_root` path check and (for dill) a `_SafeUnpickler` allowlist. Never load pickle/joblib artifacts from untrusted sources.
- `torch.load` is always called with `weights_only=True`.
- SQL field names in `experiments.py` are validated against an allowlist.

See `CHANGELOG.md` (2026-04-14 entry) for the full audit/fix history.

## TODO

### Meta-test infrastructure — deferred for future iteration

**B1 — GitHub Actions workflow for meta-tests.** Currently the suite runs locally via the pre-commit hook. A standalone CI job with its own status badge and aggressive caching would surface drift in PR reviews directly, instead of requiring the reviewer to pull the branch and run pytest locally.

**B2 — Auto-PR for `_USER_DEFERRED_*` drain.** Monthly cron-agent that walks every `_USER_DEFERRED_*` / `_GRANDFATHERED` set across every meta-test, sorts by ease-of-fix, and opens a punch-list issue: "16 deferred items, 6 are easy, 7 need decisions, 3 need refactors". Without this, deferred items accumulate silently — the [test_deferred_drift.py](tests/test_meta/test_deferred_drift.py) tracker catches *growth* but doesn't suggest cleanup.

**G — Mutation testing on the meta-tests.** `mutmut run --paths-to-mutate src/mlframe/training/configs.py tests/test_meta/` would surface meta-tests that over-trust their inputs (e.g. an assertion that doesn't actually depend on the value being checked). Likely surfaces 5–15 weak spots; mostly low-priority but high-quality-bar.

### Post-hoc isotonic calibration wrapper
`_PostHocCalibratedModel` ([trainer.py:761-814](mlframe/training/trainer.py#L761-L814)) and the `_maybe_apply_posthoc_calibration` hook ([trainer.py:817-833](mlframe/training/trainer.py#L817-L833)) are currently unused — the hook is a no-op and nothing fits the wrapper. The class/hook are retained because the user may revive post-hoc isotonic calibration on a held-out eval set as an alternative to eval-metric-based early-stopping calibration. Before deleting, decide: (a) ship isotonic fitting on the eval_set predictions at the end of `_train_model_with_fallback`, then wrap the estimator, OR (b) remove all three definitions + the `_mlframe_posthoc_calibrate` attribute (no longer set anywhere after the 2026-04-18 fix).

### CatBoost `custom_metric` support
`helpers.py:234-244` has a commented-out `custom_metric=tuple(catboost_custom_classif_metrics)` entry for `CB_CLASSIF` / `CB_REGR`. The blocker: CatBoost mutates this parameter in-place post-init, breaking `sklearn.clone()` used by `RFECV`. Proposed fix: keep `CB_CLASSIF` / `CB_CALIB_CLASSIF` / `CB_REGR` clean (RFECV path stays cloneable), and attach `custom_metric` via `_cb_model.set_params(custom_metric=tuple(...))` on the base-path CatBoost instance only (after construction in `configure_training_params`, trainer.py ~L2215). This gives the base training its extra plotted metrics (AUC/BrierScore/PRAUC) without affecting RFECV. Upstream issue to file: https://github.com/catboost/catboost/issues — "sklearn.clone() fails on CatBoostClassifier constructed with custom_metric=tuple".

### Investigate `pl.Categorical` → `pl.Enum` cast as a general XGBoost speedup

Prod observation 2026-04-20 on a 7.3M × 114 frame (19 Categorical cat_features): casting every `pl.Categorical` to a shared `pl.Enum(union_of_categories)` before XGB fit drops `MakeCuts` wall-clock from **0.901962s to 0.018451s — ~50×** (the latter matches the no-categoricals baseline of 0.014539s). XGB appears to take a fast numeric-like bucketing path for `pl.Enum` but a slow per-chunk dict-reconciliation path for `pl.Categorical`.

Currently wired into the suite as `TrainingBehaviorConfig.align_polars_categorical_dicts=True` (default) primarily as a crash-avoidance measure. Beyond MakeCuts the end-to-end impact hasn't been measured, and the same pattern may apply to CatBoost, LightGBM, and HGB paths that also touch categoricals. Proposed work:

1. Benchmark `Categorical` vs `Enum` end-to-end training time for XGB / CB / LGB / HGB on a prod-shaped frame — is the 50× local win visible in total wall-clock or only during DMatrix construction?
2. If CB / LGB / HGB also benefit, push the Enum cast upstream of all strategies (currently only runs when mlframe knows the column is a cat_feature; could generalize to any Polars Categorical in the schema).
3. File upstream issues: xgboost (why is per-chunk Categorical 50× slower than Enum?) and polars (optional: can `pl.DataFrame` expose a cheap `.rechunk_and_consolidate_categoricals()` helper?).

Until (1) and (2) land the speedup is a pleasant side effect of the crash fix rather than a first-class optimization.

### Probability calibration: ship `prefer_calibrated_classifiers=True` as the default

The 2026-04-23 / 2026-04-24 prod logs consistently reported `CALIBRATIONs: MAEW=11-17%` on VAL across every model (CB, XGB, LGB) — the scores rank well (AUC 0.98+), but the predicted probabilities are mis-calibrated by 11–17 probability points. Anything downstream that consumes the probability value itself (expected-value sorting, threshold tuning, cost-sensitive routing, uncertainty-gated routing) sees skewed numbers; rank-only use cases (top-K screening) are fine.

mlframe already supports post-hoc isotonic calibration via `TrainingBehaviorConfig.prefer_calibrated_classifiers=True` (sklearn `CalibratedClassifierCV` under the hood). The field exists; the default is currently `False`. Flipping the default is the one-flag fix, but before doing so, the **placement of the calibration_set** needs careful thought — it's a live trade-off, not a settled question:

- **Option A — held-out slice of train**: the sklearn-standard path. `CalibratedClassifierCV(cv=...)` internally k-folds the training data, fits per-fold models, and fits the isotonic map on their out-of-fold predictions. Zero extra config for the user. Downside: on strongly time-indexed data (prod_jobsdetails is 22-year train / 14-month test), a random-k-fold calibration set is temporally mixed — the isotonic map learns on "mostly-past" data while the deployment distribution is "strictly-future". Calibration can drift with the same distance-under-drift artifact that motivated `val_placement="backward"`.
- **Option B — explicit `calibration_df` drawn from the end of train (or a separate holdout)**: closer to deployment distribution; avoids the temporal-leak of in-train k-fold. Downside: carves further rows out of `train_df`, shrinking the learn set; needs a new config field (`calibration_size: float` or `calibration_df: DataFrame`) and wiring through `splitting.py`.
- **Option C — piggyback on `val_df` (the set already used for early stopping)**: conceptually clean, but then the calibration map is fit on the very metric signal that decided early-stopping, double-counting one set's information. Risk of overconfident calibration on VAL that doesn't generalize.
- **Option D — add a separate `TrainingSplitConfig.calibration_size` that carves a small contiguous slice just before test**: temporally closest to deployment, no contamination of train or val. New config field + propagation through the splitting function. This is probably the right answer but needs design review.

Before flipping the default: decide A/B/C/D, add the tests, measure MAEW reduction end-to-end on the prod frame (the 11–17% should drop into single digits if calibration is placed correctly), and document the placement rule in `TrainingBehaviorConfig.prefer_calibrated_classifiers` docstring. Don't ship the flag-flip without a placement decision — a default calibration that makes things worse on drift-heavy data is worse than no calibration.

### Persist ensemble member predictions for post-hoc inference reuse

Currently `score_ensemble` runs on-the-fly from in-memory fitted members and discards per-member VAL / TEST `probs` arrays after the ensemble score is logged. If you want to re-run ensemble evaluation on new subsets, compare different ensemble methods later, or use the ensemble in an offline scoring pipeline without re-fitting all members, you need the per-member probs saved to disk alongside the `.dump` model files.

Proposed format: one parquet-or-numpy file per (model, weight_schema) recording `val_probs` / `test_probs` / row index → stable under `data_dir/<target>/<model>/preds/`. `score_ensemble` would gain an `optional from_disk=True` mode that reads those instead of re-computing.

**Leakage warning — load-bearing, do NOT skip**: the natural follow-up "automatically pick the best ensemble method by TEST metric" is **data leakage**. TEST is the deployment proxy; using TEST metrics to choose between ensemble methods (arithm / harm / median / geo / etc.) bleeds TEST information into model selection, and the reported TEST metric is no longer an honest estimate of deployment performance. The persistence feature must expose ensemble probs + metrics but **not** offer a "pick the best by TEST" selector. Method choice must happen on VAL (or a separate hold-out), with TEST computed only once for the chosen method. Add an explicit comment in the persistence loader and in the README note above: "Select ensemble method by VAL metric; then compute TEST once for the chosen method. Comparing TEST across methods and picking the argmax is leakage."

