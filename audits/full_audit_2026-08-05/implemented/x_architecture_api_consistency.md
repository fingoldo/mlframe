# Cluster: x_architecture_api_consistency

**Scope:** Public API surface across `src/mlframe` — every top-level `__init__.py`'s exports
(`__all__`), and constructor signatures / parameter-naming conventions of the major public
estimator/transformer classes across `feature_engineering`, `feature_selection/wrappers`,
`training`, `calibration`, `preprocessing`, `models`. Excludes
`feature_selection/filters/**` (MRMR engine) and `feature_selection/shap_proxied_fs/**` per
the audit brief (separately, fully audited 2026-07-25).

**Files reviewed:** 72 `__init__.py` files enumerated (excluding `_benchmarks/`); of those, the
~40 non-implementation `__init__.py` files (root + every subpackage facade: `calibration`,
`competition`, `core`, `data`, `data_valuation`, `estimators`, `evaluation`,
`feature_engineering` (+`transformer`), `feature_selection` (+`wrappers`, +`wrappers/rfecv`,
+`boruta_shap`), `inference`, `inspection`, `integrations`, `metrics` (+`calibration`,
+`classification`, +`regression`), `models` (+`ensembling`), `preprocessing`, `reporting`
(+`charts`, +`renderers`), `signal`, `system`, `testing`, `training` (+`baselines`,
+`callbacks`, +`cb`, +`composite`, +`core`, +`diagnostics`, +`extractors`,
+`feature_handling`, +`neural`+`base`, +`pipeline`, +`ranking`, +`reporting`, +`slicing`,
+`strategies`, +`targets`), `utils`, `votenrank`) were read in full for exports/`__all__`
curation. In addition, ~85 public `BaseEstimator`-derived classes (all `class .*BaseEstimator`
/ `TransformerMixin` / `RegressorMixin` / `ClassifierMixin` hits outside the excluded
directories) were enumerated and their `__init__`/`fit`/`predict` signatures inspected via
targeted `grep`/`Read`, with the entire `training/composite/*.py` family (35 sibling
`BaseEstimator` classes), the `training/neural/*.py` family (7 sklearn-shaped estimators), and
`feature_selection/{wrappers/rfecv,boruta_shap,hybrid_selector,ace}.py` read closely for
parameter-naming and `fit`/`sample_weight` signature comparison. Import-path claims in the root
package docstring were verified empirically by executing every documented import line against
the installed source tree.

**LOC reviewed (approximate):** ~9,500 lines read directly (root `__init__.py` 429 + the ~40
facade `__init__.py` files totaling ~6,800 + `RFECV.__init__`/docstring (~250) +
`BorutaShap.__init__`/docstring (~200) + `MLPRanker`/`ShortlistTransformerAdapter`/neural
estimator constructors (~350) + `training/composite` fit/predict signature sampling across 35
files (~450) + `calibration/post.py` `BinaryPostCalibrator` (~150) + `target_encoders.py` /
`classification_discovery.py` excerpts (~150)); plus ~450 `grep` matches triaged across the
full non-excluded tree (~217k LOC) for `random_state`/`seed`/`rng`, `verbose`, `n_jobs`,
`def __init__`, `def fit`, `def predict`, `get_params`/`set_params`/`__sklearn_clone__` to
build the cross-file consistency picture that grounds each finding below.

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|---|---|---|---|---|---|
| X_ARCHITECTURE_API_CONSISTENCY-1 | P1 | `src/mlframe/__init__.py:9-13` | 3 of the 8 "public API convention" example imports in the root docstring are broken (`ImportError`) against the current tree: `MRMR` isn't exported from `mlframe.feature_selection`, `expected_calibration_error` doesn't exist anywhere in `mlframe.metrics.*` under that name, and `predict_from_models` lives in `mlframe.training.core`, not `mlframe.inference.predict`. | Fix the three example lines to real, currently-working paths (`mlframe.feature_selection.filters.MRMR`; the actual ECE symbol, e.g. `compute_ece_debiased`/`compute_ece_brier_full_and_debiased` in `mlframe.metrics.calibration`; `mlframe.training.core.predict_from_models` or re-export it under `mlframe.inference.predict`), and add the regression test below so this docstring can't silently rot again. | A test that regex-extracts every `from mlframe... import ...` line from `mlframe/__init__.py`'s module docstring and `exec()`s each one, failing on any `ImportError`; generalize to scan every package docstring in the repo for the same pattern. |
| X_ARCHITECTURE_API_CONSISTENCY-2 | P1 | `training/composite/direct_multi_horizon.py:196`, `dual_direction.py:96`, `orthogonal.py:183`, `per_group_router.py:82`, `regime_split_ensemble.py:74`, `segmented_model_factory.py:129`, `feature_subset_bagging.py:149`, `gated_regression_mixture.py:129`, `meta.py:173`, `simplex.py:267` (10 of 35 `training/composite/*.py` `BaseEstimator+RegressorMixin` siblings) | `fit(self, X, y)` on these 10 sibling composite regressors has no `sample_weight` parameter and no `**kwargs` catch-all, while 11+ siblings in the same package (`classification.py:163`, `count_weighted_blend.py:144`, `gated_outlier.py:101`, `glm.py:253`, `grouped_block_stacking.py:174`, `missing.py:170`, `qrf.py:260/413/492`, `segment_routed.py:151`) do accept it — so a caller that threads `sample_weight` uniformly across the composite-estimator family (the pattern `mlframe`'s own `SampleWeights` config exists to drive) hard-crashes with `TypeError` on exactly these 10, with no upfront way to tell which composite estimators support weighting short of reading each source file. | Either add `sample_weight: Optional[np.ndarray] = None` to the 10 listed `fit()` signatures (threading it into whatever inner CV-fold/`kf.fit` call each already makes), or — if weighting is genuinely inapplicable to a given estimator's math (e.g. a CV-index-only router) — say so explicitly in the class docstring and raise a clear `NotImplementedError` on a non-None `sample_weight` rather than a generic `TypeError: unexpected keyword argument`. | An AST scanner over every `class *(BaseEstimator, *RegressorMixin*)`/`ClassifierMixin` in a shared "estimator family" directory (grouped by directory, since `training/composite/` is explicitly documented as a single family of pluggable regressors): collect each `fit()` signature's param names, flag any class in the group missing `sample_weight` when >50% of its siblings have it, unless the class docstring contains an explicit "does not support sample_weight" marker. |
| X_ARCHITECTURE_API_CONSISTENCY-3 | P2 | `training/neural/ranker.py:605,617,918`; `feature_engineering/transformer/_suite_adapter.py:62,68,99` | `MLPRanker` and `ShortlistTransformerAdapter` — both `sklearn.base.BaseEstimator` subclasses — name their RNG-seeding constructor parameter `seed`, while every other RNG-seeding `BaseEstimator`/`TransformerMixin` subclass in the non-excluded tree (~40 hits: `BorutaShap`, `RFECV`, every `training/composite/*.py` regressor, `FieldGroupedMLPRegressor`, `Tabular1DCNNRegressor/Classifier`, `TrunkResidualMLPRegressor`, `GaussianMixtureClassifier`, `MyDecorrelator`, etc.) uses the sklearn-standard `random_state`. | Rename `seed` to `random_state` on both classes' `__init__` (keep a deprecated `seed` alias forwarding with a `DeprecationWarning` for one release given these are public, importable classes) so `get_params()`/`set_params(random_state=...)`-based tooling (e.g. `sklearn.model_selection` utilities and any generic hyperparameter search that special-cases `random_state`) works uniformly across the whole estimator surface. | A grep/AST scanner over every `BaseEstimator` subclass's `__init__` parameter list: flag any RNG-related parameter name (`seed`, `rng`, `random_seed`) that differs from `random_state` when the same package/family already has 3+ siblings using `random_state`. |
| X_ARCHITECTURE_API_CONSISTENCY-4 | P2 | `training/neural/ranker.py:585,600,612,918` vs `field_grouped_mlp.py:70,74`, `tabular_1dcnn.py:149,152,213,216`, `trunk_residual_mlp.py:82,85` | `MLPRanker`'s epoch-count constructor parameter is named `n_estimators` (docstring: "n_estimators : epochs (default 100)"; used directly as `max_epochs=self.n_estimators` at line 918), while its three sibling sklearn-shaped neural regressors in the same `training/neural/` package (`FieldGroupedMLPRegressor`, `Tabular1DCNNRegressor`, `Tabular1DCNNClassifier`, `TrunkResidualMLPRegressor`) all name the identical concept `n_epochs`. `n_estimators` is also a loaded sklearn name (number of base learners in an ensemble, e.g. `RandomForestRegressor`/`GradientBoostingRegressor`) — reusing it for "training epochs" on a single neural network is actively misleading, not just inconsistent. | Rename `MLPRanker.n_estimators` to `n_epochs` (deprecated-alias forward for one release, as it's a public constructor param). | Grep for `n_estimators` on any class that is not tree/boosting-ensemble-shaped (no `estimator_`/`estimators_` list of fitted sub-models) and cross-check the docstring for words like "epoch"/"iteration" instead of "tree"/"base learner". |
| X_ARCHITECTURE_API_CONSISTENCY-5 | P3 | `calibration/__init__.py` (entire file, 63 lines) | `calibration/__init__.py` never defines `__all__`, unlike every sibling subpackage facade that follows the documented "curate the star-import surface explicitly" convention (`preprocessing/__init__.py:36`, `models/__init__.py:24`, `data/__init__.py:15`, `estimators/__init__.py:21`, `inference/__init__.py:23`, `utils/__init__.py:35`, `core/__init__.py:36`, all `__all__ = sorted(name for name in globals() if not name.startswith("_"))`). It does define a PEP 562 `__dir__()` for introspection of the lazily-resolved `quality`/`probabilities` symbols, but that doesn't substitute for `__all__` controlling `from mlframe.calibration import *`, and static tools (docs generators, IDEs, `pydoc`) that read `__all__` will see only the 12 eagerly-imported names, silently omitting anything reachable only via the lazy `__getattr__`. | Add `__all__` mirroring the sibling pattern for the eager names, and document (or extend `__all__` dynamically, matching the existing `__dir__` logic) that the lazily-resolved `quality`/`probabilities` symbols are intentionally excluded from `__all__` to avoid eagerly importing their heavy deps. | A meta-test that imports every `mlframe.*` subpackage `__init__.py` and asserts `hasattr(module, "__all__")` whenever the module docstring or sibling packages in the same tree level follow the "curate explicitly" convention. |
| X_ARCHITECTURE_API_CONSISTENCY-6 | P3 | `calibration/post.py:126` (`BinaryPostCalibrator.__init__`) vs the ~85 other `BaseEstimator` subclasses in scope | Two competing, undocumented idioms coexist for storing constructor params on public sklearn estimators: the overwhelming majority (`BorutaShap`, every `training/composite/*.py` class, every `training/neural/*.py` class, etc.) does explicit `self.param = param` per line; a handful (`BinaryPostCalibrator`, `RFECV`, `PytorchLightningEstimator`, `EarlyStoppingWrapper`, `MLPRanker`'s (out-of-scope) `_ice_metric`/`_optimization_search` cousins) use the reflection-based `store_params_in_object(obj=self, params=get_parent_func_args())` helper. Both are proven to satisfy `get_params()`/`clone()` in this codebase, so this is not a functional bug, but it is a real, silent architectural fork with no comment anywhere explaining when a new estimator should reach for which idiom, and the reflection-based one is materially harder to grep/statically-analyze (`self.<param>` never appears as text in the file). | Add a one-paragraph convention note (e.g. in `estimators/base.py` or a `CONTRIBUTING`-style doc) stating which idiom new public estimators should use, and why `store_params_in_object` is reserved for a specific case (e.g. very-high-arity constructors where per-line assignment is the bulk of the file). | A grep-based check that any new file adding a `class *(BaseEstimator...)` either has as many `self.<param> = <param>` lines as `__init__` params, or calls `store_params_in_object`/`get_parent_func_args` — flags a class matching neither pattern (a genuinely missed param, the real bug class this convention exists to prevent). |
| X_ARCHITECTURE_API_CONSISTENCY-7 | P3 | Repo-wide, e.g. `calibration/_post_train_calibrators.py:36`, `core/helpers.py:137/163/183`, `feature_engineering/bruteforce.py:86`, `feature_selection/general.py:84/128` (`verbose: int`) vs `estimators/baselines.py:33`, `feature_engineering/timeseries.py:520/652`, `feature_selection/boruta_shap/__init__.py:144`, `preprocessing/cleaning.py:132/199/509`, `preprocessing/outliers.py:46`, `models/ensembling/predict.py:120/309`, `models/ensembling/score.py:99` (`verbose: bool`) | `verbose` is typed `bool` in some public functions/classes and `int` (0/1/2 level) in others, with no discernible pattern by subsystem — `feature_engineering` alone has both (`bruteforce.py` uses `int`, `timeseries.py` uses `bool`), as does `preprocessing`/`feature_selection`. Both conventions are individually reasonable (a flag vs. a verbosity level), but the split is not documented anywhere and callers moving between sibling functions in the same module family have no way to predict which one a given call site expects without checking the signature. | Standardize on one convention repo-wide (an `int` level is the more expressive superset — `bool` truthiness still works against `verbose > 0` checks) or, at minimum, document in each subsystem's `__init__.py` docstring which convention its own functions follow. | A grep/AST scanner over every `def __init__`/top-level function with a `verbose` parameter, collecting its declared type per subpackage; flag any subpackage where both `bool` and `int` appear among its own public functions. |
| X_ARCHITECTURE_API_CONSISTENCY-8 | P3 | `feature_selection/wrappers/rfecv/__init__.py:163-232` (`RFECV.__init__`, ~70 named parameters) vs `feature_selection/boruta_shap/__init__.py:120-147` (`BorutaShap.__init__`, ~26 flat params), `feature_selection/hybrid_selector.py:162-172` (`HybridSelector.__init__`, ~20 flat params), `feature_selection/ace.py` (`ACESelector`, flat params) | `RFECV` is the only class in the `feature_selection`/`wrappers` sibling family that mixes three grouped pydantic config objects (`search_config`, `fi_config`, `robustness_config`) with ~60 additional flat kwargs for the same/overlapping settings (explicitly documented as intentional back-compat in the constructor's own comment at line 166-168), while every sibling selector (`BorutaShap`, `HybridSelector`, `ACESelector`) uses purely flat kwargs. This is a deliberate, documented design choice, not an oversight, but it means the "config vs flat-kwargs" convention differs across the one family where the review brief specifically asks about it, and `RFECV`'s ~70-parameter constructor is a genuine outlier in constructor width across the whole audited surface. | No functional change needed given the documented back-compat rationale; consider a short note in `feature_selection/__init__.py`'s module docstring flagging `RFECV` as the sole hybrid-config estimator in the family, so future selectors default to the (simpler, consistent) flat-kwargs pattern the rest of the family already uses. | N/A (documented, intentional design choice) — a meta-test isn't warranted here beyond a lint rule warning on any *new* class exceeding ~40 constructor parameters, prompting a design review rather than blocking. |

**Findings by severity:** P0: 0, P1: 2, P2: 2, P3: 4. Total: 8.

## Narrative detail

**X_ARCHITECTURE_API_CONSISTENCY-1 (P1).** The root `mlframe/__init__.py` module docstring
opens with "Public API convention: deep-import from subpackages, not from this top-level
module" and gives 8 example import lines as the canonical way to reach mlframe's ~1k
entry-point symbols across 15 subpackages. I executed every one of those 8 lines directly
against the installed source tree (`python -c "from mlframe.feature_selection import MRMR,
RFECV"` etc.) rather than trusting a source read, since import-path claims are exactly the
kind of thing that silently rots after a refactor. 3 of 8 fail: `MRMR` is not exported from
`mlframe.feature_selection` (it lives at `mlframe.feature_selection.filters.MRMR` — confirmed
importable there); `expected_calibration_error` does not exist anywhere under that name in
`mlframe.metrics.*` (grepped the whole `metrics/` tree for `def expected_calibration_error` —
zero hits; the closest live symbols are `compute_ece_debiased` /
`compute_ece_brier_full_and_debiased` in `metrics/calibration/_calibration_metrics.py`); and
`predict_from_models` is defined in `training/core/_predict_main_from_models.py`, not
`mlframe.inference.predict` (confirmed via `grep -rn "def predict_from_models"`). This is the
single most load-bearing piece of documentation in the package — a new contributor or
downstream user has no other authoritative map of the 15-subpackage surface — and getting
3/8 of its own worked examples wrong actively misdirects exactly the audience it's written for.

**X_ARCHITECTURE_API_CONSISTENCY-2 (P1).** `training/composite/` is documented (module
docstring) as a family of interchangeable composite-target regressors meant to be plugged into
the same discovery/suite machinery. I enumerated every `def fit(self, ...)` across the 35
`BaseEstimator`-derived classes in that directory and cross-referenced which accept
`sample_weight`. 10 do not (no `sample_weight` param and no `**kwargs` catch-all that could
absorb it), while 11+ siblings in the same directory do, including ones solving structurally
similar problems (e.g. `GatedOutlierEstimator.fit` at `gated_outlier.py:101` has
`sample_weight`, but its sibling `GatedRegressionMixture.fit` at
`gated_regression_mixture.py:129` does not, despite both being gated-mixture-style composite
regressors). I confirmed this is a known gap rather than an oversight I'm mis-weighting: the
codebase already has a dedicated `_model_fit_accepts_sample_weight()` guard using
`sklearn.utils.validation.has_fit_parameter` in `training/composite/post_shim.py:34-51`,
built specifically to avoid crashing when an arbitrary *wrapped* inner model doesn't support
`sample_weight` — but that guard is local to `PrePipelinePredictShim` and does not cover the
top-level composite estimators themselves, several of which mlframe's own `SampleWeights`
config (exported at the very top of `mlframe/__init__.py`) exists to drive. A caller that
threads `sample_weight` uniformly across this family gets a hard `TypeError` for the 10 listed
classes with no upfront signal which ones support it short of reading source.

**X_ARCHITECTURE_API_CONSISTENCY-3 (P2).** Grepping every `BaseEstimator`/`TransformerMixin`-
derived class's `__init__` for RNG-seeding parameters across the non-excluded tree turned up
~40 classes using `random_state` (sklearn's own convention) and exactly two outliers —
`MLPRanker` (`training/neural/ranker.py:605`) and `ShortlistTransformerAdapter`
(`feature_engineering/transformer/_suite_adapter.py:62`) — using `seed` instead, despite both
being genuine `sklearn.base.BaseEstimator` subclasses (confirmed via `class MLPRanker(
BaseEstimator, RegressorMixin):` and `class ShortlistTransformerAdapter(BaseEstimator,
TransformerMixin):`). This matters beyond cosmetics: sklearn's own ecosystem (some
meta-estimators, `check_estimator`-style compatibility probes, generic hyperparameter-search
code that special-cases `random_state` for reproducibility bookkeeping) looks for that literal
name via `get_params()`; a caller doing `set_params(random_state=0)` on either of these two
classes silently no-ops (no exception — `set_params` just sets an attribute that the class
never reads) rather than seeding anything, since the constructor never registered
`random_state` as a param name.

**X_ARCHITECTURE_API_CONSISTENCY-4 (P2).** `MLPRanker`'s own docstring (`training/neural/
ranker.py:585`) glosses its `n_estimators` parameter as "epochs (default 100)", and the
constructor threads it straight into `max_epochs=self.n_estimators` at line 918 inside the
Lightning `Trainer` setup — it is unambiguously an epoch count, not a count of base learners.
Its three siblings in the exact same `training/neural/` package (`FieldGroupedMLPRegressor`,
`Tabular1DCNNRegressor`/`Tabular1DCNNClassifier`, `TrunkResidualMLPRegressor`) all name the
identical "how many training epochs" concept `n_epochs`. Beyond being an internal
inconsistency, `n_estimators` already carries a well-known, different sklearn meaning (number
of trees/boosting rounds in `RandomForestRegressor`, `GradientBoostingRegressor`, etc.) — a
caller porting hyperparameter-search code or intuition from a tree ensemble to `MLPRanker`
would reasonably expect `n_estimators=100` to mean "100 small networks", not "100 epochs of one
network".

**X_ARCHITECTURE_API_CONSISTENCY-5 (P3).** `calibration/__init__.py` is the only package facade
in scope that never defines `__all__` — every sibling (`preprocessing`, `models`, `data`,
`estimators`, `inference`, `utils`, `core`) follows the same one-line curation idiom
(`__all__ = sorted(name for name in globals() if not name.startswith("_"))`), itself called out
in those files' own comments as mirroring a shared convention ("Curate the star-import surface
explicitly (mirrors mlframe.metrics.__init__'s pattern)"). `calibration/__init__.py` has a good
reason to special-case introspection (its `quality`/`probabilities` submodules are lazily
resolved via PEP 562 `__getattr__`/`__dir__` to avoid a ~2s matplotlib/properscoring import
chain), but that reasoning applies to `__dir__`, not to omitting `__all__` for the dozen names
it *does* import eagerly at module load — those could still be curated like every sibling.

**X_ARCHITECTURE_API_CONSISTENCY-6 (P3).** While auditing constructor bodies for sklearn
`__init__`-contract compliance (no mutation/validation of stored params), I found
`BinaryPostCalibrator.__init__` (`calibration/post.py:126`) uses
`store_params_in_object(obj=self, params=get_parent_func_args(), postfix="")` — a pyutilz
frame-introspection helper that reads the caller's local variables and assigns them to
`self` — rather than the explicit `self.param = param` pattern used by the ~80 other public
estimators in scope. Grepping for `store_params_in_object` confirmed it is not a one-off: it is
also used by `RFECV.__init__`, `PytorchLightningEstimator.__init__`
(`training/neural/base/__init__.py`), and `EarlyStoppingWrapper.__init__`
(`estimators/early_stopping.py`), so both idioms are established, working, and get-`params`
compliant in this codebase — I verified this is not itself a bug. It is, however, an undocumented
architectural fork: nothing in the repo says which pattern a new public estimator should follow,
and the reflection-based one is materially harder to `grep`/statically verify for "did every
constructor param actually get stored" (the exact bug class the audit brief calls out under
"mutable-default-argument bugs" and similar param-handling correctness issues).

**X_ARCHITECTURE_API_CONSISTENCY-7 (P3).** Collecting every `verbose` parameter's declared
type across the non-excluded tree shows a near-even split between `bool` and `int` (0/1/2-style
level) with no correlation to subsystem: `feature_engineering/bruteforce.py:86` uses `verbose:
int = 1` while `feature_engineering/timeseries.py:520` and `:652` use `verbose: bool = False`
in the same package; `preprocessing/cleaning.py` and `preprocessing/outliers.py` use `bool`
while `feature_selection/general.py` uses `int`. Neither convention is wrong in isolation, but
a caller moving between sibling functions in the same subsystem (the exact scenario this
cluster is meant to catch) cannot predict which one a given signature expects without opening
the source.

**X_ARCHITECTURE_API_CONSISTENCY-8 (P3).** `RFECV.__init__` (`feature_selection/wrappers/
rfecv/__init__.py:163-232`) accepts three grouped pydantic config objects (`search_config`,
`fi_config`, `robustness_config`) *in addition to* roughly 60 flat keyword arguments covering
overlapping settings — a hybrid pattern explicitly justified in an inline comment ("Grouped
pydantic configs. When passed, their non-None fields override matching flat kwargs. All flat
kwargs are kept for back-compat AND because some power-users want flat call-sites"). Every
sibling selector in the same `feature_selection` family — `BorutaShap` (~26 flat params),
`HybridSelector` (~20 flat params), `ACESelector` — uses pure flat kwargs with no config-object
option. This is a deliberate, well-documented design choice rather than an oversight, so I am
not proposing a functional change, but it is exactly the "config vs flat-kwargs across sibling
classes" divergence the audit brief asks to surface, and `RFECV`'s ~70-parameter constructor
is the single widest public constructor found anywhere in scope.

## Dimensions with no findings

- **ML correctness (leakage / calibration correctness / class-imbalance handling)**: no
  leakage or calibration-correctness issues were found within this cluster's narrow scope
  (public API surface + constructor/signature consistency); ML-correctness bugs inside
  individual algorithms are covered by other clusters' deeper per-module review.
- **Computational efficiency**: out of scope for this cluster (API-surface/signature review
  does not exercise runtime hot paths); no efficiency findings are claimed here.
- **Edge cases / robustness (empty input, all-NaN, single-class, etc.)**: not applicable to
  this cluster's dimension (constructor/export surface, not runtime input handling).
- **Test coverage gaps**: no dedicated coverage-gap findings beyond the meta-test ideas attached
  to each finding above; a full test-inventory sweep is out of this cluster's scope.
- **Security**: no security-relevant findings in the public API surface reviewed.
