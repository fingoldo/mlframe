# models_estimators

Files reviewed: 24 of 44 non-benchmark `.py` in the cluster | LOC: 11,543 (cluster total, non-benchmark)

## Summary

Most hot kernels here are already `@njit`/`prange`/KTC-gated, so there is no low-hanging perf work; the one perf
finding below is a combinatorial-explosion bug, not a kernel. The serious problems are silent correctness: three
DEFAULT paths produce wrong or NaN answers with no error. Every finding below was reproduced by running the code,
not inferred from reading.

## Findings

### MODELS_ESTIMATORS-1 [P0] silent-wrong-result-class-labels

**File:** `src/mlframe/models/ensembling/selection.py` :137

**Summary:** `_score_blend`'s default metric feeds raw class labels straight into `fast_roc_auc`, whose njit kernel
does `tps += y_true[i]` and therefore requires strictly 0/1 labels -- any other binary encoding silently yields
NaN and a garbage ensemble.

**Failure scenario:** `caruana_greedy_selection(stacked, y, max_picks=10)` with `metric=None` (the documented
default) and y encoded `{1,2}` or `{-1,+1}`. Reproduced: with `y in {0,1}` the walk returns `score=1.0`,
`weights=[1,0]` (correct model picked); with the identical predictions and `y+1` or `2y-1` it returns `score=nan`.
Because `nan > best` is always False, every greedy comparison fails and the bag degenerates to whichever model
index came first -- no exception, no warning, and the caller gets a well-formed-looking result.

**Suggested fix:** Binarise before scoring: derive the positive label explicitly
(`classes = np.unique(y); pos = classes[-1]`) and pass `(y == pos).astype(np.int64)`, or raise a clear
`ValueError` when `np.setdiff1d(np.unique(y), [0, 1])` is non-empty. The current `y.astype(np.int64)` converts
dtype but not encoding. Add an `np.isfinite(score)` guard in the greedy loop so a NaN metric aborts loudly.

**Evidence:** `metrics/_core_auc_brier.py` :749-750 (`tps += y_true[i]`, `fps += 1 - y_true[i]`); `fast_roc_auc`
:558-582 performs no 0/1 validation; direct run on the three encodings.

### MODELS_ESTIMATORS-2 [P0] silent-wrong-result-generator-exhaustion

**File:** `src/mlframe/models/additive_interaction_diagnostic.py` :96

**Summary:** `_cv_score` iterates the caller's `cv_splits` on every invocation, but the parameter is documented as
"Iterable of `(train_idx, test_idx)` index pairs" -- pass the natural `KFold(...).split(X)` generator and every
call after the first sees an exhausted iterator, producing `np.mean([]) == nan` instead of an error.

**Failure scenario:** `additive_interaction_diagnostic(X, y, KFold(3).split(X), r2)`. Reproduced on synthetic
interaction data: with a generator the result is `additive_model_cv_score=nan`, `additive_signal_ratio=nan`,
`recommend_interaction_engineering=False`; with `list(KFold(3).split(X))` it is `0.804`, `0.835`, `True`. The
diagnostic's whole point -- the recommendation flag -- flips, with only a suppressible numpy RuntimeWarning. With
`per_feature_report=True` all `2 * n_features` further calls are NaN too, so the entire per-feature table is NaN.

**Suggested fix:** `cv_splits = list(cv_splits)` at the top of `additive_interaction_diagnostic`, and raise in
`_cv_score` when `fold_scores` is empty rather than returning `float(np.mean([]))`.

**Evidence:** `_cv_score` is called at :113, :114 and twice per feature at :141-142, each time doing
`for train_idx, test_idx in cv_splits`; empirical A/B above.

### MODELS_ESTIMATORS-3 [P0] early-stopping-is-a-no-op

**File:** `src/mlframe/estimators/early_stopping.py` :243

**Summary:** The `staged` backend "truncates" the snapshot with `set_params(n_estimators=best_stage)`, but
sklearn's `GradientBoosting*.predict` walks the fitted `self.estimators_` array and ignores the `n_estimators`
hyperparameter -- so `best_model_` silently predicts with the full `max_iter` ensemble. Early stopping has no
effect whatsoever.

**Failure scenario:** `EarlyStoppingWrapper(base_model=GradientBoostingClassifier(), max_iter=60, patience=3)`.
Reproduced: `best_model_.get_params()['n_estimators'] == 1` while `len(best_model_.estimators_) == 60`, and
`predict_proba` is bit-identical to the untruncated model. Every caller relying on this wrapper to prevent
overfitting gets the fully-overfit model back, while `best_score_`/`n_iterations_` report a plausible early stop.

**Suggested fix:** Truncate the fitted arrays, not the hyperparameter: after the deepcopy, slice
`best_model_.estimators_ = best_model_.estimators_[:best_stage]` and set `n_estimators_`/`n_estimators`
consistently. Alternatively snapshot lazily by storing `best_stage` and wrapping `predict`/`predict_proba` to
call `staged_predict*` and take element `best_stage - 1`. Add a regression test asserting `predict_proba` differs
from the full-budget model when an early stop fired.

**Evidence:** sklearn 1.8.0, empirical checks above; `set_params` on a fitted estimator affects only a subsequent
warm-start refit.

### MODELS_ESTIMATORS-4 [P1] sklearn-api-compliance-mixin-mro

**File:** `src/mlframe/estimators/base.py` :142, :152; `src/mlframe/estimators/custom.py` :289, :314, :350, :695,
:701

**Summary:** Every custom estimator declares its sklearn mixin AFTER a base that already inherits `BaseEstimator`,
which puts `BaseEstimator.__sklearn_tags__` ahead of the mixin's in the MRO -- so under sklearn >= 1.6
`is_classifier` and `is_regressor` both return **False** for all of them.

**Failure scenario:** Verified on sklearn 1.8.0: `ArithmAvgClassifier`, `GeomAvgClassifier`,
`PureRandomClassifier`, `IdentityClassifier`, `IdentityRegressor`, `ClassifierWithEarlyStopping`,
`RegressorWithEarlyStopping` all report `is_clf=False is_reg=False`. Concrete in-cluster breakage:
`estimators/baselines.py` :47-54 (`get_best_dummy_score`) raises `TypeError: estimator must be a sklearn
classifier or regressor`. Wider: `GridSearchCV`/`cross_val_score` silently pick plain `KFold` instead of
`StratifiedKFold` for these classifiers, and default-scoring resolution picks the wrong metric.

**Suggested fix:** Put the mixin first at every declaration
(`class RegressorWithEarlyStopping(RegressorMixin, EstimatorWithEarlyStopping)`, etc.) -- sklearn's own
estimators use mixin-first for exactly this reason. Add a parametrised test asserting `is_classifier` /
`is_regressor` for the whole set.

**Evidence:** MRO dump plus `is_classifier`/`is_regressor` run against the installed sklearn 1.8.0;
`RegressorMixin.__sklearn_tags__` calls `super().__sklearn_tags__()` and is never reached when `BaseEstimator`
precedes it.

**Disposition:** RESOLVED. 66 class declarations reordered so sklearn mixins precede `BaseEstimator`. Two of them then broke: `MRMR` and `ShapProxiedFS` had a project mixin supplying `_get_support_mask`/`transform` that the sklearn mixin must NOT precede, leaving `MRMR` abstract. Both corrected, and every reordered class was audited by rebuilding its old base tuple and diffing attribute resolution: only `__sklearn_tags__` and the `__init_subclass__` entry point changed, and both sklearn chains still run. `tests/test_meta/test_sklearn_mixins_come_first.py`.

### MODELS_ESTIMATORS-5 [P1] caller-array-side-effect

**File:** `src/mlframe/models/ensembling/predict.py` :120

**Summary:** `_compute_outlier_gate` sets `p.flags.writeable = False` on the CALLER's own member-prediction arrays
as a staleness guard for its `id()`-keyed cache, and only restores writeability on LRU eviction or an explicit
`_clear_gate_cache()` -- so the public `ensemble_probabilistic_predictions` permanently freezes its inputs, which
its docstring never mentions.

**Failure scenario:** `ensemble_probabilistic_predictions(a, b, c)`. Reproduced: `a.flags.writeable` is True
before and False after; a subsequent caller-side in-place operation raises `ValueError: assignment destination is
read-only` with a traceback pointing at the caller, far from the call that caused it. Secondarily the cache holds
strong references to up to 16 complete member-prediction sets for the process lifetime -- at the 9M-row /
6-member scale the module's own comments cite (~2.2 GB per set) that is a large, invisible retention.

**Suggested fix:** Replace the `id()` key plus freeze with a content fingerprint the cache can validate cheaply,
and drop the `writeable = False` mutation. If the freeze must stay, restore writeability before returning and
document the contract; hold weakrefs (or only shapes) rather than the arrays.

**Evidence:** :118-121 (freeze on insert), :57-68 `_unfreeze_gate_cache_entry` called only from
`_clear_gate_cache` and the eviction branch at :129-133; the public docstring :151-183 makes no mention.

**Disposition:** RESOLVED. `_member_fingerprint` replaces the `id()` key plus permanent `writeable=False` freeze; the cache no longer retains the caller's arrays. `tests/models/test_ensemble_gate_does_not_freeze_caller_arrays.py`.

### MODELS_ESTIMATORS-6 [P1] unpicklable-model-param

**File:** `src/mlframe/models/masked_multilabel_objective.py` :80

**Summary:** `masked_multilabel_logloss_objective()` returns a nested local function -- exactly the "closure
attached to model params" shape: an XGBoost model built with `objective=<that closure>` cannot be pickled, so any
save routes through the dill bytecode path.

**Failure scenario:** `xgb.XGBRegressor(objective=masked_multilabel_logloss_objective())` then `joblib.dump`.
Reproduced: `PicklingError: Can't pickle local object masked_multilabel_logloss_objective.<locals>.objective`.
`training/io.py` serialises bundles with dill, so the failure is invisible at save time but the bundle carries
bytecode, which is not guaranteed to load under a different Python/xgboost -- precisely the fragility the
cluster's safe-load allowlist exists to avoid.

**Suggested fix:** Make the objective a module-level picklable callable -- a small class with
`__init__(self, sentinel, use_sample_weight)` and `__call__(self, y_pred_margin, dtrain)`, returned by the
factory. Behaviour unchanged; the instance pickles by reference. Add a test asserting `pickle.dumps` succeeds.

**Evidence:** :80 `def objective(...)` nested inside the factory; empirical pickle/joblib failures;
`training/io.py` :35, :213, :338.

**Disposition:** RESOLVED. The nested closure became a module-level `_MaskedMultilabelLogloss` with `__slots__`, explicit state hooks and `__name__`. `tests/models/test_masked_multilabel_objective_pickles.py`.

### MODELS_ESTIMATORS-7 [P1] eager-whole-frame-conversion

**File:** `src/mlframe/inference/explainability.py` :86

**Summary:** `_X = Pool(X, cat_features=...)` builds a CatBoost Pool over the ENTIRE X before the CV loop, but
`_X` is only read inside the `catboost_native_feature_importance=True` branch -- which is not the default.

**Failure scenario:** Any `compute_shap_on_cv(...)` with the default `False` (including for a LightGBM/XGBoost
`model_class`) pays a full-frame Pool construction -- a whole-frame copy plus quantisation -- whose result is
discarded. Related: the unconditional `from catboost import EFstrType, Pool` at :77 means explaining a
non-CatBoost model still hard-requires catboost.

**Suggested fix:** Move the Pool construction and the catboost import inside the branch at :178-181.

**Evidence:** `_X` appears exactly twice more, at :180 and :181, both inside the native branch.

**Disposition:** RESOLVED. The Pool and the catboost import are both inside the `catboost_native_feature_importance` branch. `tests/inference/test_explainability_pool_is_lazy.py`.

### MODELS_ESTIMATORS-8 [P1] perf-combinatorial-duplication

**File:** `src/mlframe/estimators/pipelines.py` :163

**Summary:** `optimize_pipeline_by_gridsearch` recurses on EVERY still-unassigned pipeline block at each level
instead of fixing one and recursing, so each configuration is evaluated once per permutation of assignment order
-- `k! * m^k` full CV runs instead of `m^k`.

**Failure scenario:** With k blocks of m options, the loop never breaks, so level 0 branches on block A and B and
C; each branch re-branches on the rest. Every leaf ends with a complete assignment reached via each of the k!
orderings, producing an identical `paramset_hash` and an identical `cv_func(...)` call. For a realistic 4-block
search that is 24x redundant cross-validation, and `cv_func` is the dominant cost of the function.

**Suggested fix:** Fix the first unassigned block and recurse only on it
(`var = next(v for v in possible_pipeline_blocks if v not in constants)`, or `break` at the end of the loop
body). Leaf count becomes exactly `m^k` with identical coverage.

**Evidence:** :163-170; the recursion passes `possible_pipeline_blocks=unexplored_options` but the enclosing
`for` continues to the next `var`.

**Disposition:** RESOLVED. One block is fixed per level, so the leaf count is `m^k`. An assignment where every remaining block was already pinned by a constant used to be dropped without being evaluated at all; it is now evaluated. `tests/estimators/test_pipeline_gridsearch_visits_each_config_once.py`.

### MODELS_ESTIMATORS-9 [P1] results-silently-discarded

**File:** `src/mlframe/estimators/pipelines.py` :168

**Summary:** The recursive call forwards neither `cv_results` nor `output_dir`, so each leaf builds a fresh
`cv_results = {}` and dumps to `tempfile.gettempdir()` under the same filename -- the caller's accumulator stays
empty and every leaf overwrites the previous leaf's dump.

**Failure scenario:** On return the caller's dict is still `{}` (the docstring promises "Saves results on each
cycle, summarizes by desired params"), the given `output_dir` is empty, and the only artifact is
`%TEMP%/cv_results-<title>.dump` containing exactly ONE configuration -- the last leaf visited, because the dump
path at :152 is identical for every leaf sharing a title. Combined with MODELS_ESTIMATORS-8, a multi-hour sweep
produces a single-entry result file.

**Suggested fix:** Forward both through the recursion and return `cv_results` so the accumulation contract holds.
Optionally include `paramset_hash` in the dump filename so leaves cannot clobber each other.

**Evidence:** :137-138, :151-153, :168-170.

**Disposition:** RESOLVED. `cv_results` and `output_dir` are forwarded through the recursion and `cv_results` is returned. Same test file.

### MODELS_ESTIMATORS-10 [P2] contract-drift-docstring

**File:** `src/mlframe/models/tuning_rules.py` :521

**Summary:** `justify_estimator`'s docstring states that when the CV gate passes and `refit=False`, "`est` is
returned unfitted" -- the code sets `est = None`, so the caller gets `(None, mean_score)`, which every caller
reads as "the gate rejected the estimator".

**Failure scenario:** `justify_estimator(model, X, y, refit=False, min_score=0.6)` where `mean_score = 0.8`.
Documented: an unfitted estimator plus the score. Actual: `(None, 0.8)`, indistinguishable from the
below-threshold rejection at :525, so a caller branching on `if fitted_model is None: fall back to random
sampling` silently abandons ML-guided sampling despite a passing gate. `get_model` at :429-437 caches
`[fitted_model, ...]`, so returning None for a passing gate also poisons the cache entry.

**Suggested fix:** Return `est` unfitted (matching the docstring), or update the docstring and return a third
value so "gate passed but not refit" is distinguishable from "gate rejected".

**Evidence:** docstring :449-455 versus `else: est = None` at :520-521; rejection path :523-525 returns the same
None.

### MODELS_ESTIMATORS-11 [P3] documented-knob-not-read

**File:** `src/mlframe/estimators/early_stopping.py` :318

**Summary:** `max_runtime_mins` is documented and computed into a `deadline`, but `_fit_staged` is called without
it -- the `staged` backend never checks the wall clock.

**Failure scenario:** `EarlyStoppingWrapper(..., max_iter=5000, max_runtime_mins=10)`. `_fit_partial` (:207) and
`_fit_warm` (:264) both honour the deadline; `_fit_staged` receives none and its stage loop runs the full sweep
regardless of elapsed time. The single underlying `fit` at :227 is also unbounded, so a 5000-stage budget blows a
10-minute cap with no log line.

**Suggested fix:** Pass `deadline` to `_fit_staged` and check it at the top of the stage loop, breaking with the
same `logger.info` the other two backends emit. Document that the initial full-budget fit cannot be interrupted.

**Evidence:** `fit` at :312 computes the deadline; :316 and :320 forward it, :318 does not; `_fit_staged`'s
signature at :216 has no such parameter.

### MODELS_ESTIMATORS-12 [P3] dead-parameter

**File:** `src/mlframe/inference/predict.py` :246

**Summary:** `get_models_raw_predictions(trained_models, X, Y)` never reads `Y`.

**Failure scenario:** Not a crash, but `Y` is a required positional parameter of a public, re-exported entry
point, so every caller must construct and pass a ground-truth array that is discarded. It also invites the
reading that predictions are scored or aligned against `Y`, which they are not; a caller passing a misaligned `Y`
gets no error and no effect.

**Suggested fix:** Drop the parameter (or make it `Y=None` and deprecate) and update the docstring; if an
alignment check was intended, implement `len(Y) == len(X)`.

**Evidence:** `Y` does not appear anywhere in the body (:258-274).

## Coverage

Read in full or in substantial regions: `estimators/base.py`, `estimators/baselines.py`, `estimators/custom.py`
(:40-200, :266-400, :640-730), `estimators/early_stopping.py`, `estimators/pipelines.py`, `estimators/__init__.py`,
`inference/predict.py`, `inference/explainability.py`, `inference/native_gpu_shap.py`, `inference/postanalysis.py`,
`inference/_ktc_dispatch.py`, `inference/logical_constraints.py` (:165-215), `inference/time_budget_ensemble.py`,
`inference/recursive_forecast.py`, `inference/entity_prediction_collapse.py` (:1-120), `inference/__init__.py`,
`models/additive_interaction_diagnostic.py`, `models/lgbm_defaults.py` (:1-120),
`models/masked_multilabel_objective.py`, `models/rf_proximity.py` (:40-110), `models/selection.py` (:1-60),
`models/tuning.py`, `models/tuning_catboost.py`, `models/tuning_rules.py` (:248-290, :335-540),
`models/_optimization_search.py` (several regions), `models/_optimization_shared.py` (:195-230),
`models/__init__.py`, and the `models/ensembling/` modules `base.py` (:180-260), `predict.py` (:1-335),
`per_member_tuning.py`, `selection.py` (:100-230), `score.py` (:410-470), `score_gate.py`, `quality_gate.py`
(:55-110), `process_method.py` (:340-470), `member_metrics.py` (:55-100), `__init__.py` (:115-160). Also
`metrics/_core_auc_brier.py` (:150-200, :500-600, :737-761) as the contract behind finding 1.

Cluster-wide greps: `count_nonzero`; `__getstate__`/`__setstate__`/`__reduce__` (zero occurrences);
`except Exception`/bare `except` (28 sites triaged -- narrow, logged, no estimator swapping);
`.fit(`/`fit_params`/`set_params`/`get_params`; `text_processing`/`cat_features`/`text_features`/
`embedding_features`/`monotone_constraints`/`feature_weights` (no constructor-param-in-`fit()` shape found; the
CatBoost fit sites at `estimators/base.py` :103, :118 and `tuning_rules.py` :505 pass only genuine fit kwargs);
`lambda`/`partial(`; `[:, 1]`/`argmax`/`pos_label`/`classes_`. Perf pre-check per the standing rule: `@njit`,
`prange`, `cuda.jit`, `cupy`, `kernel_tuning_cache` all confirmed present and actively covering the cluster's
numeric paths, which is why no kernel-level perf finding is asserted.

Not read (no signal from the targeted greps; the next pass): `models/ensembling/score_flavours.py`,
`score_validate.py`, `float_aggregation.py`, `models/optimization.py`, `models/ensembling/base.py` :260-990,
`models/tuning_rules.py` remainder, `inference/group_zero_sum_constraint.py`,
`estimators/early_stopping_monotonic.py`.
