# training/ top-level orchestration C (data helpers, splitting, model shims/factories, AutoML) -- mlframe audit

## Scope

All 27 files in the assigned cluster were read in full (no partial reads; every file's line count below matches `wc -l`).

| # | File | LOC |
|---|------|-----|
| 1 | src/mlframe/training/_data_helpers.py | 940 |
| 2 | src/mlframe/training/feature_drift_report.py | 929 |
| 3 | src/mlframe/training/splitting.py | 872 |
| 4 | src/mlframe/training/lgb_shim.py | 841 |
| 5 | src/mlframe/training/xgb_shim.py | 759 |
| 6 | src/mlframe/training/_composite_target_discovery_config.py | 674 |
| 7 | src/mlframe/training/_model_configs_behavior.py | 658 |
| 8 | src/mlframe/training/_reporting_configs.py | 574 |
| 9 | src/mlframe/training/_model_factories.py | 541 |
| 10 | src/mlframe/training/suite_artefact_cache.py | 536 |
| 11 | src/mlframe/training/_calibration_models.py | 473 |
| 12 | src/mlframe/training/__init__.py | 456 |
| 13 | src/mlframe/training/_partial_fit_es_wrapper.py | 423 |
| 14 | src/mlframe/training/automl.py | 416 |
| 15 | src/mlframe/training/_precompute.py | 292 |
| 16 | src/mlframe/training/_conformal_finalize.py | 284 |
| 17 | src/mlframe/training/loss_recommendation.py | 256 |
| 18 | src/mlframe/training/quantile_wrapper.py | 225 |
| 19 | src/mlframe/training/_overlapping_walk_forward_cv.py | 219 |
| 20 | src/mlframe/training/_training_loop_objectives.py | 172 |
| 21 | src/mlframe/training/_conformal_split.py | 159 |
| 22 | src/mlframe/training/_splitting_helpers.py | 146 |
| 23 | src/mlframe/training/_feature_name_sanitize.py | 120 |
| 24 | src/mlframe/training/_easy_ensemble.py | 102 |
| 25 | src/mlframe/training/_aggregate_cv_early_stopping.py | 82 |
| 26 | src/mlframe/training/_uncertainty_eval.py | 74 |
| 27 | src/mlframe/training/_model_configs_ensembling.py | 54 |

**Total: 27 files reviewed, 11277 LOC reviewed.**

For context only (not analyzed for findings, per audit exclusion rules): briefly opened `src/mlframe/training/_gpu_probe.py` (not in this cluster's 27-file list) to confirm the canonical `CUDA_IS_AVAILABLE` probe pattern that `_model_factories.py` duplicates without the same guards -- this file is cited in findings below but was not itself audited for its own bugs.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | perf/architecture | lgb_shim.py:283-303,433-501 | LGB Dataset-reuse shim has no module-level cache fallback (unlike xgb_shim), so `sklearn.clone()` of the LGB shim silently loses all Dataset-reuse benefit |
| F2 | P1 | perf/correctness | lgb_shim.py:502-582 | Single instance-level val-Dataset cache slot gets overwritten on every eval-set iteration within one `fit()` call, defeating caching whenever `eval_set` has more than one pair (slice-stable ES / extra_eval_sets) |
| F3 | P1 | robustness/crash | _model_factories.py:505-507 | Unguarded `numba.cuda.is_available()` probe at import time; crashes package import on hosts without numba / a working CUDA driver, a failure mode the sibling `_gpu_probe.py` explicitly wraps in try/except for the identical call |
| F4 | P1 | correctness | _partial_fit_es_wrapper.py:196-207 | `PartialFitESWrapper.get_params()` omits `external_X_val`/`external_y_val`; `sklearn.clone()` silently drops the caller-supplied external val set and falls back to an internal random split |
| F5 | P1 | correctness | automl.py:245-266 | `train_lama_model` always builds a fake binary `[1-p, p]` probability pair from `test_predictions.data[:, 0]` regardless of the actual LAMA `Task` type, silently producing a bogus class-0-vs-rest "test_auc" for multiclass tasks instead of erroring |
| F6 | P1 | correctness | _overlapping_walk_forward_cv.py:204-209 | `cv_stability_check` hardcodes `np.argmax` (higher-is-better) with no direction parameter or documented convention; a loss-type metric curve (RMSE/log-loss) silently computes "stability"/"agreement" against the WORST hyperparameter |
| F7 | P1 | correctness | _conformal_split.py:21-46 | `carve_calib_conformal_iid`/`_temporal` don't guard a small non-zero `calib_frac`/`conformal_frac` flooring to 0 rows, unlike `carve_calib_conformal_grouped` which explicitly raises on the identical failure mode |
| F8 | P2 | consistency | _data_helpers.py:200-217 | `_setup_sample_weight` doesn't normalize boolean `train_idx` masks the way the sibling `_extract_target_subset` does (fixed there for exactly this reason), and has no polars-Series branch |
| F9 | P2 | robustness | _data_helpers.py:686-699 | `_groupids_to_sizes` silently assumes `group_ids` is pre-sorted by qid, with no validation; an out-of-order qid vector silently corrupts LGB `eval_group` sizes |
| F10 | P2 | correctness/edge-case | _data_helpers.py:422-425 | `if model_type_name not in model_name` is a raw substring check; a coincidental substring match skips prefixing and can collide `model_file_name`s across different model types |
| F11 | P2 | docs/UX | _model_configs_behavior.py:614-619,649-652 | `point_estimate_alpha`'s docstring claims "validator enforces membership" but the code silently snaps to the nearest alpha with no warning/log |
| F12 | P2 | logging accuracy | _splitting_helpers.py:67-97,105-144 | When a shuffled val/test count is clamped for pool exhaustion, `_build_details` still reports the ORIGINAL (pre-clamp) `n_shuffled` count in the summary line |
| F13 | P2 | correctness/edge-case | splitting.py:796-813 | The post-split empty-split guard checks `val_idx`/`test_idx` for zero rows but has no analogous check for an empty `train_idx` |
| F14 | P2 | dead code/docs | suite_artefact_cache.py:1-28 | Entire ~537-LOC module is never imported anywhere in the repo; its docstring's claimed "Wire-in proofs-of-concept" (`core._phase_helpers_fit_pipeline`) does not actually reference this module |
| F15 | P2 | stale comment | _precompute.py:284-286 | Comment claims the two precompute stubs "currently return empty dicts"; they actually always `raise NotImplementedError`, and `precompute_all` never calls them at all |
| F16 | P2 | consistency/dead code | _feature_name_sanitize.py:112-120 | `sanitize_name_list` uses a plain non-collision-aware map, unlike `build_safe_mapping`'s dedup-with-suffix logic used for the DataFrame's own columns; currently unused anywhere in the repo |
| F17 | P2 | architecture | __init__.py:407-429 | Bottom-of-file eager imports of a batch of submodules contradict the module's own stated "lazy import, cheap bare import" design used for everything else in the file |
| F18 | P2 | architecture | _model_factories.py:505-507 | `CUDA_IS_AVAILABLE` is computed via a second, independent probe instead of importing the canonical `_gpu_probe.CUDA_IS_AVAILABLE` the rest of the package uses |

### F1 -- lgb_shim.py has no module-level Dataset cache (unlike xgb_shim)

`xgb_shim.py` explicitly maintains a module-level `_XGB_DMATRIX_CACHE` (an `OrderedDict` with LRU eviction) specifically because "the CompositeCrossTargetEnsemble OOF refit helper calls `clone(inner)` per component, each clone has an empty instance cache, so QuantileDMatrix was rebuilt 4 times (20+s wasted per ensemble round)" (xgb_shim.py:174-179). `lgb_shim.py`'s `_DatasetReuseMixin` only has instance-level `_cached_train_dataset`/`_cached_train_key` (lgb_shim.py:271-303) with no equivalent module-level fallback. Any code path that clones an `LGBMClassifierWithDatasetReuse`/`LGBMRegressorWithDatasetReuse` (the same composite-ensemble OOF refit pattern documented as the motivating case for the XGB fix) will silently get a fresh, empty cache on every clone and rebuild the full binned `lightgbm.Dataset` from scratch every time -- exactly the cost this shim exists to eliminate, just for LGBM instead of XGBoost. Suggested fix: add a module-level LRU `Dataset` cache to `lgb_shim.py` mirroring `_XGB_DMATRIX_CACHE`/`_xgb_cache_get`/`_xgb_cache_put`.

### F2 -- lgb_shim.py single-slot val cache defeated by multi-eval-set fits

Inside `fit()`'s per-eval-set loop (lgb_shim.py:516-566), each iteration checks `self._cached_val_key == val_key` against the single instance attributes `_cached_val_dataset`/`_cached_val_key`. When `eval_set` has more than one pair (e.g. the slice-stable-ES `extra_eval_sets` path documented in `_data_helpers._setup_eval_set`, which builds `[full_val, shard_0, ..., shard_K-1]`), the FIRST iteration's built/reused Dataset gets stored into the single cache slot, then the SECOND iteration's different `X_val` always misses (its key never matches slot i=0's key) and OVERWRITES the slot with its own Dataset. Every subsequent `fit()` call thus only ever gets a cache hit for whichever eval-set pair happened to be processed LAST in the previous call -- the other N-1 val Datasets are rebuilt from scratch on every single fit. This silently defeats the entire point of the shim for any caller using multiple eval sets with LGB. Suggested fix: key the val cache by `(index_in_eval_set, val_key)` or store a dict of slots, not a single pair of instance attributes.

### F3 -- unguarded numba.cuda probe crashes package import on numba/CUDA-less hosts

```python
from numba.cuda import is_available as is_cuda_available   # line 505
...
CUDA_IS_AVAILABLE = is_cuda_available()                      # line 507
```
This exact call sequence is also present in `_gpu_probe.py`, but there it is deliberately wrapped: *"numba is an optional dep: probing CUDA via numba.cuda is convenient but the training package must still import on machines without numba (or without a working CUDA driver). Wrap both the import and the call so any failure - ImportError, OSError on missing libcuda, runtime probe errors - degrades silently to CPU-only mode."* `_model_factories.py` performs the identical probe completely unguarded at module scope. On a host that lacks numba, or has numba but a broken/absent CUDA driver (the file's own torch-import block a few lines above explicitly handles this exact class of failure for torch: `except (ImportError, OSError)` with a comment about "Windows DLL load failures... shadow CUDA installs"), importing `mlframe.training._model_factories` -- and therefore anything that imports it, including `mlframe.training` itself via `_LAZY_IMPORTS` resolving `_patch_lgb_feature_names_in_setter` -- raises uncaught. This value is also consumed live at `_confidence_analysis.py:103`. Suggested fix: wrap the import + call exactly like `_gpu_probe.py` does, or better, import `CUDA_IS_AVAILABLE` directly from `_gpu_probe` (see F18).

### F4 -- PartialFitESWrapper.get_params() drops external_X_val/external_y_val

The constructor accepts `external_X_val`/`external_y_val` (lines 168-169) and `fit()` prefers them over an internal split whenever explicit `X_val=`/`y_val=` kwargs are absent (lines 233-238) -- this is the documented mechanism for suite callers to "plug into `model.fit(X, y, **fit_params)`-style calling conventions where the caller can't easily inject X_val/y_val into the fit signature." But `get_params()` (lines 196-207) returns only `estimator, metric, patience, min_delta, val_size, random_state, max_iter, is_classification, budget_param, budget_min, budget_max, verbose` -- `external_X_val`/`external_y_val` are missing. Per the sklearn estimator protocol, `sklearn.base.clone()` calls `get_params(deep=False)` then reconstructs via `type(self)(**params)`; any clone of a `PartialFitESWrapper` therefore always gets `external_X_val=None, external_y_val=None` regardless of what was originally passed, silently falling back to an internal `val_size`-based random split instead of the caller's intended held-out val. This changes which rows drive early stopping without any error or warning. Suggested fix: add both fields to the returned dict.

### F5 -- train_lama_model silently mis-scores non-binary LAMA tasks

`train_lama_model`'s default (`init_params=None`) uses `Task("binary")`, but the docstring explicitly notes "For regression or multiclass tasks, pass an appropriate Task object in init_params" (automl.py:206-208) -- implying the function supports other task types via caller-supplied `init_params`. However the post-fit probability block (lines 245-266) unconditionally does:
```python
pred_col = test_predictions.data[:, 0] if test_predictions.data.ndim > 1 else test_predictions.data
test_probs = np.vstack([1 - pred_col, pred_col]).T
```
For a genuine multiclass `Task`, `test_predictions.data` has K>2 columns; this code silently keeps only column 0 and discards the rest, builds a fabricated 2-column "probability" pair, and then computes and reports `metrics["test_auc"]` against the (multiclass) `test_target` as if it were a real evaluation -- with no shape/task-type check and no warning that only class-0-vs-rest was scored. Contrast with the sibling `train_autogluon_model`, which correctly branches on `test_probs.shape[1] > 2` to route multiclass through `multi_class="ovr"`. Suggested fix: mirror the AutoGluon branch, or detect non-binary output shape and skip/warn instead of silently faking a binary score.

### F6 -- cv_stability_check has no metric-direction parameter

`cv_stability_check` (docstring: "Flag a hyperparameter-vs-metric curve as noisy/untrustworthy before acting on it... `stable` (bool: ... safe to act on the mean curve's optimum)") always does `mean_argmax = int(np.argmax(mean_curve))` (line 204) and compares each seed's own `np.argmax` against it. This is only correct for a higher-is-better metric (AUC, accuracy, correlation). If a caller feeds a loss-type curve (RMSE, log-loss -- exactly the kind of metric `PartialFitESWrapper`/`select_best_iteration_by_aggregate_cv` elsewhere in this SAME cluster explicitly support via a `mode`/`maximize` parameter), `cross_seed_argmax_agreement`/`stable` silently measure agreement on the WORST hyperparameter, not the best, and a caller acting on "stable=True, use `mean_curve.argmax()`" would pick the single worst config with high cross-seed confidence. No `maximize` parameter exists, no docstring caveat exists, and the module's own test file (`tests/training/test_biz_val_overlapping_walk_forward_cv.py`) only exercises higher-is-better-style synthetic curves. Suggested fix: add a `maximize: bool = True` parameter (mirroring `select_best_iteration_by_aggregate_cv` in the same cluster) and route `argmax`/`argmin` through it.

### F7 -- iid/temporal conformal carvers lack the grouped carver's zero-floor guard

`carve_calib_conformal_grouped` explicitly checks (lines 109-112): *"A non-zero requested fraction that floors to 0 groups silently produces an empty calib/conformal slice (too few groups for the fraction)"* and raises `ValueError`. `carve_calib_conformal_iid` and `carve_calib_conformal_temporal` share the exact same `_resolve_counts` floor-to-zero mechanism (`int(np.floor(calib_frac * n))`) but have no equivalent per-slice check -- only `carve_calib_conformal_iid` checks that `n_calib + n_conf < n` (both slices together don't consume the whole set), and `carve_calib_conformal_temporal` only checks that the FIT slice isn't empty. A small `calib_frac` (or `conformal_frac`) on a small `n` therefore silently produces a genuinely EMPTY calib or conformal array in the iid/temporal paths -- the exact silent-zero-floor failure mode the grouped carver was explicitly hardened against, left unfixed in its two siblings. Suggested fix: apply the same explicit-raise guard used in the grouped carver to the iid and temporal carvers.

### F8 -- _setup_sample_weight boolean-mask / polars inconsistency

`_extract_target_subset` (lines 89-135) was explicitly fixed to normalize a boolean `idx` mask to integer positions before indexing, with the comment explaining the pre-fix divergence between the pandas and polars branches ("polars `target.gather(mask)` rejects boolean and raises `InvalidOperationError`"). `_setup_sample_weight` (lines 200-217), which does the conceptually identical "subset an array-like by `train_idx`" operation just a few dozen lines away in the same file, has no such normalization and no polars-Series branch at all -- a `pl.Series` `sample_weight` falls into the generic `sample_weight[train_idx]` path, whose boolean-mask behaviour for polars was exactly what the sibling function had to special-case. Suggested fix: route both helpers through one shared subsetting primitive so a future fix to one automatically covers the other.

### F9 -- _groupids_to_sizes trusts caller-supplied qid ordering

`_groupids_to_sizes` documents "Rows must be already sorted by qid -- the standard ranker contract" but performs no check. If a caller violates that (e.g. an upstream shard builder that doesn't re-sort after a filter), the run-length encoding silently produces wrong per-query group sizes, which LightGBM would then silently misinterpret as different query boundaries -- a silently-wrong eval, not a raised error. Suggested fix: a cheap `np.all(np.diff(arr) >= 0)`-style assertion (or WARN) would catch this class of bug close to its source.

### F10 -- substring containment check for "already has model type in name"

`if model_type_name not in model_name: model_name = model_type_name + " " + model_name` (_data_helpers.py:424-425) uses Python's `in` operator for substring containment rather than an exact-token check. A `model_name` that happens to already contain `model_type_name` as a substring of an unrelated word (e.g. `model_type_name="CB"` and a user-supplied `model_name_prefix` containing "CBOW-embeddings") will silently skip the intended prefixing. Since `model_file_name` is derived from `model_name`, two different model types could coincidentally end up building the same `.dump` filename and overwrite each other. Suggested fix: check for `model_type_name` as a whitespace-delimited token, not a raw substring.

### F11 -- point_estimate_alpha docstring vs. actual silent-snap behaviour

`QuantileRegressionConfig.point_estimate_alpha`'s docstring says: "Mean-of-alphas is the alternative if user picks an alpha not in the set; validator enforces membership" (_model_configs_behavior.py:614-619). The actual validator (lines 649-652) does neither -- it silently rewrites the field to the nearest alpha in the set via `object.__setattr__`, with no log line and no exception. A caller who sets `point_estimate_alpha=0.4` on `alphas=(0.1, 0.5, 0.9)` gets `0.5` silently substituted with zero indication anything changed. Suggested fix: either implement what the docstring promises (raise, or genuinely average), or update the docstring and add a debug/warning log on snap.

### F12 -- clamped shuffled-count not reflected in the reported detail string

`_perform_split` clamps `n_test_shuf`/`n_val_shuf` down to the available pool with a WARN log when the caller's requested count exceeds what's left (_splitting_helpers.py:67-97), but the CALLER (`splitting.py:430-431`) still passes the un-clamped, original `n_val_shuf`/`n_test_shuf` into `_build_details`, whose `"+{n_shuffled}{unit}"` suffix therefore reports the pre-clamp count even though fewer rows were actually mixed in. This is a log/summary-line accuracy bug, not a data-correctness bug (the actual split rows are correct), but it can mislead an operator reading the "N train rows / N val rows ... +45000R" summary about how much of the val/test set is randomly-sampled vs. sequential.

### F13 -- no empty-train guard symmetric to the empty-val/test guards

`splitting.py:796-813` raises an actionable `ValueError` when `val_size > 0` but `val_idx` came out empty, and again for `test_idx` -- explicitly because a silent 0-row split "surfaces far from the cause as e.g. CatBoost 'Input data must have at least one feature'". The same floor-to-zero / wholeday-collapse mechanisms that can empty `val_idx`/`test_idx` can, on a very small `n` combined with a large `trainset_aging_limit`, also empty `train_idx` -- but there is no equivalent guard for that case, so the same class of confusing downstream crash the val/test guards were added to prevent remains reachable via the train side.

### F14 -- suite_artefact_cache.py is entirely unwired dead code

The module docstring lists concrete wire-in sites under "Wire-in proofs-of-concept (this commit)": `mlframe.training.core._phase_helpers_fit_pipeline._cached_fit_and_transform_pipeline` and `._cached_apply_preprocessing_extensions`. `_phase_helpers_fit_pipeline.py` exists but contains no reference whatsoever to `cache_artefact`, `SuiteArtefactCache`, or `get_default_cache`, and a repo-wide grep finds zero importers of `suite_artefact_cache` anywhere in `src/mlframe/`. The entire ~537-LOC module (key builder, LRU disk cache, decorator) is unreachable production code with a docstring that overstates its integration status.

### F15 -- stale comment in _precompute.py

`precompute_all` (_precompute.py:284-286) comments: "The two helpers below currently return empty dicts; preserve the None sentinel..." -- but `precompute_dummy_baselines`/`precompute_composite_target_specs` are defined a few lines above to always `raise NotImplementedError` (never return anything, empty or otherwise), and `precompute_all` doesn't even call them. The comment describes behaviour the code has never had in this file.

### F16 -- sanitize_name_list doesn't share build_safe_mapping's collision-avoidance

`build_safe_mapping` (_feature_name_sanitize.py:53-75) explicitly dedupes a collision where two different hostile names translate to the same safe base string, appending `_1`, `_2`, etc. `sanitize_name_list` (lines 112-120), intended to remap an auxiliary name list (e.g. `cat_features`) "through the same pure map," instead calls `safe_feature_name(n)` directly per entry with no collision tracking at all. If a caller renamed a DataFrame's columns via `sanitize_frame_columns` (collision-safe) and separately renamed an associated name list via `sanitize_name_list` (collision-unsafe), a genuine collision would leave the name list referencing a name that either doesn't exist on the renamed frame or points at the WRONG column. Currently this function has zero call sites anywhere in the repository, so the bug is latent rather than live.

### F17 -- eager imports at the bottom of a file designed around lazy imports

The top of `__init__.py` invests significant effort explaining why heavy imports are deferred via `_LAZY_IMPORTS`/`__getattr__` ("Bare `import mlframe.training` no longer mutates lightgbm internals," wmic-subprocess-cost rationale, etc.), and most of the public surface is indeed lazy. But lines 407-429 unconditionally `import` a batch of submodules (`_conformal_finalize`, `_regression_calibration`, `_tta`, `_uncertainty_eval`, `_mc_dropout`, `_noise_ensemble`, `_pseudo_group_reconstruction`, `_overlapping_walk_forward_cv`, `_easy_ensemble`, `_direct_horizon_bucket_forecaster`) at module scope, alongside `PartialFitESWrapper` at line 280. If any of these transitively pull in something expensive, `import mlframe.training` pays that cost every time regardless of whether the caller ever touches those symbols -- contradicting the file's own stated design goal.

### F18 -- duplicate CUDA_IS_AVAILABLE probe

`_model_factories.py` computes its own `CUDA_IS_AVAILABLE` (line 507) rather than importing the canonical one from `_gpu_probe.py`, which `helpers.py`, `_helpers_training_configs.py`, and `__init__.py` (via `_LAZY_IMPORTS`) all use instead. Two independent probes of the same underlying fact is an unnecessary maintenance/consistency risk on top of being the unguarded copy described in F3.

## Proposals

| ID | Category | File | Summary |
|----|----------|------|---------|
| PR1 | test-coverage | _overlapping_walk_forward_cv.py | Add a `test_biz_val_*` case for `cv_stability_check` on a loss-type (lower-is-better) curve once F6 is fixed, so the direction convention is pinned by a test, not just a docstring |
| PR2 | test-coverage | lgb_shim.py / xgb_shim.py | Add a parity test asserting LGB and XGB shims both survive `sklearn.clone()` with a cache hit (currently only implied by the xgb_shim module docstring's prod anecdote; nothing pins the LGB side, which is why F1 went unnoticed) |
| PR3 | test-coverage | _partial_fit_es_wrapper.py | Add a `test_sklearn_clone_preserves_external_val` regression test for F4 -- `sklearn.clone(wrapper)` then compare `get_params()` round-trip against the original |
| PR4 | refactor | suite_artefact_cache.py | Either finish the documented wire-in (F14) or delete the module -- as committed dead code it currently costs review/maintenance attention for zero runtime benefit |
| PR5 | refactor | _conformal_split.py | Factor `_resolve_counts` + a shared "raise on non-zero-fraction-floors-to-zero" check into one helper all three carvers call, so F7's fix can't silently regress again in a future edit to just one carver |
| PR6 | perf | lgb_shim.py | Once F1/F2 land, bench the composite-ensemble OOF refit path with LGB models the same way the xgb_shim module docstring benchmarks XGB ("20+s wasted per ensemble round") to quantify the actual win |
| PR7 | docs | _model_configs_behavior.py | Fix the `point_estimate_alpha` docstring (F11) to describe the real snap-to-nearest behaviour, or implement the documented raise/mean-of-alphas alternative |

## Coverage notes

- All 27 assigned files were read in full; no file was too large to review in depth (largest was 940 LOC).
- `_split_helpers.py` (imported by `splitting.py` for `_carve_calib_from_train`, `_deleak_tied_boundaries`, `_stratified_split`, `_stratified_split_3way`, `_use_multilabel_3way`) and `_gpu_probe.py` (referenced for F3/F18 context) are NOT in this cluster's 27-file list and were only opened where necessary to confirm a cross-file finding is real (not to hunt for their own bugs); neither is claimed as fully audited here.
- Did not execute any test or benchmark (per the read-only mandate); all findings above are static-analysis conclusions. Where I traced a candidate issue (e.g. the `_perform_split`/`_build_details` bisection-style dichotomic search algorithm in `_partial_fit_es_wrapper.py`) and, after manually tracing several numeric examples, could not construct a case where the algorithm converges to a genuinely wrong answer, I did NOT report it as a finding (per the "no hand-wave hypotheses" rule) -- it remains an unusual but apparently-correct bisection variant worth a closer look only with an actual discriminating test, not flagged here.
- Did not attempt to verify at runtime whether `pl.Series` bracket-indexing with a boolean array in `_setup_sample_weight` (F8) actually raises or silently misbehaves on the installed polars version; the finding is reported as an inconsistency with the sibling function's explicit fix rather than a confirmed crash, per the objectivity convention.
