# training/ top-level orchestration B (trainer core, feature importances, drift, MLP defaults) -- mlframe audit

## Scope

All 27 files were read in full (no file skipped or partially reviewed):

- src/mlframe/training/trainer.py (975 LOC)
- src/mlframe/training/_training_loop.py (883 LOC)
- src/mlframe/training/_helpers_training_configs.py (880 LOC)
- src/mlframe/training/utils.py (865 LOC)
- src/mlframe/training/_feature_importances.py (741 LOC)
- src/mlframe/training/_eval_helpers.py (672 LOC)
- src/mlframe/training/metrics_registry.py (646 LOC)
- src/mlframe/training/_predict_guards.py (583 LOC)
- src/mlframe/training/_training_runtime_configs.py (549 LOC)
- src/mlframe/training/mlp_runtime_defaults.py (498 LOC)
- src/mlframe/training/models.py (480 LOC)
- src/mlframe/training/honest_diagnostics.py (464 LOC)
- src/mlframe/training/_confidence_analysis.py (454 LOC)
- src/mlframe/training/drift_report.py (383 LOC)
- src/mlframe/training/_training_loop_refit.py (357 LOC)
- src/mlframe/training/_split_helpers.py (283 LOC)
- src/mlframe/training/_cv_aggregation.py (251 LOC)
- src/mlframe/training/phases.py (222 LOC)
- src/mlframe/training/logging_transformers.py (217 LOC)
- src/mlframe/training/_calib_oof_outputs.py (160 LOC)
- src/mlframe/training/_direct_horizon_bucket_forecaster.py (150 LOC)
- src/mlframe/training/_tta.py (133 LOC)
- src/mlframe/training/provenance.py (117 LOC)
- src/mlframe/training/_pseudo_group_reconstruction.py (103 LOC)
- src/mlframe/training/_gpu_probe.py (83 LOC)
- src/mlframe/training/grid.py (77 LOC)
- src/mlframe/training/_trainer_train_and_evaluate_helpers.py (47 LOC)

Total: **27 files, 11273 LOC reviewed** (sum of `wc -l` over the list above).

I did not open any file outside this list (e.g. `_trainer_train_and_evaluate.py`, `_classif_helpers.py`, `cb/*`, `core/*`) except to `grep` for call sites/consumers of functions defined in my scope, to check whether a suspicious internal contract (e.g. missing `sample_weight` parameter) is actually exercised. Those greps did not constitute analysis of the excluded files' internals.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | memory-discipline | _training_loop.py:401,422,436,451 | `_train_model_with_fallback`'s CatBoost NaN-cat/text fill path uses bare `.copy()` (deep=True) on train/eval frames instead of `.copy(deep=False)`, contradicting the project's own memory-safety rule and the identical fix already applied in sibling files. |
| F2 | P1 | ml-best-practice | trainer.py:174-274 | `_compute_oof_preds` (level-1 stacking OOF) clones the model and calls `cross_val_predict` without ever threading `sample_weight`, so OOF predictions for a weighted fit come from an effectively unweighted refit. |
| F3 | P1 | correctness / state-consistency | _training_loop_refit.py:347-357 | After the collapse-recovery ladder is exhausted, `_inner.network_params` is reset to the pre-collapse (bad) config while the live `_inner.network` remains fit under the LAST (different) candidate architecture -- a metadata/live-state divergence that a later `sklearn.clone()` (e.g. trainer.py's own OOF path, F2) would silently reproduce. |
| F4 | P2 | robustness | _training_loop.py:307,655,761 | `isinstance(train_df, pl.DataFrame)` / `isinstance(X_val, pl.DataFrame)` used unguarded 3x despite the module's own `try/except ImportError: pl = None` header; would raise `AttributeError` (not a clean message) if polars ever fails to import, inconsistent with the guarded pattern used correctly elsewhere in the same package. |
| F5 | P2 | contract-mismatch | utils.py:435-489,681-701 | `get_pandas_view_of_polars_df`'s own `TypeError` guard + docstring say it accepts `pl.Series`, but the body (`df.schema`, `df.to_arrow().columns`) only works for `pl.DataFrame`; a `pl.Series` argument would crash with an unrelated `AttributeError` instead of the advertised behaviour. |
| F6 | P2 | doc-accuracy | _gpu_probe.py:72-83 | Dangling comment block at end of file documents a "canonicalizer + decision-rule pair below" that isn't in this file (the functions live in `_classif_helpers.py`); leftover header from a monolith split, misleading to a reader of this module. |
| F7 | P2 | consistency | _calib_oof_outputs.py:24-62 | `maybe_run_confidence_analysis` forwards `fit_params` to the confidence-analysis helper only `if model_type_name == "CatBoostRegressor"`, silently excluding `CatBoostClassifier` and every other backend from the same context-forwarding with no documented rationale. |
| F8 | P2 | ml-best-practice | _feature_importances.py:130-189; _confidence_analysis.py:32-105 | Permutation-importance scoring (`_adaptive_scorer`) and the confidence regressor's `.fit()` never thread `sample_weight`; FI ranking / confidence diagnostics silently revert to an unweighted objective for a weighted-fit model. |
| F9 | P2 | edge-case | drift_report.py:271-290 | Multiclass branch's `np.concatenate([a for a in (train, val, test) if a is not None and a.size > 0])` raises a bare, unhelpful `ValueError` when `train_target` is a non-None empty array and `val`/`test` are `None`, instead of the graceful "no drift report" path used for `train is None`. |
| F10 | P2 | code-quality | honest_diagnostics.py:183-225 | On the mlframe-metrics `ImportError` path, `out["status"] = "skipped"` is set unconditionally, but execution still falls through to the independent ECE try-block and the per-metric bootstrap loop; if those succeed, `out` ends up with both a whole-block `"status": "skipped"` key and legitimate per-metric (`"ece"`) entries -- a self-contradictory shape for downstream consumers. |

### F1 -- unnecessary deep-copy of production-scale frames in the CatBoost NaN-fill path

`_train_model_with_fallback` (the hot path executed on every model `.fit()` call) has two blocks that defensively fill NaN in categorical/text columns before handing the frame to CatBoost (`_training_loop.py:384-457`). Both blocks -- and their mirror for `eval_set` -- use bare `train_df.copy()` / `_eval_df_filled.copy()` (pandas default `deep=True`) to avoid mutating the caller's frame before rewriting 1-2 columns. The file's own comments elsewhere (`utils.py`, `_eval_helpers.py:139,250`, `_confidence_analysis.py:270`) explicitly call out this exact pattern and use `.copy(deep=False)` with a comment citing "100+ GB frame OOM risk" -- CLAUDE.md's own memory-discipline rule ("never `.copy()`/`.clone()`/reconstruct a frame ... mutate-and-restore or use views"). Here the fix was missed: any CatBoost fit whose cat_features carry NaN or whose text/embedding routing left a stray `category` dtype (the file's own comment says this is a fuzz-confirmed, non-rare case, "fuzz c0062") pays a full deep-copy of the entire train (and val) frame -- on a 100+ GB production frame this is the exact OOM/slowdown pattern the codebase elsewhere fixed. Suggested fix: `deep=False` at all four sites, matching the sibling-file pattern exactly.

### F2 -- OOF stacking predictions silently drop sample_weight

`_compute_oof_preds` (trainer.py:174) computes K-fold out-of-fold predictions for level-1 ensemble stacking by `clone(model)` + `sklearn.model_selection.cross_val_predict(estimator, train_df, train_target, cv=splitter, groups=_groups_arg, method=method, n_jobs=1)`. There is no `sample_weight` parameter anywhere on this function, and none is threaded into `cross_val_predict`'s `fit_params`. When the suite trains with `sample_weight` (a first-class, documented feature of `DataConfig`), the OOF sub-models used for stacking are fit WITHOUT the weights the "real" model used, so the meta-learner sees OOF predictions from a systematically different (unweighted) model than the one actually deployed as an ensemble member. This is silent: no warning, no error, just a quietly different training objective for the OOF pass. Suggested fix: accept `sample_weight` and thread `fit_params={"sample_weight": sample_weight[train_index]}`-equivalent per-fold slicing into `cross_val_predict` (and the manual `_compute_oof_preds_timeseries` loop).

### F3 -- collapse-recovery ladder leaves network_params/live-network state inconsistent

`_maybe_refit_on_collapsed_predictions` (_training_loop_refit.py:174) tries a ladder of architecture patches (enable batchnorm -> shrink to 1 layer -> bump dropout) when an MLP's predictions collapse to near-constant. If ALL rungs fail to recover a healthy fit, the function restores `_inner.network_params = _orig_snapshot` (the ORIGINAL, collapse-triggering config) at line 350 -- but the actual `_inner.network` (torch weights) at that point were fit under the LAST candidate's patched params (`dropout_prob=0.15, nlayers=1, use_batchnorm=True`), not the original. The live predict path still works (it reads `.network` directly), but `network_params` -- the dict that `sklearn.clone()`/`get_params()` would read to rebuild a fresh, UNFITTED instance -- now describes an architecture that is known, by this very code path, to collapse. Any caller that clones this model post-fit (trainer.py's own `_compute_oof_preds`, F2, is exactly such a caller for stacking) would silently reproduce the original collapse in the cloned/refit sub-models. This is a derived risk chain from reading both code paths, not an empirically-reproduced failure in this session. Suggested fix: persist the LAST-attempted (not original) `network_params` snapshot on exhaustion, or stamp a `_mlframe_known_collapsed=True` marker that `clone()`-based OOF/stacking callers can check and skip.

### F4 -- unguarded `isinstance(x, pl.DataFrame)` despite optional-polars header

`_training_loop.py` opens with `try: import polars as pl / except ImportError: pl = None` (lines 16-19), signalling polars is meant to be optional for this module. But `_train_model_with_fallback` then does `isinstance(train_df, pl.DataFrame)` (line 307), `isinstance(train_df, pl.DataFrame)` (line 655) and `isinstance(X_val, pl.DataFrame)` (line 761) with no `pl is not None` guard. If polars ever fails to import (a real risk class even though it is currently a hard pyproject dependency -- e.g. a bad wheel or ABI mismatch), every model fit through this function crashes with `AttributeError: 'NoneType' object has no attribute 'DataFrame'` instead of a clean, diagnosable error. `utils.py`'s `drop_columns_from_dataframe` and `_predict_guards.py`'s `_pl_DataFrame()` helper both guard this correctly in the same package, showing the intended pattern.

### F5 -- get_pandas_view_of_polars_df's pl.Series contract isn't actually implemented

The function's own input guard (`utils.py:476-477`) is `if pl is None or not isinstance(df, (pl.DataFrame, pl.Series)): raise TypeError("Input must be a Polars DataFrame or Series, ...")`, explicitly advertising `pl.Series` as a valid input. But every code path past that guard assumes a DataFrame: `df.schema`, `df.with_columns(...)`, and unconditionally `tbl = df.to_arrow()` followed by `tbl.columns` (a `pa.Table` API that a Series-derived `pa.Array`/`pa.ChunkedArray` does not have). Calling with a genuine `pl.Series` would crash with `AttributeError: 'pyarrow.lib.ChunkedArray' object has no attribute 'columns'` rather than behaving as documented. Currently dead code (`grep` across `src/` shows every one of the ~25 call sites passes a DataFrame), so no live bug today, but the discrepancy is a trap for the next caller who takes the docstring/type-check at face value. Suggested fix: either implement the Series path (wrap in a 1-column frame) or narrow the guard + docstring to DataFrame-only.

### F6 -- orphaned doc comment in _gpu_probe.py

Lines 72-83 are a substantial comment block ("Multi-output (multiclass + multilabel) dispatch helpers -- 2026-04-24 ... The canonicalizer + decision-rule pair below wraps that heterogeneity ...") with no code following it -- the file ends at line 83. `grep` confirms the referenced functions (`_canonical_predict_proba_shape` and its sibling) actually live in `_classif_helpers.py`. This is a leftover section header from a module split that was never cleaned up; it actively misleads anyone reading `_gpu_probe.py` looking for the described helpers. Low-cost fix: delete the block or move it to `_classif_helpers.py`.

### F7 -- confidence-analysis fit_params forwarded only for CatBoostRegressor

`maybe_run_confidence_analysis` (_calib_oof_outputs.py:24) passes `fit_params=fit_params if model_type_name == "CatBoostRegressor" else None` into `run_confidence_analysis`. The confidence model built inside `run_confidence_analysis` is ALWAYS a fresh `CatBoostRegressor` regardless of what `model_type_name` was (trainer.py never builds it from the caller's actual estimator), so there is no type-compatibility reason to special-case exactly `"CatBoostRegressor"` and exclude `"CatBoostClassifier"` (or any other backend) from forwarding CB-native fit kwargs (e.g. custom text_processing settings the caller already tuned). The asymmetry has no comment explaining why only the regressor variant qualifies; it reads like an incomplete generalization rather than an intentional gate.

### F8 -- permutation FI and confidence-regressor scoring drop sample_weight

`_permutation_feature_importances`/`_adaptive_scorer` (_feature_importances.py) and `run_confidence_analysis`'s `confidence_model.fit(test_df, confidence_targets, **fit_params_copy)` (_confidence_analysis.py) never accept or forward `sample_weight`. Both are read-only diagnostics (not the production training path), so the blast radius is narrower than F2, but the effect is the same class of bug: a model trained with `sample_weight` gets its post-hoc feature-importance ranking / confidence chart computed against an UNWEIGHTED r2/accuracy/CB-fit objective, which can rank features differently than the weighted objective the model was actually optimized for.

### F9 -- multiclass drift report crashes ungracefully on an empty (non-None) train_target

`compute_label_distribution_drift`'s multiclass branch (drift_report.py:271-290) does `all_arr = np.concatenate([a for a in (train, val, test) if a is not None and a.size > 0])`. The only upstream guard is `if train is None: return {...}` (line 194) -- it does not check `train.size == 0`. If `train_target` is a legitimate-but-empty array (0-row train split, non-None) and `val_target`/`test_target` are `None`, the list comprehension yields `[]` and `np.concatenate([])` raises `ValueError: need at least one array to concatenate` with no context tying it back to drift reporting. This is consistent with the codebase's general philosophy of failing loud rather than silently swallowing 0-row splits (per `_eval_helpers.py`'s own comment on the same topic), but the specific error message here gives the caller no clue it originated in the drift report.

### F10 -- honest_diagnostics bootstrap block: status flag can be contradicted by later success

`_bootstrap_block` (honest_diagnostics.py:81) sets `out["status"] = "skipped"` unconditionally inside the `except ImportError` handler for the `mlframe.metrics.core` import (line 183-185), but execution does not return there -- it falls through to the independent ECE try-block (`from mlframe.calibration.policy import _ece_score`) and, if `metric_fns` ends up non-empty (e.g. only the ECE import succeeded), the subsequent `bootstrap_metrics(...)` call still runs and populates `out["ece"] = {...}`. The returned dict can therefore carry both a top-level `"status": "skipped"` key (implying total block failure) and a legitimate `"ece"` entry with real CI values -- a shape that a naive downstream consumer checking `if block.get("status") == "skipped"` would misinterpret as "nothing here" when partial results exist.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | test-gap | utils.py:435 | Add a regression test that calls `get_pandas_view_of_polars_df` with a real `pl.Series` (per its own documented contract) so F5 either gets fixed or the contract gets narrowed deliberately. |
| PR2 | ml-best-practice | trainer.py:174; _feature_importances.py; _confidence_analysis.py | Thread `sample_weight` through OOF computation (F2) and the diagnostic FI/confidence paths (F8) for full weighted-training parity; ship as one change since the plumbing is similar. |
| PR3 | observability | _direct_horizon_bucket_forecaster.py:73 | `predict()` silently leaves `NaN` for any row whose `(group, bucket)` has no fitted model (out-of-range `horizon_day`, unseen group at predict time, or `NaN` in `group_col`). No logger exists in this file at all. Add a `logging`-based WARNING summarizing the count/fraction of NaN rows so operators notice missing bucket coverage instead of discovering it downstream. |
| PR4 | code-quality | logging_transformers.py:185 | The dynamically-created `ProxyCls = type("_LoggingProxy", (_LoggingProxy,), {})` doesn't declare its own `__slots__`, so every wrapped instance gets a `__dict__` in addition to the base class's slotted `_inner` -- minor unnecessary memory overhead per wrapped estimator. Add `"__slots__": ()` to the `type()` call's namespace dict. |
| PR5 | consistency | _calib_oof_outputs.py:53 | Generalize the `fit_params` forward-gate (F7) from `model_type_name == "CatBoostRegressor"` to `model_type_name in CATBOOST_MODEL_TYPES` (or document why only the regressor qualifies). |
| PR6 | test-gap | drift_report.py | Add a unit test for `compute_label_distribution_drift` with an empty-but-non-None `train_target` and `target_type="multiclass_classification"` to pin the desired behaviour (graceful message vs the current bare `ValueError`, F9). |
| PR7 | thread-safety | metrics_registry.py:58 | `_REGISTRY` is a bare module-level mutable dict written by `register_metric`/`unregister_metric` with no lock. Currently only written at import time by the four `_register_builtin_*` calls plus any user code calling `register_metric` -- fine today, but worth a comment (or a lock) if runtime registration from parallel workers is ever added. |

## Coverage notes

- I did not execute any test, benchmark, or reproduction script to empirically confirm F2/F3's downstream consequences (sample_weight-dropped OOF quality delta, or a live `clone()` reproducing an MLP collapse) -- both are derived from reading the code paths in my scope plus one `grep` to confirm `_compute_oof_preds`'s only caller and confirm `sklearn.clone()` is the mechanism used. I flagged them as P1 "real bug, narrower/less-common blast radius" rather than P0 specifically because I could not observe the failure live and because both require a specific precondition (sample_weight in use; MLP collapse-ladder exhaustion) to trigger.
- Per the audit's exclusion list, I did not read or evaluate `src/mlframe/feature_selection/filters/**`, `src/mlframe/feature_selection/shap_proxied_fs/**`, or their mirrored test directories, even though a couple of files in my scope (`_helpers_training_configs.py`, `models.py`) import symbols that ultimately originate near those trees (e.g. `mlframe.feature_selection.wrappers.RFECV`, `mlframe.feature_selection.importance.plot_feature_importance`) -- I only read the imported symbol's usage in my files, not its implementation.
- I did not open `_trainer_train_and_evaluate.py` or `_trainer_configure.py` (the two ~600-700 LOC siblings `trainer.py`'s tail re-exports from) since they are not in my assigned 27-file list; any bugs specific to their internals are out of scope for this report.
