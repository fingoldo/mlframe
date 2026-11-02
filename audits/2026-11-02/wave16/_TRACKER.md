# Wave 16 audit 2026-11-02 -- tracker

Three read-only reports by Opus 5.5 low-effort agents: avoidable whole-frame memory use, import-time hygiene, and the quality of errors on plausible wrong inputs. The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | MEM16-1 | memory | High | functional adapters deep-copied the whole pandas frame to ordinal-encode a few string columns (`feature_selection/functional_adapters.py`) |
| **FUTURE** | MEM16-2 | memory | High | Boruta builds about three full frames per round (`filters/_boruta.py:177-184`); fix is one preallocated (n, 2p) buffer refilled in place |
| **FUTURE** | MEM16-3 | memory | High | RFECV makes a float64 copy of every all-numeric pandas X (`rfecv/_fit.py:554`), 2x-8x the frame; needs a size gate or native dtype and a check which estimators need float64 |
| **FUTURE** | MEM16-4 | memory | Medium | `pd.concat([X, new_df], axis=1)` copies the input in `_cat_pair_fe.py:534` and `_binned_numeric_agg_fe.py:799` |
| **FUTURE** | MEM16-5 | memory | Medium | `CorrelatedFeaturesSelector.fit` converts a polars frame with `to_pandas()`; the zero-copy Arrow view changes column dtypes and an earlier fix there was about exactly dtype preservation, so it needs a clustering-parity check first |
| **RESOLVED** | MEM16-6 | memory | Medium | ACE forced a copy of an array it only reads (`feature_selection/ace.py`) |
| **FUTURE** | MEM16-7 | memory | Medium | hybrid selector standardises into a second full array (`hybrid_selector.py:129-138`) |
| **FUTURE** | MEM16-8 | memory | Medium | `compose_pair_fe` makes up to three full-size arrays (`filters/composition.py:114-129`) |
| **FUTURE** | MEM16-9 | memory | Low | `compute_naive_outlier_score` casts train and test to float64 (`preprocessing/outliers.py:206`) |
| **FUTURE** | MEM16-10 | memory | Low | RFECV auto-tune fingerprint copies X to float64 for heuristic signals (`wrappers/_auto_tune.py:86`) |
| **RESOLVED** | ERR16-1 | error_quality | High | the default-extractor path indexed `df` before it was validated: Path, list, ndarray and LazyFrame gave AttributeError or a "use select()" TypeError |
| **RESOLVED** | ERR16-2 | error_quality | Medium | a `.csv` path or a parquet file without the target column failed inside polars |
| **FUTURE** | ERR16-3 | error_quality | Low | a datetime target column is routed to regression by `_infer_target_is_classification`; where it first fails downstream was not traced |
| **FUTURE** | IMP16-1 | import | High | `import mlframe.metrics` installs the CatBoost `__sklearn_clone__` patch at import, costing about 3.2 s (catboost, IPython, xgboost); consumers rely on the import to guarantee the patch, so it needs a post-import hook or explicit installs at each CatBoost construction site |
| **FUTURE** | IMP16-2 | import | High | `training/_model_factories.py` imports torch (12.5 s) at module level although the neural stack is meant to be deferred |
| **FUTURE** | IMP16-3 | import | Medium | ngboost imported eagerly in four modules (4.1 s with sympy) |
| **FUTURE** | IMP16-4 | import | Medium | six `parallel=True` numba kernels compile at `import mlframe.training.core` (`_corr_numba.py`, `_collinear_numba.py`) |
| **FUTURE** | IMP16-5 | import | Low | `matplotlib.pyplot` and `mlframe.reporting` imported at module level in deep helpers |
| **FUTURE** | IMP16-6 | import | Low | `import mlframe` imports `pyutilz.system` and replaces `is_cuda_available` in it |
| **REJECTED** | IMP16-7 | import | Low | `import mlframe.training` sets `LOKY_MAX_CPU_COUNT`; the comment in `training/__init__.py` records this single env write as a deliberate choice after the physical-core patch was made opt-in |
| **RESOLVED** | IMP16-8 | import | Low | `callbacks/stop_file.py` imported lightgbm at module level only to test availability |
| **FUTURE** | IMP16-9 | import | Low | `training/slicing/_slice_pareto_plot.py:184` calls `matplotlib.use("Agg", force=True)` when a plot is drawn, switching the backend for the process |

## Counts

23 rows. Dispositions: 5 RESOLVED, 17 FUTURE, 1 REJECTED.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| MEM16-1, MEM16-6 | Stop deep-copying the whole frame in the functional adapters and ACE | `tests/feature_selection/test_selection_inputs_are_not_deep_copied.py` (checks the numeric buffer stays shared; the ACE `copy=False` change has no separate test) |
| ERR16-1, ERR16-2 | Validate df before the default extractor reads it in train_mlframe_models_suite | `tests/training/test_suite_default_extractor_input_errors.py` (fails without the fix) |
| IMP16-8 | Probe for lightgbm in the stop-file callbacks instead of importing it | `tests/training/test_stop_file_callback_does_not_import_lightgbm.py` (fails without the fix) |

## Not covered by this wave

- Memory: only feature selection, preprocessing and the training entry paths were read; the cost of a fix was not measured, and none of the FUTURE rows was benchmarked.
- Errors: calibration, evaluation, metrics and preprocessing functions were not traced case by case.
- Imports: `matplotlib.use` in the Pareto plot was outside the agent's import scope and is listed from its note.
