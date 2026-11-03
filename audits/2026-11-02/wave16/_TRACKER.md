# Wave 16 audit 2026-11-02 -- tracker

Three read-only reports by Opus 5.5 low-effort agents: avoidable whole-frame memory use, import-time hygiene, and the quality of errors on plausible wrong inputs. The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | MEM16-1 | memory | High | functional adapters deep-copied the whole pandas frame to ordinal-encode a few string columns (`feature_selection/functional_adapters.py`) |
| **RESOLVED** | MEM16-2 | memory | High | Boruta built about three full frames per round; one (n, 2p) buffer is now refilled in place with the same permutation draws (a pandas frame with mixed column dtypes keeps the per-round concat) |
| **RESOLVED** | MEM16-3 | memory | High | RFECV's float64 numpy view of a pandas X is built only up to 2 GiB; above it the fit reads the pandas frame and logs the choice (native-dtype fitting was not changed: the view is a documented float64 estimator-input contract) |
| **RESOLVED** | MEM16-4 | memory | Medium | the cat-pair and binned-numeric-agg stages appended columns with `pd.concat`, which copies the whole input on pandas 2; they now use `append_columns` (shallow copy plus column assignment) |
| **RESOLVED** | MEM16-5 | Medium | `CorrelatedFeaturesSelector.fit` converted a polars frame with `to_pandas()`, which consolidates blocks into a second copy; it now uses the Arrow-backed `get_pandas_view_of_polars_df`. The clustering-parity check came first: on a frame with floats, a nullable float, integers and a categorical, dtypes match (the categorical only differs in its `ordered` flag) and the spearman / pearson / SU cluster assignments are identical; the pandas and polars fits agree on clusters and support (`tests/feature_selection/filters/test_correlated_features_polars_input_is_not_copied.py`) |
| **RESOLVED** | MEM16-6 | memory | Medium | ACE forced a copy of an array it only reads (`feature_selection/ace.py`) |
| **RESOLVED** | MEM16-7 | memory | Medium | the blocked correlation clustering built two more full-size temporaries while standardising; it now divides in place and frees the float copy (peak 9.4 MB to under 5 MB on a 3.2 MB float32 frame) |
| **RESOLVED** | MEM16-8 | memory | Medium | `compose_pair_fe` copied its input matrix up front although the cascade only reads it until `np.column_stack`; the copy is gone (the int-to-float64-to-crit double cast was left as it is: it keeps the returned originals exact) |
| **RESOLVED** | MEM16-9 | memory | Low | `compute_naive_outlier_score` cast train and test to float64; floating inputs now keep their dtype (the comparison against train min/max is exact in float32) |
| **RESOLVED** | MEM16-10 | memory | Low | the RFECV auto-tune fingerprint copied the whole array to float64; numeric arrays keep their dtype and each column is widened on its own |
| **RESOLVED** | ERR16-1 | error_quality | High | the default-extractor path indexed `df` before it was validated: Path, list, ndarray and LazyFrame gave AttributeError or a "use select()" TypeError |
| **RESOLVED** | ERR16-2 | error_quality | Medium | a `.csv` path or a parquet file without the target column failed inside polars |
| **RESOLVED** | ERR16-3 | error_quality | Low | a datetime or timedelta target column was routed to regression; the default extractor now raises a ValueError naming the dtype and the two ways out |
| **RESOLVED** | IMP16-1 | import | High | the CatBoost `__sklearn_clone__` patch is installed by a post-import hook (`utils/post_import_hook.py`) when catboost is loaded, not by importing catboost at `import mlframe.metrics` |
| **RESOLVED** | IMP16-2 | import | High | `trainer.py` and `_model_factories.py` no longer import torch at module level; the single use site imports it locally and `trainer` serves `torch`, `nn`, `F` through a lazy module `__getattr__` for existing importers |
| **RESOLVED** | IMP16-3 | import | Medium | ngboost is imported only where an NGB model is built; the three unused module-level imports are removed (a missing ngboost now raises ImportError at the builder) |
| **RESOLVED** | IMP16-4 | import | Medium | the import-time warm-ups of the discovery correlation and collinearity kernels are removed; they compile on the first real call |
| **RESOLVED** | IMP16-5 | import | Low | the ACF/PACF kernels moved to `mlframe/utils/acf_kernels.py` and `acf_lag_selection` imports them from there, no longer through `mlframe.reporting.charts`; the other module-level `matplotlib.pyplot` imports in `calibration/quality.py`, `evaluation/reports.py`, `reporting/charts/shap_panels.py` and `boruta_shap/_io_plot.py` are untouched, and `import mlframe.feature_engineering` still loads `mlframe.reporting` through another module |
| **RESOLVED** | IMP16-6 | import | Low | the GPU-runtime wrapper around `is_cuda_available` is installed by a post-import hook when `pyutilz.system.gpu_dispatch` is imported; `import mlframe` no longer imports `pyutilz.system` (the wrapper still replaces a function in another package once that package loads) |
| **REJECTED** | IMP16-7 | import | Low | `import mlframe.training` sets `LOKY_MAX_CPU_COUNT`; the comment in `training/__init__.py` records this single env write as a deliberate choice after the physical-core patch was made opt-in |
| **RESOLVED** | IMP16-8 | import | Low | `callbacks/stop_file.py` imported lightgbm at module level only to test availability |
| **RESOLVED** | IMP16-9 | import | Low | the Pareto plot is drawn on an Agg `Figure` that is not registered with pyplot, so `matplotlib.use("Agg", force=True)` is gone and the process keeps its backend |

## Counts

23 rows. Dispositions: 20 RESOLVED, 2 FUTURE, 1 REJECTED.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| MEM16-1, MEM16-6 | Stop deep-copying the whole frame in the functional adapters and ACE | `tests/feature_selection/test_selection_inputs_are_not_deep_copied.py` (checks the numeric buffer stays shared; the ACE `copy=False` change has no separate test) |
| MEM16-2 | Reuse one shadow buffer across Boruta rounds | `tests/feature_selection/filters/test_boruta_shadow_buffer_reuse.py` (the reuse test fails without the fix) |
| IMP16-6 | Wrap is_cuda_available when pyutilz.system.gpu_dispatch is imported, not at import mlframe | `tests/test_gpu_runtime_trigger_is_installed_lazily.py` (the first test fails without the fix); 10 GPU opt-out and autoconfig tests pass |
| IMP16-1 | Install the CatBoost clone patch when catboost is imported, not when mlframe.metrics is | `tests/utils/test_post_import_hook.py` (catboost-not-imported test fails without the fix); 47 clone-safety tests pass |
| IMP16-2 | Stop importing torch when the training package is imported | `tests/training/test_torch_is_not_imported_by_training_core.py` (fails without the fix); the 5 MLP tests that import `nn` from `trainer` failed first and pass after the lazy `__getattr__` |
| IMP16-3 | Import ngboost only where an NGB model is built | `tests/training/test_ngboost_is_imported_only_when_an_ngb_model_is_built.py` (fails without the fix); 29 NGB-related tests pass |
| IMP16-5 | Move the ACF and PACF kernels to a neutral module | `tests/feature_engineering/test_acf_lag_selection_does_not_import_reporting.py` (checks the shared functions and the pure-numpy module; 65 ACF tests pass) |
| IMP16-9 | Draw the Pareto plot on an Agg Figure instead of switching the matplotlib backend | `tests/training/slicing/test_pareto_plot_keeps_the_matplotlib_backend.py` (fails without the fix) |
| IMP16-4 | Stop compiling the discovery correlation kernels at import | `tests/training/composite/test_discovery_numba_kernels_not_compiled_at_import.py` (fails without the fix); 24 discovery kernel tests pass |
| MEM16-7 | Standardise the blocked correlation matrix in place | `tests/feature_selection/test_hybrid_corr_clusters_blocked_memory.py` (the memory test fails without the fix) |
| MEM16-8 | Drop the up-front copy of the input matrix in compose_pair_fe | `tests/feature_selection/filters/test_compose_pair_fe_no_upfront_copy.py` (fails without the fix) |
| ERR16-3 | Refuse a datetime target in the default extractor | `tests/training/test_default_extractor_rejects_datetime_target.py` |
| MEM16-9 | Score outliers on float32 inputs without casting both frames to float64 | `tests/preprocessing/test_naive_outlier_score_keeps_float32.py` (fails without the fix) |
| MEM16-10 | Read the auto-tune fingerprint's ndarray without a float64 copy | `tests/feature_selection/wrappers/test_auto_tune_signals_do_not_copy_the_frame.py` (fails without the fix) |
| MEM16-4 | Append engineered columns without copying the input frame | `tests/feature_selection/test_append_columns_shares_buffers.py`; 43 tests of the two stages pass |
| MEM16-3 | Gate RFECV's float64 numpy view of a pandas X at 2 GiB | `tests/feature_selection/wrappers/test_rfecv_numpy_view_size_gate.py` |
| ERR16-1, ERR16-2 | Validate df before the default extractor reads it in train_mlframe_models_suite | `tests/training/test_suite_default_extractor_input_errors.py` (fails without the fix) |
| IMP16-8 | Probe for lightgbm in the stop-file callbacks instead of importing it | `tests/training/test_stop_file_callback_does_not_import_lightgbm.py` (fails without the fix) |

## Not covered by this wave

- Memory: only feature selection, preprocessing and the training entry paths were read; the cost of a fix was not measured, and none of the FUTURE rows was benchmarked.
- Errors: calibration, evaluation, metrics and preprocessing functions were not traced case by case.
- Imports: `matplotlib.use` in the Pareto plot was outside the agent's import scope and is listed from its note.
