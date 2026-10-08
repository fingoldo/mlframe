# pydoclint signature report (DOC106/DOC107)

Result: `pydoclint --style=google src/mlframe` reports 0 findings (was 53 in 36 functions).

## Annotations chosen (SIGNATURE only; no defaults, bodies or names changed except two typed locals)
- core/ewma.py `ewma`: `x: Any`; added `from typing import Any`.
- feature_engineering/mps.py `backfill_zeros`: `arr: np.ndarray, direction: str = "right") -> np.ndarray`; body local `out: np.ndarray` (mypy no-any-return).
- boruta_shap/_shadow_stats.py `calculate_Zscore(array: Any) -> np.ndarray` (typed local `zscores`), `feature_importance(self: Any, normalize: bool) -> Any`.
- filters/_cat_kway_materialize.py `_build_kway_chained_lookup`: `dtype: Any`.
- filters/_conditional_permutation.py: `statistic_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = None`.
- filters/_evaluation_driver.py `_hoist_round_shared_columns`: selected_vars/y/dtype/mrmr_relevance_algo `Any`, factors_data/factors_nbins `np.ndarray`, `-> tuple`.
- filters/_hermite_fe_optimise.py `_eval_coef_pair_batch`: ndarray coefs/z/y/B_*, `Callable[..., Any]` closures, `Sequence[str]` bf_names, str/int/float/bool scalars, `Optional[...]` for the None defaults, `-> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- filters/_mrmr_artifacts.py `_fill_marginals_and_su`: `feature_names_in: Any`, `dtype: Any`.
- filters/_mrmr_degenerate.py `record_degenerate_column_audit`: `self: Any`.
- filters/_mrmr_fe_step/_step_pairs_rank.py `_make_cached_operand`: `self: Any, X: Any, cols: Any -> Callable[[int], Optional[np.ndarray]]`.
- filters/_mrmr_fit_impl/_eng_dedup_scan.py `_strongest_first_cmp`: `_eng_dedup_prefer: Callable[[str, str], bool]`.
- filters/evaluation.py `_materialise_knob_columns`: ndarray/Optional[np.ndarray]/Optional[int]/Optional[list]/Any per parameter.
- filters/feature_engineering.py `apply_gpu_unary_batched`: `cols_data: Any -> Any` (CuPy array).
- wrappers/_univariate_ht.py `calculate_relevance_table`: `y: Any`.
- rfecv/_configs.py `SearchConfig.__init__` fallback: `**kwargs: Any) -> None`.
- rfecv/_diagnostics.py `selection_stability_`: `self: Any`.
- rfecv/_stability_select.py `_sklearn_ranking_vector`: `Sequence[Any]`, `Optional[Sequence[Any]]`, `Any`.
- metrics/_gpu_metrics.py `gpu_multiple_roc_auc_scores` / `gpu_multiple_pr_auc_scores`: `(actual: Any, predicted: Any) -> Any` (numpy or cupy in, cupy out).
- training/_calibration_models.py `_PerClassIsotonicCalibrator.__init__`: `calibrators: dict ... -> None`.
- training/_precompute.py: `train_df: Union[pd.DataFrame, pl.DataFrame]` (Optional[...] = None for the composite stub).
- training/core/_post_xt_ensemble_mtr.py `fit(self, X: Any, y: Any)`.
- training/core/_phase_recurrent.py `_apply_recurrent_to_ensemble`: `ctx: TrainingContext` (TYPE_CHECKING import), `target_type: Any`, `target_values: Any`.
- training/neural/_muon_optimizer.py `Muon.__init__`, `MuonAdamWHybrid.__init__`: `params: Iterable[torch.nn.Parameter]`, `betas: Tuple[float, float]`.
- training/neural/base/_base_fit.py `fit(X: Any, y: Any, sample_weight: Optional[Any] = None, **fit_params: Any) -> Any`.
- training/neural/base/_base_predict.py `_predict_raw`, `predict` (x2), `predict_proba`: `X: Any`.
- training/neural/data.py `_create_dataloader`: `features: Any`, `labels/sample_weight: Optional[Any] = None`.
- training/pipeline/_pipeline_helpers.py `_extract_feature_selector(Any) -> Any`, `_is_fitted(Any) -> bool`.
- training/pipeline/_pipeline_helpers_apply.py `_apply_pre_pipeline_transforms`: Any for models/frames/targets, bool flags, `Optional[str]` model_file_name, `-> tuple`.
- training/strategies/__init__.py `get_strategy(model_name: Any)`.
- training/utils.py `compute_model_input_fingerprint`: `df_at_fit: pd.DataFrame | pl.DataFrame`.

## Results
- mypy-full pre-push hook: first run 2 no-any-return errors (mps.py, _shadow_stats.py), fixed with typed locals; second run Passed (1857 files). Log: %TEMP%\mypy_pdc_agent.log
- `pydoclint --style=google src/mlframe`: No violations.
- `ruff check src --ignore C901`: all passed. black_filtered_apply --check on 30 changed files: clean.
- tests/test_meta/test_public_annotations.py: passes; it reported 14 drained functions, so tests/test_meta/_annotation_baseline.json (the actual file name) was refreshed with --refresh-annotation-baseline (shrink only).
- Targeted tests (muon x2, test_caching_caching_coverage, conditional_permutation strata, test_mps, ewma alpha validation): 41 passed. Log: %TEMP%\pdc_agent_targeted.log

## Open issues (not resolved)
- tests/test_meta/test_pydoclint_baseline.py FAILS, for a reason independent of this work. The test runs plain `pydoclint src/mlframe` (no `--style=google`, so the default numpy style), which still reports 1003 findings (DOC101/103/111/201 etc.), unlike the google-style run that has 0. The instruction "baseline must end with no accepted findings" is therefore not achievable without changing the test's pydoclint invocation (or the config); I did not touch the test. I pruned the 43 entries of tests/test_meta/_pydoclint_baseline.json that no longer correspond to a finding (1045 -> 1002 accepted; includes the DOC106/DOC107 ones I fixed). One NEW finding exists that is not in the baseline and is not from my edits: `src/mlframe/training/composite/cache_store.py::__init__:DOC001` (docstring parsed with an empty parameter name). That is what fails the test; it needs a docstring fix in cache_store.py or a baseline note.
- The tree has other uncommitted modifications by others (e.g. tests/test_meta/test_no_unsafe_module_reload.py, _nested_parallel_scan.py); not mine.

## Changed paths (all under C:\Users\Admin\Machine learning\mlframe)
- src/mlframe/core/ewma.py
- src/mlframe/feature_engineering/mps.py
- src/mlframe/feature_selection/boruta_shap/_shadow_stats.py
- src/mlframe/feature_selection/filters/_cat_kway_materialize.py
- src/mlframe/feature_selection/filters/_conditional_permutation.py
- src/mlframe/feature_selection/filters/_evaluation_driver.py
- src/mlframe/feature_selection/filters/_hermite_fe_optimise.py
- src/mlframe/feature_selection/filters/_mrmr_artifacts.py
- src/mlframe/feature_selection/filters/_mrmr_degenerate.py
- src/mlframe/feature_selection/filters/evaluation.py
- src/mlframe/feature_selection/filters/feature_engineering.py
- src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_pairs_rank.py
- src/mlframe/feature_selection/filters/_mrmr_fit_impl/_eng_dedup_scan.py
- src/mlframe/feature_selection/wrappers/_univariate_ht.py
- src/mlframe/feature_selection/wrappers/rfecv/_configs.py
- src/mlframe/feature_selection/wrappers/rfecv/_diagnostics.py
- src/mlframe/feature_selection/wrappers/rfecv/_stability_select.py
- src/mlframe/metrics/_gpu_metrics.py
- src/mlframe/training/_calibration_models.py
- src/mlframe/training/_precompute.py
- src/mlframe/training/utils.py
- src/mlframe/training/core/_phase_composite_post_xt_ensemble/_post_xt_ensemble_mtr.py
- src/mlframe/training/core/_phase_recurrent.py
- src/mlframe/training/neural/_muon_optimizer.py
- src/mlframe/training/neural/base/_base_fit.py
- src/mlframe/training/neural/base/_base_predict.py
- src/mlframe/training/neural/data.py
- src/mlframe/training/pipeline/_pipeline_helpers.py
- src/mlframe/training/pipeline/_pipeline_helpers_apply.py
- src/mlframe/training/strategies/__init__.py
- tests/test_meta/_annotation_baseline.json
- tests/test_meta/_pydoclint_baseline.json
