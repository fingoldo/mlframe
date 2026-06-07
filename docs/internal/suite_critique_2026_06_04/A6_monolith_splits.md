# A6 — Monolith split plans (mlframe, 2026-06-04)

READ-ONLY architecture pass. Authoritative list regenerated from `src/**/*.py` (lines > 1000), sorted descending. **25 files** qualify.

## Method & conventions applied

- AST skeleton (top-level defs/classes + nested defs + line spans) extracted for every file; module-level state (caches/locks/registries/globals/`__all__`) catalogued; existing bottom-of-file re-exports recorded.
- **Project split convention** (CLAUDE.md "Monolith split"): move a cohesive sub-block to a NEW SIBLING module; parent RE-EXPORTS the moved names at its bottom (`from ._sibling import X`) so `from parent import X` keeps working. Lazy-import parent helpers inside the moved body when init-order/cycles bite.
- **AST gate (mandatory before any commit):** walk each new sibling; every `ast.Name` Load not module-bound, not builtin, not function-local is a candidate runtime `NameError` (Python resolves names lazily at call time, so a missing `from .parent import _helper` import-checks clean but blows up on first call, often swallowed into a recurring WARN). Add explicit imports; smoke-import + `hasattr`; the split sensor must CALL into the moved body, not just import it.

Many of these packages are ALREADY mid-split (filters/=120 files, training/core=46, neural=27, wrappers=21, metrics=26) and most monoliths already carry bottom re-exports — the convention is mature and the remaining monoliths are the residue that resisted the easy carves. Two distinct shapes dominate:

1. **Bag-of-functions** files (metrics extras, recipe builders, the FE-scorer modules) — trivially splittable by responsibility, low risk. **Do these first.**
2. **Single giant function / `__init__`** files (`_fit_impl` 5799-line fn; `_train_one_target` 952; `_run_fe_step` 980; `check_prospective_fe_pairs` 862; `_composite_discovery_fit.fit` 924; `screen_predictors` 962; `_build_cross_target_ensemble_for_target` 848; `MRMR.__init__` 1590-of-which-1550-is-the-param-doc) — NOT carve-by-symbol. They need **extract-helper** (pull self-contained inner phases into module-level sibling functions the body calls) or are **genuinely cohesive / unsplittable** (flagged as such).

---

## Tier 1 — Bag-of-functions, low-risk symbol carves (do first)

### 1. `metrics/_regression_extras.py` — 1136 LOC

40 free functions: per-metric `fast_*` public wrappers each with a private `_*_kernel_seq`/`_par` numba kernel, plus a fused block, plus a Tweedie family. No classes, only module-state `_KENDALL_NUMBA_MAX_N`. Cleanly grouped by metric family.

Proposed siblings (parent keeps the fused block + re-exports):
- `_regression_error_metrics.py` (~360) — `_rmsle_*`/`fast_rmsle`, `_mape_*`/`fast_mape_mean`, `_smape_*`/`fast_smape`, `fast_mdape`, `_wmape_*`/`fast_wmape`, `_naive_mae_kernel`/`fast_mase`, `_mean_bias_error_kernel`/`fast_mean_bias_error`, `fast_cv_rmse`, `_huber_loss_kernel`/`fast_huber_loss`.
- `_regression_assoc_metrics.py` (~230) — `_pearson_corr_kernel`/`fast_pearson_corr`, `fast_spearman_corr`, `_kendall_tau_b_kernel`/`fast_kendall_tau` (+ `_KENDALL_NUMBA_MAX_N`), `fast_concordance_index`, `_nash_sutcliffe_kernel`/`fast_nash_sutcliffe`, `_explained_variance_kernel`/`fast_explained_variance`.
- `_regression_deviance_metrics.py` (~200) — `_tweedie_deviance_*_kernel` (poisson/gamma/general), `_maybe_warn_tweedie`, `fast_poisson_deviance`, `fast_gamma_deviance`, `fast_tweedie_deviance`.
- Parent keeps (~330) `_fused_regression_ext_pass1/2_seq/par` + `fast_regression_metrics_block_extended` (calls the individual kernels — keep kernel defs together with it OR lazy-import; see risk) + re-exports.

Risk: LOW. `fast_regression_metrics_block_extended` re-calls several `_*_kernel` njit funcs; if those move, the parent must `from ._regression_error_metrics import _rmsle_kernel_seq, ...` (njit funcs cross modules fine; the AST gate will flag every kernel name it references — add them). `_KENDALL_NUMBA_MAX_N` travels WITH the kendall kernel. Priority **High value, S effort** (heavy reuse, clean seams).

### 2. `metrics/_classification_extras.py` — 1049 LOC

32 free functions: scalar binary metrics, multiclass metrics, calibration/probability metrics, and three fused blocks. No module-state.

Proposed siblings:
- `_classification_scalar_metrics.py` (~330) — `matthews_corrcoef_binary`, `cohen_kappa_binary`, `balanced_accuracy_binary`, `g_mean_binary`, `gini_from_auc`, `specificity_npv_fpr_fnr`, `f_beta_score`, `lift_at_k`, `top_k_accuracy`, `_multiclass_confusion_kernel`/`matthews_corrcoef_multiclass`.
- `_classification_calibration_metrics.py` (~260) — `_brier_skill_score_kernel`/`brier_skill_score`, `_spiegelhalter_z_kernel`/`spiegelhalter_z`, `_rps_kernel`/`_par`/`ranked_probability_score`, `_hosmer_lemeshow_kernel`/`hosmer_lemeshow_test`, `accuracy_ratio`.
- `_classification_ks.py` (~70) — `_ks_statistic_kernel`/`ks_statistic`.
- Parent keeps (~360) the confusion primitives (`_confusion_counts_binary*`/`_dispatch`) + the three fused blocks (`fast_binary_confusion_metrics_block`, `fast_binary_probability_metrics_block`, `fast_multiclass_confusion_metrics_block`) + re-exports.

Risk: LOW. Fused blocks call the scalar kernels; add `from ._classification_scalar_metrics import ...` to parent if blocks need them (verify with AST gate — likely they inline their own kernels). Priority **High, S**.

### 3. `feature_selection/filters/discretization.py` — 1254 LOC

27 free functions: binning-edge algorithms (Knuth, Bayesian-blocks), the categorize/digitize core, the 1D/2D discretizers (numpy + njit + CUDA), and the dataset-level driver. Module-state `_DISCRETIZE_2D_CUDA_MIN_CELLS`.

Proposed siblings:
- `_discretization_edges.py` (~290) — `_knuth_log_posterior`, `_knuth_bin_edges`, `_bayesian_blocks_inner`, `_bayesian_blocks_midpoints`, `_bayesian_blocks_bin_edges`, `histogram`, `get_binning_edges`, `discretize_sklearn`, `edges`.
- `_discretization_2d.py` (~290) — `_discretize_2d_array_njit`, `discretize_2d_array`, `discretize_2d_array_cuda`, `_discretize_quantile_rawkernel` (+ `_DISCRETIZE_2D_CUDA_MIN_CELLS`).
- `_discretization_dataset.py` (~260) — `create_redundant_continuous_factor`, `categorize_dataset` (223-line driver).
- Parent keeps (~400) the native helpers + `categorize_1d_array`/`digitize`/`quantize_*`/`discretize_uniform`/`discretize_array`/`_handle_missing`/`_maybe_collect_lazy` + re-exports.

Risk: MED. `categorize_dataset` calls many parent helpers (`categorize_1d_array`, `_handle_missing`, native encoders) — lazy-import inside body to dodge any cycle, OR top-import (probably no cycle, parent is leaf-ish). CUDA path (`_discretize_quantile_rawkernel`) carries its own lazy `cupy` init — keep that init local to the sibling. AST gate must verify the 2D njit/cuda funcs see their module-globals. Priority **High, M** (discretization is on every MRMR hot path; central).

### 4. `feature_selection/filters/engineered_recipes.py` — 2420 LOC

~55 functions: a registry of `build_*_recipe` factories + matching `_apply_*` executors + `apply_recipe` dispatcher + the `EngineeredRecipe` dataclass. ALREADY partly split (6 bottom re-exports to `_orthogonal_triplet_fe_recipes`, `_numeric_decompose_fe`, `_temporal_agg_fe`, etc.). The recipe families are extremely cohesive — each `build_X`/`_apply_X` pair is independent.

Proposed siblings (group the in-file families that haven't been carved yet):
- `_orth_basis_recipes.py` (~420) — `_apply_orth_pre_transform`, `_eval_orth_basis_column`, `_apply_orth_univariate`, `_apply_orth_pair_cross`, `build_orth_univariate_recipe`, `build_orth_pair_cross_recipe`, `build_orth_diff_basis_recipe`, `build_orth_cluster_basis_recipe`, `_bspline_basis_values`, `_fit_spline_knots`, `_apply_orth_spline`, `_apply_orth_fourier`, `build_orth_spline_recipe`, `build_orth_fourier_recipe`.
- `_encoding_recipes.py` (~360) — `_apply_target_encoding`, `_apply_kfold_target_encoded`/`build_*`, `_apply_count_encoded`, `_apply_frequency_encoded`, `_apply_cat_num_residual`, `build_count_encoded_recipe`, `build_frequency_encoded_recipe`, `build_cat_num_residual_recipe`, `build_cat_pair_cross_recipe`, `build_cat_triple_cross_recipe`.
- `_grouped_recipes.py` (~340) — `build_grouped_delta_recipe`, `build_grouped_agg_recipe`, `build_composite_group_agg_recipe`, `build_grouped_quantile_recipe`, `build_target_aware_group_bin_recipe`, `build_lagged_diff_recipe`, `_apply_cluster_aggregate`, `build_cluster_aggregate_recipe`, `_apply_factorize`, `_apply_factorize_kway`.
- `_missingness_ratio_recipes.py` (~180) — `build_missing_indicator_recipe`, `build_missingness_count_recipe`, `build_missingness_pattern_recipe`, `build_pairwise_ratio_recipe`, `build_mi_greedy_transform_recipe`, `_apply_mi_greedy_transform`.
- Parent keeps (~700) `EngineeredRecipe` dataclass, `_extra_equal`, `_extract_column`, `_coerce_to_int_with_nan_handling`, `apply_recipe` (central dispatcher), unary/binary family (`_apply_unary_binary`, `build_unary_binary_recipe`), hermite-pair (`_apply_hermite_pair`, `build_hermite_pair_recipe`), `_orjson_pp`, + ALL re-exports.

Risk: MED. `apply_recipe` is the dispatcher that `if kind == ...` routes into every `_apply_*` — it must import all moved `_apply_*` (top-level import in parent for the moved siblings; AST gate flags each). The `_apply_*` executors reference `EngineeredRecipe` and `_extract_column`/`_coerce_to_int_with_nan_handling` → siblings need `from .engineered_recipes import EngineeredRecipe, _extract_column, _coerce_to_int_with_nan_handling` — **lazy-import inside bodies** because the parent imports the siblings (circular). This is the highest-care item in Tier 1. Priority **High, M-L** (2420 LOC, very central to FE).

### 5. `feature_selection/filters/_orthogonal_univariate_fe.py` — 2291 LOC

~33 functions: univariate-basis generation, MI-uplift scoring (sklearn + numba backends), pair-cross-basis generation+scoring, the `*_with_recipes` wrappers, and the heavy extra-basis (spline/Fourier/chirp) detector block. Has `__all__`, module-state `_BASIS_CODE`/`_MI_BACKEND`/`_EXTRA_BASIS_KINDS`. `generate_extra_basis_features` alone is 302 lines; `_detect_fourier_freqs_for_col` 147.

Proposed siblings:
- `_orth_mi_backends.py` (~140) — `_mi_classif_batch_sklearn`, `_mi_classif_batch_numba`, `_select_mi_backend`, `_mi_classif_batch`, `score_features_by_mi_uplift` (+ `_MI_BACKEND`).
- `_orth_pair_cross_fe.py` (~470) — `_pair_eng_col_name`, `generate_pair_cross_basis_features`, `score_pair_cross_basis_by_mi_uplift`, `hybrid_orth_mi_pair_fe`, `hybrid_orth_mi_pair_fe_with_recipes`, `_col_basis_for_recipe`.
- `_orth_extra_basis_fe.py` (~720) — the spline/Fourier/chirp detector cluster: `_fit_spline_for_col`, `_fit_fourier_for_col`, `_fit_chirp_warp_for_col`, `_chirp_axis`, `_corr_sq_centered`, `_periodogram_power`, `_power_centered`, `_refine_peak_freq`, `_deflate_sincos`, `_detect_fourier_freqs_for_col`, `_heldout_smooth_r2`, `_detect_fourier_freq_for_col`, `generate_extra_basis_features`, `_build_recipe_from_meta`, `hybrid_orth_extra_basis_fe_with_recipes` (+ `_EXTRA_BASIS_KINDS`).
- Parent keeps (~560) `_evaluate_basis_column`, `_dedup_collinear_source_cols` (199), `basis_route_by_signal`, `generate_univariate_basis_features`, `hybrid_orth_mi_fe`, `hybrid_orth_mi_fe_with_recipes` (+ `_BASIS_CODE`) + re-exports + `__all__`.

Risk: MED. Siblings call shared helpers (`_evaluate_basis_column`, `generate_univariate_basis_features`, `build_basis_matrix` from hermite_fe) — lazy-import the parent helpers inside bodies (parent imports siblings → cycle). Note `_orthogonal_scorer_auto_fe.py` already re-exports `generate_univariate_basis_features` from here; keep that name in the parent. `__all__` must be extended with re-exported names. Priority **High, M** (2291 LOC, core FE).

### 6. `feature_selection/filters/_orthogonal_scorer_auto_fe.py` — 1169 LOC

17 functions split cleanly into "auto best-scorer-per-column" and "rank-fusion ensemble" halves. Has `__all__`, module-state `SCORER_NAMES`/`ENSEMBLE_AGGREGATORS`/`MUTUAL_RANK_AGGREGATORS`; re-exports `generate_univariate_basis_features`.

Proposed siblings:
- `_orth_auto_scorer_fe.py` (~480) — the per-column scorer family: `_score_plug_in`, `_score_ksg`, `_score_copula`, `_score_dcor`, `_score_hsic`, `_compute_lcb`, `_bootstrap_subsample_indices`, `select_best_scorer_per_column`, `score_features_by_auto_scorer_uplift`, `hybrid_orth_mi_auto_scorer_fe`, `hybrid_orth_mi_auto_scorer_fe_with_recipes` (+ `SCORER_NAMES`).
- Parent keeps (~600) the ensemble/rank-fusion family: `_compute_per_scorer_rank_table`, `_aggregate_ranks`, `score_features_by_ensemble_uplift`, `hybrid_orth_mi_ensemble_fe`, `hybrid_orth_mi_ensemble_fe_with_recipes` (+ `ENSEMBLE_AGGREGATORS`, `MUTUAL_RANK_AGGREGATORS`) + re-exports + `__all__`.

Risk: LOW-MED. Ensemble half calls the `_score_*` family → parent imports them from the new sibling (`from ._orth_auto_scorer_fe import _score_plug_in, ...`); AST gate flags each. `SCORER_NAMES` travels with the scorers. Priority **Med, S-M**.

### 7. `feature_selection/wrappers/_helpers.py` — 1020 LOC

14 functions: thread-pinning utils, train/test split, CV-score storage, the two big workhorses `_conditional_permutation_importance` (143) and `get_feature_importances` (354), ranking + next-subset suggestion (incl. scipy local/global). Module-state `_MULTITHREADED_ESTIMATOR_PATTERNS`/`_THREAD_PARAMS`/`_PERM_AUTO_CELL_CAP`; already re-exports `_knockoffs`.

Proposed siblings:
- `_helpers_importance.py` (~520) — `_conditional_permutation_importance` (+ `_PERM_AUTO_CELL_CAP`), `get_feature_importances`, `select_appropriate_feature_importances`, `_impute_ragged_fi_table`, `get_actual_features_ranking`.
- `_helpers_subset_search.py` (~170) — `get_next_features_subset`, `_suggest_dichotomic`, `_suggest_scipy_local`, `_suggest_scipy_global`.
- Parent keeps (~330) threading utils (`_detect_multithreaded`, `_pin_threads_to_one` + `_MULTITHREADED_ESTIMATOR_PATTERNS`/`_THREAD_PARAMS`), `suppress_irritating_3rdparty_warnings`, `split_into_train_test`, `store_averaged_cv_scores` + re-exports.

Risk: LOW. `_rfecv.py` re-exports `get_feature_importances`/`get_next_features_subset`/`get_actual_features_ranking`/`select_appropriate_feature_importances` FROM `_helpers` — keep those names re-exported at the parent bottom so `_rfecv`'s import is unaffected. `get_feature_importances` may call `_detect_multithreaded`/`_pin_threads_to_one` — sibling needs `from ._helpers import _detect_multithreaded, _pin_threads_to_one`. Priority **Med, S-M**.

### 8. `feature_selection/_shap_proxy_revalidate.py` — 1802 LOC

~22 free functions + small `HonestLossCache` class: disk-cache key builders, honest-loss + permutation-importance machinery, sampling utilities, and three big public entry points (`proxy_trust_guard` 220, `revalidate_top_n` 344, `within_cluster_refine` 380). Module-state: 4 cache-prefix/param constants.

Proposed siblings:
- `_shap_proxy_loss.py` (~340) — `_build_honest_loss_disk_key`, `_build_perm_fit_disk_key`, `_open_disk_cache`, `_expand`, `HonestLossCache`, `_try_cap_n_estimators`, `_loss_from_predictions`, `_honest_loss`, `_permutation_importance_ranking`, `_parallel_honest_losses` (+ the 4 prefix/param constants).
- `_shap_proxy_sampling.py` (~180) — `_softmax_weights`, `_weighted_choice_no_replace`, `_zipf_card_probs`, `_sample_anchor_subsets`.
- `_shap_proxy_refine.py` (~520) — `_ucb_stop_remaining_cannot_win`, `_winner_from_per_candidate`, `_ucb_auto_slack`, `active_learning_revalidate`, `within_cluster_refine`, `importance_topk_ablation`.
- Parent keeps (~620) `proxy_trust_guard`, `revalidate_top_n` (the two most-called public APIs) + re-exports.

Risk: MED. `revalidate_top_n` / `within_cluster_refine` / `proxy_trust_guard` all call `_honest_loss`, `_parallel_honest_losses`, `_permutation_importance_ranking`, the sampling helpers, and the UCB helpers — heavy cross-references. Parent imports moved names from each sibling; the refine sibling imports loss+sampling helpers. No cycle if loss/sampling are leaf siblings (they don't import back). AST gate critical here (many cross-refs). Priority **High, M**.

---

## Tier 2 — Class with many small methods (carve method clusters)

### 9. `feature_selection/boruta_shap.py` — 1045 LOC

Single `BorutaShap` class (950) + `load_data` (29). `fit`/`explain` ALREADY moved to `_boruta_shap_fit_explain` (re-exported). The remaining 43 methods are small (mostly < 50 lines) and group by responsibility. sklearn-class-method-cluster pattern: pull cohesive method groups into mixins in siblings.

Proposed siblings (mixin classes the main class inherits, OR module-level helpers bound as methods — prefer mixins for `self`-heavy methods):
- `_boruta_shap_importance.py` (~230) as `BorutaShapImportanceMixin` — `create_shadow_features`, `calculate_Zscore`, `feature_importance`, `isolation_forest`, `calculate_hits`, `create_importance_history`, `update_importance_history`, `store_feature_importance`, `create_mapping_between_cols_and_indices`.
- `_boruta_shap_stats.py` (~150) as `BorutaShapStatsMixin` — `binomial_H0_test`, `bonferoni_corrections`, `test_features`, `TentativeRoughFix`, `calculate_rejected_accepted_tentative`, `get_5_percent`, `get_5_percent_splits`, `find_sample`, `symetric_difference_between_two_arrays`, `find_index_of_true_in_array`.
- `_boruta_shap_io_plot.py` (~150) as `BorutaShapReportMixin` — `results_to_csv`, `plot`, `box_plot`, `create_mapping_of_features_to_attribute`, `to_dictionary`, `Subset`, `create_list`, `filter_data`, `hasNumbers`, `load_data` (free fn).
- Parent keeps (~400) `__init__`, `check_model`, `check_X`, `check_missing_values`, validation/encoding methods, `transform`, `fit_transform`, the class declaration with the mixin bases + re-exports.

Risk: MED. This is a vendored third-party class (BorutaShap) with idiosyncratic naming; methods share lots of `self.*` state set in `fit`. Mixin split is mechanically safe (methods stay methods, `self` resolves at call time) but the AST gate's "function-local" heuristic must treat `self` correctly and NOT flag `self.X` attrs. Verify `_boruta_shap_fit_explain` still finds every method it calls on `self`. Lower value: it's a single vendored estimator, not on the core suite hot path. Priority **Low-Med, M**.

### 10. `feature_selection/wrappers/_rfecv.py` — 1136 LOC

`RFECV` class (1041) + nothing else. `__init__` is 375 lines (param storage + validation). `fit`, `_fit_stability_selection`, `select_optimal_nfeatures_` ALREADY moved (re-exported). Remaining methods: SFFS swap, checkpoint I/O, and a large cluster of `@property` diagnostic accessors (stability/pareto/elbow/bootstrap-CI).

Proposed siblings:
- `_rfecv_diagnostics.py` (~290) as `RFECVDiagnosticsMixin` — `cv_results_df_`, `selection_stability_`, `n_features_one_se_`, `stability_vs_n_curve_`, `n_stability_elbow_`, `pareto_front_`, `pareto_knee_`, `n_features_bootstrap_ci_` (all read-only `@property` accessors over fitted state).
- `_rfecv_checkpoint.py` (~90) as `RFECVCheckpointMixin` — `_save_checkpoint`, `_load_checkpoint`.
- Parent keeps (~760) `__init__` (375), `_sffs_swap_pass`, `__sklearn_tags__`, `get_feature_names_out`, `transform`, class decl with mixin bases + re-exports.

Risk: LOW-MED. Properties only READ `self.*_` fitted attrs → safe as a mixin (no new imports, `self` resolves lazily). `_save/_load_checkpoint` may reference module-level config dataclasses (`SearchConfig` etc.) — those are already imported at parent top and re-exported; the checkpoint sibling needs `from ._rfecv_configs import ...`. Note: `__init__` at 375 lines is mostly param storage + the giant docstring — NOT separately splittable (see "unsplittable" note). Priority **Med, M** (RFECV is central; the diagnostics carve is clean).

### 11. `training/neural/_flat_torch_module.py` — 1183 LOC

Single `MLPTorchModel(LightningModule)` class. Methods cluster into: Lightning train/val loop hooks, loss computation, optimizer config, and the heavy predict-acceleration block (`_maybe_compile_predict_forward` 73, `_maybe_cuda_graph_forward` 169, `_invalidate_predict_caches`, `predict_step` 74).

Proposed siblings (mixins):
- `_flat_torch_predict_accel.py` (~360) as `_PredictAccelMixin` — `_invalidate_predict_caches`, `_maybe_compile_predict_forward`, `_maybe_cuda_graph_forward`, `predict_step`, `_apply_torch_compile`.
- `_flat_torch_loss.py` (~230) as `_LossMixin` — `_unpack_batch`, `_loss_unreduced`, `_compute_weighted_loss`, `compute_metrics`.
- Parent keeps (~600) `__init__` (119), `__getstate__`/`__setstate__`, `forward`, `training_step`, `on_train_epoch_end`, `validation_step`, `on_validation_epoch_end`, `configure_optimizers`, `on_train_end`, class decl with mixin bases.

Risk: MED-HIGH. **Pickle hazard** (memory: "Runtime caches break pickle"): `__getstate__`/`__setstate__` already exclude live torch/CUDA-graph state; the cuda-graph + compiled-forward caches live as instance attrs set by the accel mixin methods. Splitting the methods across mixins does NOT change instance-attr layout, so pickle stays correct AS LONG AS `__getstate__` exclusion list (in parent) still names every cache attr the moved methods create. **Verify the exclusion list covers attrs created in the moved `_maybe_cuda_graph_forward`/`_maybe_compile_predict_forward`** before committing — run the pickle suite. The accel methods reference module-level torch-compile guards; mixin needs `from ._flat_torch_module import ...` only if those guards live in parent (they're inline). Priority **Med, M** (neural path, but pickle-fragile — handle with care).

---

## Tier 3 — Giant single function / `__init__` (extract-helper, not symbol-carve)

These have no symbol-level seams; the work is to pull self-contained inner phases into module-level sibling functions that the body calls (passing the needed locals as args). Higher effort, higher correctness risk (must thread locals carefully), so RANK BELOW Tier 1/2. Each is itemized with the concrete phase boundaries found.

### 12. `feature_selection/filters/_mrmr_fit_impl.py` — 5993 LOC ⭐ (biggest)

One function `_fit_impl` (5799 lines) — the MRMR fit body. It is a **sequential FE pipeline**: dozens of self-contained "Layer NN" stages, each with the SAME shape (gate on a config flag → generate engineered columns via a `filters/*` helper → MI/CMI-uplift score → append columns + register `EngineeredRecipe`), bracketed by a setup head (L195-516), a big cross-stage dedup/remap/finalize tail (L4300-5460), and a target-type/return-assembly tail (L5610-5993).

This is the **single highest-value split in the suite** and the most tractable Tier-3 target because the stages are uniform and already comment-banner-delimited.

Proposed approach — extract each FE stage (or cluster of stages) into `_mrmr_fit_stage_*.py` siblings, each exposing one `def apply_<stage>(ctx) -> None` that takes a small mutable context object (or an explicit tuple of `data, cols, engineered_recipes, hybrid_orth_features_, rng, config-flags, ...`) and mutates/returns the appended columns + recipes. Concrete stage clusters (LOC approx from banner spans):

- `_mrmr_fit_stage_orth_basis.py` (~520) — Layer 21/22 univariate + pair-cross-basis stages, Layer 32 extra-basis (B-spline/Fourier) (L357-683).
- `_mrmr_fit_stage_highorder_cross.py` (~440) — Layer 56 tri-product, Layer 77 quadruplet, Layer 78 adaptive-arity, Layer 57 adaptive per-column degree, Layer 58 conditional basis routing (L684-1248).
- `_mrmr_fit_stage_scorer_family.py` (~420) — Layer 65/66/67 auto-scorer + Layer 68/69 bootstrap-LCB / rank-fusion stages (L1249-2492).
- `_mrmr_fit_stage_encoding.py` (~520) — Layer 33/34 (count/freq/cat-num-residual/kfold-TE), cat-pair/triple crosses, Layer 89/94 synergy crosses (L2785-3850, encoding subset).
- `_mrmr_fit_stage_grouped.py` (~520) — Layer 87/93 grouped + composite-key aggregates, Layer 88 grouped-quantile, Layer 92 temporal leak-safe, Layer 90 numeric-decompose, Layer 95 periodic + group-distance (L3473-4290).
- `_mrmr_fit_dedup_remap.py` (~620) — the cross-stage dedup (Layer 27/64), per-family cleanup mirror (Layer 33/34/37/38/87/88/89/90/93/94), Layer 91 unified second-pass CMI gate, feature_names_in_ exclusion + recipe-routing remap (L4300-5118).
- `_mrmr_fit_finalize.py` (~480) — target-type resolution (L5610-5770), cols-space stage / return assembly (L5771-5993), cluster-membership accessors (Layer 41/48).
- Parent `_fit_impl` keeps (~1500) the setup head, the FE-step loop / cat-FE driver, the screening orchestration, and the sequence of `apply_<stage>(...)` calls + the existing `_dispatch_default_scorer`, `_mrmr_instance_state_size_bytes`, `_mrmr_cache_bytes_total`.

Risk: **HIGH** (highest in the audit). (a) Each stage shares a LARGE local namespace (`data`, `cols`, `nbins`, `y`, `rng`, `engineered_recipes`, `hybrid_orth_features_`, dozens of config flags) — the extracted fn signature must thread them explicitly; a dropped local becomes a runtime `NameError` swallowed into the exact "name 'X' is not defined → fold reported NaN" WARN that CLAUDE.md calls out as the canonical sibling-split regression. (b) RNG threading: stages advance a shared RNG; order + state MUST be preserved (memory: determinism whenever an earlier stage advanced it). (c) The nested `_eng_dedup_prefer` (L4367) closure moves with the dedup block. (d) `_run_fe_step` lazy-imports back into this module — verify no new cycle. **Mandatory:** the split sensor must run a FULL `MRMR.fit` on a synthetic with each FE family enabled and assert bit-identical support + recipes vs pre-split (the AST gate alone is insufficient given the local-threading risk). Do this incrementally — ONE stage cluster per commit, re-run the biz_value MRMR suite each time. Priority **High value, L effort** (do it, but carefully and incrementally).

### 13. `feature_selection/filters/mrmr.py` — 3119 LOC

`MRMR` class (2928). `fit`/`_fit_impl`/`_run_fe_step`/transform/partial_fit/provenance/semi-supervised ALREADY moved (8 bottom re-exports). The residual bulk is **`__init__` = 1590 lines, of which ~1550 are the parameter list + interleaved explanatory comments (1 assignment, 1229 comment/doc lines)** — this is a single function signature and **CANNOT be split** by the sibling convention. `__setstate__` (268) and `fit` (294, the thin orchestration wrapper kept here) are the only other large bodies.

Proposed siblings:
- `_mrmr_setstate.py` (~290) — `__setstate__` (268-line pickle-compat migration) + `clear_fit_cache` if it shares migration logic. Exposed as a module-level `def _mrmr_setstate(self, state)` bound, or a mixin method.
- `_mrmr_fit_helpers.py` (~280) — `recommend_default_scorer`, `_stability_outer_fit` (+ nested `_inner_selector`), `_resolve_target_prefix`, `_coerce_target_dtype`, `_rfecv_cv_kwargs`, `_maybe_resample_for_sample_weight`, `_print_fit_summary`, `_fit_identity_shortcut` as a `_MRMRFitHelpersMixin`.
- `_mrmr_recommend.py` (~150) — `recommend_enabled_fe`, `export_artifacts`, `get_feature_names_out` as `_MRMRRecommendMixin`.
- Parent keeps (~2100, dominated by the un-splittable `__init__`) — `__init__`, `fit` (thin wrapper), `get_support`, `transform`, `__sklearn_is_fitted__`, class decl with mixins + all re-exports.

**Un-splittable note:** the file will stay > 1000 LOC purely because of the `__init__` parameter doc-block. The realistic LOC reduction is only ~700 (down to ~2400). The legitimate way to shrink `__init__` is the SEPARATE concern flagged by CLAUDE.md ("no audit/phase/refactor junk in comments" + 160-char rule): the inline param comments carry banned date/wave/Layer stamps (`2026-05-29 Wave 7`, `Layer 32`, `F13`, `A1`) that a comment-hygiene pass would strip — but that is a comment cleanup, NOT a module split, and must not be conflated. Flag it; don't do it under "split". Priority **Med, M** for the mixin carves; the `__init__` is **genuinely un-carve-able**.

### 14. `training/core/_phase_train_one_target_body.py` — 1019 LOC

One function `_train_one_target` (952). Setup helpers + schema + MLP helpers ALREADY moved (re-exported). The body is a long linear orchestration with cohesive blocks: (a) pre-pipeline / transformer caching setup (L149-228), (b) the per-model strategy loop with suite-scoped feature caching (the bulk, L~230-990), (c) cache-stats merge + writeback tail (L992-1019).

Proposed approach — extract the per-model inner loop body and the cache-setup into siblings:
- `_phase_train_one_target_caches.py` (~180) — the suite-scoped cache scaffolding (tier_dfs / pl.Enum map / prepared-polars-frame / neural-budget setup, L149-228) as `def _setup_target_caches(...) -> CacheBundle`.
- `_phase_train_one_target_model_loop.py` (~520) — the per-model strategy loop body as `def _run_model_for_target(model, strategy, caches, ...) -> result`, called once per model.
- Parent `_train_one_target` keeps (~320) the outer setup, the loop driver calling `_run_model_for_target`, and the writeback tail.

Risk: HIGH. Same local-threading hazard as `_fit_impl` but smaller. The loop body reads many setup locals (caches, strategy maps, target slug, y, sample_weight) and writes to per-suite accumulators (`cache_stats`, `_non_neural_train_times`, `slug_to_original_target_*`) — pass these as an explicit mutable bundle; a dropped write silently breaks suite-level telemetry. Memory note: "CODE-LOW-4 per-target reset is INTENTIONAL" — preserve that reset semantics across the extract. Sensor must run `train_mlframe_models_suite` end-to-end (multi-model, multi-target) and assert identical metrics + cache_stats. Priority **Med-High, L** (central to the suite; risky locals).

### 15. `feature_selection/filters/_mrmr_fe_step.py` — 1018 LOC

One function `_run_fe_step` (980). The FE-iteration step delegated out of `_fit_impl`. Linear body with phases (banners at L249, L367, L418): FE pool resolution + bootstrap (first-step only), the generate/score block, the promote-survivors + recipe-persist block, parent-provenance backfill, cluster-aggregate (first-step) block.

Proposed approach — extract the post-generation bookkeeping:
- `_mrmr_fe_step_promote.py` (~280) — survivor promotion + `selected_vars` reconciliation + recipe persistence + parent-provenance backfill (L568-916 cluster).
- `_mrmr_fe_step_cluster_agg.py` (~120) — the first-step cluster-aggregate emission (L972-1016 block).
- Parent `_run_fe_step` keeps (~620) FE-pool/bootstrap setup + the generate/score driver + the calls into the two extracted helpers.

Risk: MED-HIGH. Threads `selected_vars`, `engineered_recipes`, the MRMR instance (`getattr` keeps signature stable per the inline comment), `data`/`cols`. Tight contract with `_fit_impl` (callers depend on the FE survivors being promoted under `fe_max_steps==1`). Sensor: MRMR.fit with `fe_max_steps` 1 and >1, assert engineered columns appended + recipes present. Priority **Med, M-L**.

### 16. `feature_selection/filters/_feature_engineering_pairs.py` — 1006 LOC

`_select_single_best` (43), `_neg_name_key` class (15), and `check_prospective_fe_pairs` (862). Module-state `_PREWARP_UNARY`, `_PREWARP_SPECS_RESULT_KEY`, `_TIMES_SPENT_LOCK` (a lock!). The big function is the pair-FE candidate evaluator: prewarp fitting, pair enumeration, MI scoring, dedup, result packing.

Proposed approach:
- `_feature_engineering_pairs_prewarp.py` (~180) — prewarp-spec fitting block (operand/pair prewarp, the `_PREWARP_UNARY` consult) extracted as `def _fit_prewarp_specs(...)`.
- `_feature_engineering_pairs_score.py` (~280) — the pair-enumeration + MI-scoring + dedup core as `def _score_pairs(...)`.
- Parent `check_prospective_fe_pairs` keeps (~520) the orchestration + result packing + `_select_single_best` + `_neg_name_key` + module-state (the `_TIMES_SPENT_LOCK` MUST stay in the parent and be passed to any helper that times, never re-created).

Risk: MED-HIGH. **`_TIMES_SPENT_LOCK` is shared module-level mutable state** — if a helper that records timing moves to a sibling, it must `from ._feature_engineering_pairs import _TIMES_SPENT_LOCK` (NOT create its own) or the timing accumulator splits across two locks (the "runtime caches/registries duplicate after split" failure class). AST gate flags it. Sensor: run the pair-FE biz_value tests. Priority **Med, M-L**.

### 17. `training/_composite_discovery_fit.py` — 1074 LOC

`_process_mem_mb` (47), `_phase_ram_report` (58), and `fit` (924, containing nested `_eval_one_transform` 271). The composite-target discovery fit: candidate resolution → per-(base,transform) scoring (the nested `_eval_one_transform`) → parallel dispatch → alpha-drift detection → Phase-B tiny-model rerank → multi-base forward-stepwise.

Proposed approach:
- `_composite_discovery_eval.py` (~300) — the nested `_eval_one_transform` lifted to module level as `def _eval_one_transform(work_item, shared_arrays, config, ...)` (it is ALREADY a self-contained closure — the cleanest Tier-3 extract in the suite). Plus the parallel-dispatch wrapper.
- `_composite_discovery_rerank.py` (~220) — Phase-B tiny-model rerank (L913-940) + multi-base forward-stepwise auto-promotion (L941-1066).
- Parent `fit` keeps (~520) candidate resolution / MI down-sampling / work-list build / alpha-drift bookkeeping / the dispatch call / final assembly + `_process_mem_mb` + `_phase_ram_report`.

Risk: MED. `_eval_one_transform` already takes most state via closure → converting closure vars to explicit params is mechanical but must be exhaustive (the AST gate run on the lifted fn will list every free var to promote to a param — use it directly). The forward-stepwise block references the per-candidate base arrays stashed earlier (`ENS-Low-6` hoist) — thread that dict explicitly. Sensor: composite-discovery biz_value tests. Priority **Med, M**.

### 18. `feature_selection/filters/_screen_predictors.py` — 1002 LOC

One function `screen_predictors` (962) + `_pool_warmup_noop` (7). The greedy MRMR screening loop: RNG snapshot/restore, DCD-state construction, cardinality-bias pre-screen, Westfall-Young maxT permutation-null floor, the greedy confirmation loop (the bulk, using `ScreenContext`/`confirm_one_predictor` from `_confirm_predictor`), termination-reason summary, RNG restore in `finally`.

Proposed approach:
- `_screen_predictors_prescreen.py` (~180) — cardinality-bias pre-screen (L436-509) + Westfall-Young maxT null floor (L510-573) as two helpers returning the filtered pool + the gain floor.
- Parent `screen_predictors` keeps (~820) the RNG snapshot/restore envelope (the `try/finally` with numpy+numba+cupy seed save/restore MUST stay in the parent — it brackets everything), DCD setup, the greedy loop, the summary.

Risk: HIGH. (a) The **RNG snapshot/restore `try/finally`** is the spine of the function — do NOT move it; any extracted helper must run INSIDE the envelope. (b) The greedy confirmation loop is too entangled with `ScreenContext` mutation to extract cleanly without large param threading — leave it in the parent. Realistic reduction is modest (~180 LOC); the file may stay near 800. (c) The pre-screen helpers must NOT re-snapshot/restore RNG (the parent owns it) — pass the active RNG in. Sensor: a deterministic screen with fixed seed, assert identical selected_vars. Priority **Med, M** (only the pre-screen carves are safe; the loop is cohesive).

### 19. `training/core/_phase_composite_post_xt_ensemble.py` — 1168 LOC

`MTRPerColumnEqualMeanEnsemble` class (190), `_build_mtr_per_column_ensemble` (100), and `_build_cross_target_ensemble_for_target` (848 — the monster). The big function builds a per-target cross-target ensemble: OOF prediction collection, candidate-ensemble enumeration, honest-loss scoring, selection. Module-state `_DEFAULT_OOF_RANDOM_STATE`, `_PROB_NORM_EPS`.

Proposed approach:
- `_post_xt_ensemble_mtr.py` (~300) — `MTRPerColumnEqualMeanEnsemble` (self-contained estimator) + `_build_mtr_per_column_ensemble`.
- `_post_xt_ensemble_build.py` (~400) — extract a cohesive mid-section of `_build_cross_target_ensemble_for_target`: the OOF collection + candidate enumeration block as `def _collect_oof_and_enumerate(...)`, and/or the honest-loss scoring + selection block as `def _score_and_select_ensembles(...)`.
- Parent keeps (~470) the `_build_cross_target_ensemble_for_target` orchestration calling the extracted helpers + module-state.

Risk: MED-HIGH. The 848-line function has few internal banners (only one cohesive comment cluster at L351-360) — must read the body to find clean sub-block boundaries before extracting; the section structure is less obvious than `_fit_impl`'s. Threads OOF arrays + per-target predictions (potentially large — respect the 100GB-frame rule, pass references not copies). `_DEFAULT_OOF_RANDOM_STATE`/`_PROB_NORM_EPS` stay in parent, passed in. Priority **Med, M-L**.

### 20. `feature_selection/shap_proxied_fs.py` — 1818 LOC

5 free resolver fns + `ShapProxiedFS` class (1619). `__init__` is 661 lines (88 assignments + ~470 inline param comments — a real-code init, unlike MRMR's), `fit` is 705. `_resolve_*` methods + MMR filter cluster present.

Proposed approach:
- `_shap_proxied_resolvers.py` (~180) — the 5 module-level `_resolve_brute_force_*`/`_resolve_cluster_su_*`/`_resolve_adaptive_prescreen_*` free fns + the class's `_resolve_booster_kind`, `_resolve_revalidation_ucb_stdev_multiplier`, `_resolve_revalidation_mmr_jaccard_threshold`, `_mmr_filter_by_jaccard`, `_resolve_optimizer` as a `_ShapProxiedResolversMixin` (+ the 7 `_DEFAULT_*`/`_*_OPTIMIZERS` module constants).
- `_shap_proxied_fit.py` (~720) — `fit` (705) + `_run_search` (40) as `_ShapProxiedFitMixin`. (This mirrors the boruta_shap and mrmr "move the big fit out" pattern that already exists in the suite.)
- Parent keeps (~720, dominated by `__init__`'s param-doc) — `__init__`, `preflight`, `_to_pandas`, `_coerce_target`, `transform`, `fit_transform`, `get_support`, `get_feature_names_out`, class decl with mixins + re-exports.

Risk: MED. `fit` is the biggest single method; as a mixin it keeps `self` resolution. It calls the resolver methods + `_run_search` + the external `_shap_proxy_revalidate` entry points — mixin needs those imports (resolvers via the other mixin / parent; revalidate via top-import). `__init__` at 661 lines: the ~470 comment lines again carry banned date/Layer stamps — comment-hygiene could shrink it, but that's separate from the split. AST gate on the fit mixin (large body, many `self`/module refs). Priority **Med, M**.

### 21. `feature_selection/filters/_dynamic_cluster_discovery.py` — 1783 LOC

`use_dcd`/`set_dcd_active` toggles, `SwapDecision` + `DCDState` classes, `make_dcd_state`, `pair_su` (233), `pair_vi`, `should_be_pruned`, `discover_cluster_members`, `_select_swap_method_auto` (132), `evaluate_swap_candidate` (451 — monster), `commit_swap` (243), `dcd_summary`. Has `__all__`, **module-state `_DCD_STATE` (a live singleton via `make_dcd_state`!)** + `_AUTO_METHOD_CANDIDATES`; already re-exports `_dcd_tau_auto`, `_dcd_pair_su_batch`.

Proposed siblings:
- `_dcd_metrics.py` (~340) — `pair_su` (233), `pair_vi`, `_binarize_aggregate`, `should_be_pruned` (the pairwise redundancy metric kernels).
- `_dcd_swap.py` (~620) — `_select_swap_method_auto`, `evaluate_swap_candidate` (451), `commit_swap` (243) (+ `_AUTO_METHOD_CANDIDATES`).
- Parent keeps (~600) `use_dcd`/`set_dcd_active`, `SwapDecision`, `DCDState`, `make_dcd_state`, `discover_cluster_members`, `dcd_summary`, **`_DCD_STATE` singleton** + `__all__` + re-exports.

Risk: HIGH. **`_DCD_STATE` is a module-level mutable singleton.** Every moved fn that reads/writes it MUST `from ._dynamic_cluster_discovery import _DCD_STATE` (or, better, take the state as an explicit arg — the funcs already accept a `DCDState`, so prefer arg-passing and avoid the global entirely). Re-creating the singleton in a sibling = two divergent DCD states (the canonical registry-duplication failure). `commit_swap` is threaded with `engineered_recipes` (host-MRMR dict, per inline comment) — thread it. `evaluate_swap_candidate`'s 451 lines may itself want a sub-extract but is internally cohesive — leave whole. AST gate critical for `_DCD_STATE`/`DCDState`/`SwapDecision`/`_AUTO_METHOD_CANDIDATES`. Sensor: a DCD swap round on synthetic collinear clusters, assert identical swap decisions + state. Priority **Med-High, M-L**.

### 22. `feature_selection/filters/_confirm_predictor.py` — 1024 LOC

`_conditioning_rows_per_cell`, `_candidate_is_engineered`, `_prefer_engineered_order`, `_extract_single_raw_parent`, `_confirmable_engineered_child`, `ScreenContext` class (88), `score_candidates` (227), `confirm_candidate` (188), `confirm_one_predictor` (199). Module-state `_PARENT_TOKEN_SPLIT`; re-exports `mi_direct` from `.permutation`.

Proposed siblings:
- `_confirm_predictor_engineered.py` (~250) — `_candidate_is_engineered`, `_prefer_engineered_order`, `_extract_single_raw_parent`, `_confirmable_engineered_child`, `_conditioning_rows_per_cell` (+ `_PARENT_TOKEN_SPLIT`) — the directed-FE / prefer-engineered helper cluster.
- Parent keeps (~770) `ScreenContext`, `score_candidates`, `confirm_candidate`, `confirm_one_predictor` (the core confirmation machinery — tightly coupled via `ScreenContext`) + re-export of `mi_direct`.

Risk: MED. The confirm functions reference the engineered-helpers (`_confirmable_engineered_child`, `_prefer_engineered_order`) → parent imports them from the new sibling (`from ._confirm_predictor_engineered import ...`); the sibling needs `_PARENT_TOKEN_SPLIT` to travel with it. `score_candidates`/`confirm_candidate`/`confirm_one_predictor` are too `ScreenContext`-entangled to separate from each other — keep them together in the parent. Modest reduction (~250 LOC; parent stays ~770). Priority **Med, M**.

### 23. `training/neural/base.py` — 1698 LOC

`_make_binary_focal_loss`, `_validate_no_nan_inf`, `PytorchLightningEstimator` class (1329) with `_fit_common` (796!) and `_predict_raw` (404), `PytorchLightningRegressor`, `PytorchLightningClassifier`. Module-state `_PREDICT_ONLY_DM_PARAM_KEYS`; already re-exports `_base_tensor_helpers`, `_base_sklearn_params`, `_base_callbacks`.

Proposed approach (two of the three big methods are extractable):
- `_base_predict.py` (~440) — `_predict_raw` (404) as `_PredictMixin` + `_PREDICT_ONLY_DM_PARAM_KEYS` + the `PytorchLightningClassifier.predict`/`predict_proba` if they reuse it.
- `_base_losses.py` (~120) — `_make_binary_focal_loss`, `_validate_no_nan_inf` (free fns, module-level).
- `_base_fit.py` (~820) — `_fit_common` (796) + `fit`/`partial_fit` thin wrappers as `_FitMixin`. (Mirrors the existing `_base_*` sibling pattern.)
- Parent keeps (~320) the class declarations (`PytorchLightningEstimator` with mixin bases, `Regressor`, `Classifier`), `__getstate__`/`__setstate__`, `__init__`, `score` + re-exports.

Risk: MED-HIGH. `_fit_common` (796) and `_predict_raw` (404) as mixin methods keep `self`; both touch torch/lightning + `_base_tensor_helpers`/`_base_callbacks` (already siblings — the mixins re-import them). **Pickle:** `__getstate__`/`__setstate__` stay in parent; verify they still cover any instance attr set in the moved `_fit_common`. The accelerator/trainer live objects must be excluded (memory: "caching live framework objects needs __getstate__ exclusion"). `_fit_common` at 796 lines is itself a candidate for further phase-extraction but is highly cohesive (one training run) — split as a whole first. Sensor: fit+predict+pickle round-trip on a tiny MLP. Priority **Med, M-L** (pickle-fragile).

### 24. `feature_selection/filters/hermite_fe.py` — 1276 LOC

Large bag of numba/CUDA polynomial-eval kernels (4 bases × 3 backends), the polyeval dispatcher + oracle, preprocessing dispatch, `HermiteResult` class, prewarp/ALS fitting, KSG MI. Heavy module-state (`_BASIS_BUILDERS`, `_CUDA_AVAILABLE`, `_NJIT_FUNCS`, `_NJIT_PAR_FUNCS`, `_PAR_THRESHOLD`, `_CUDA_THRESHOLD`, `_POLY_BASES`, `_polyeval_oracle_singleton`, `_DEFAULT_BIN_FUNCS`, ...). Already re-exports `_hermite_fe_optimise`, `_hermite_fe_mi`. **This is the canonical numerical-kernel-ladder reference module (CLAUDE.md cites it explicitly).**

Proposed siblings:
- `_hermite_polyeval_kernels.py` (~360) — the raw njit/parallel/cuda Horner kernels: `_hermeval_njit`/`_legval_njit`/`_chebval_njit`/`_lagval_njit` (+ `_parallel` variants) + `_polyeval_cuda` (+ `_NJIT_FUNCS`, `_NJIT_PAR_FUNCS`, `_PAR_THRESHOLD`, `_CUDA_THRESHOLD`, `_CUDA_AVAILABLE` — the dispatcher state).
- `_hermite_polyeval_dispatch.py` (~150) — `_lookup_polyeval_thresholds`, `_polyeval_oracle_enabled`, `_polyeval_size_fingerprint`, `get_polyeval_oracle`, `benchmark_polyeval_cpu_backends`, `_polyeval_oracle_pick_cpu_backend`, `polyeval_dispatch` (+ `_polyeval_oracle_singleton`, `_POLYEVAL_ORACLE_*`).
- `_hermite_prewarp.py` (~210) — `_canonical_seeds`, `_l2_normalize_pair`, `_l2_penalty_value`, `warm_start_als_seed`, `fit_operand_prewarp`, `fit_pair_prewarp_als`, `apply_operand_prewarp`, `_ksg_mi_1d`.
- Parent keeps (~560) basis builders (`_build_basis_*`, `build_basis_matrix`, `_BASIS_BUILDERS`, `_POLY_BASES`), preprocessing (`_preprocess_*`/`_apply_*`/`_make_dispatch`, `_DEFAULT_BIN_FUNCS`), plug-in MI batch fns, `HermiteResult`, `_safe_div`/`_atan2`/`_log_abs_signed`, `basis_route_by_moments` + re-exports.

Risk: **HIGH — kernel-state coupling.** The dispatcher (`polyeval_dispatch`) reads `_NJIT_FUNCS`/`_NJIT_PAR_FUNCS`/`_PAR_THRESHOLD`/`_CUDA_THRESHOLD`/`_CUDA_AVAILABLE` which are populated at import by registering the njit kernels into dicts. If kernels move to a sibling, those dicts must be built in the SAME module as the kernels (`_hermite_polyeval_kernels.py`) and the dispatch sibling must import them — OR the lazy CUDA init (`_ensure_*_kernels`) ordering breaks. The `_polyeval_oracle_singleton` is a live singleton — keep single-homed. Per CLAUDE.md this module is the kernel-ladder REFERENCE — a botched split would mislead every future kernel author. **Highest-care numerical split.** Sensor: bit-identity of `polyeval_dispatch` across all 4 bases × backends + the prewarp biz_value tests. Priority **Med, L** (valuable but the kernel-state graph makes it delicate; do last among Tier-3 or leave if benched-clean).

### 25. `utils/_param_oracle.py` — 1004 LOC

Fingerprint/bucketize free fns, `_ParquetStore` class (113), `ParamOracle` class (434), small helpers. Has `__all__`, module-state `SCHEMA_VERSION`/`_*_DIMS`/`_STORE_COLUMNS`. This is shared infra (`pyutilz`-style kernel-tuning support).

Proposed siblings:
- `_param_oracle_fingerprint.py` (~270) — `_host_key`, `default_store_dir`, `log_bucket`, `linear_bucket`, `_as_2d_numeric`, `default_fingerprint` (116), `bucketize_fingerprint`, `_euclidean_buckets` (+ `_CONTINUOUS_FP_DIMS`, `_LOG_BUCKET_DIMS`, `_LINEAR_BUCKET_DIMS`).
- `_param_oracle_store.py` (~180) — `_ParquetStore` class, `_rss_mb`, `_stable_json`, `_median`, `_loads`, `_combo_key`, `_utc_now_iso` (+ `_STORE_COLUMNS`, `SCHEMA_VERSION`).
- Parent keeps (~560) `ParamOracle` (434) + `_default_objective` + `__all__` + re-exports.

Risk: LOW-MED. `ParamOracle` calls `default_fingerprint`/`bucketize_fingerprint`/`_ParquetStore`/`_median` → parent imports them from the two siblings (`from ._param_oracle_fingerprint import ...`, `from ._param_oracle_store import ...`). No cycle if the siblings don't import `ParamOracle` (they don't). `SCHEMA_VERSION`/`_STORE_COLUMNS` travel with the store. AST gate flags the cross-refs cleanly. Priority **Med, S-M** (clean seams; shared infra so good ROI on readability).

---

## Cross-cutting risks (apply to every split)

- **Lazy name resolution → swallowed `NameError`** (CLAUDE.md core gate): run the AST scope walker on EVERY new sibling; add explicit `from .parent import _helper` for each unresolved Load; the split sensor must CALL the moved body, not just import it.
- **Module-level singletons / locks / registries** (`_DCD_STATE`, `_TIMES_SPENT_LOCK`, `_polyeval_oracle_singleton`, `_NJIT_FUNCS`/`_CUDA_AVAILABLE` dispatch dicts, MRMR fit-cache class attrs): a moved consumer must IMPORT the single instance from its home module, never re-declare it. Prefer arg-passing where the API already accepts the state (DCD funcs take `DCDState`).
- **Circular imports** (parent re-exports the sibling AND the sibling needs a parent helper): lazy-import the parent helper INSIDE the function body. Pattern already used across the codebase (`_run_fe_step` lazy-imports back into `mrmr`).
- **Pickle** (`base.py`, `_flat_torch_module.py`, `mrmr.__setstate__`): mixin/sibling splits don't change instance-attr layout, but verify `__getstate__` exclusion lists still cover attrs created in moved methods; run the pickle suite (memory: "Runtime caches break pickle" — exclusion must land in the SAME change).
- **Determinism / RNG** (`_fit_impl`, `screen_predictors`, `_composite_discovery_fit`): preserve RNG advance order; keep the snapshot/restore envelope in the parent; never let an extracted helper re-snapshot.
- **100GB-frame rule**: extracted helpers must pass frame REFERENCES, never `.copy()`/`.clone()`; preserve mutate-and-restore try/finally patterns.
- **Test pollution**: the split sensors must NOT `del sys.modules`/`importlib.reload` (CLAUDE.md). Use parameterized fits + bit-identity asserts.

## Un-splittable / cohesive (explicitly flagged)

- **`MRMR.__init__`** (1590 LOC) — a single function signature, ~1550 lines of parameters + interleaved comments, 1 statement body. NOT splittable by the convention. `mrmr.py` will stay > 1000 LOC. The only legitimate shrink is a comment-hygiene pass (strip banned date/Wave/Layer stamps per CLAUDE.md) — a SEPARATE task, not a split.
- **`screen_predictors` greedy loop**, **`evaluate_swap_candidate`** (451), **`_fit_common`** (796), **`confirm_*` trio** — internally cohesive single algorithms; extract their PERIPHERY (pre-screen, helpers, accel) but leave the core whole. Forcing a line-count cut through them would create fragile local-threading and is rejected.

## Recommended execution order

1. **Tier 1 bag-of-functions** (items 1,2,3,4,5,6,7,8) — highest ROI, lowest risk, mechanical symbol carves. Start with the two metrics extras (1,2) and `_orthogonal_scorer_auto_fe` (6) as the cleanest.
2. **Tier 2 method-cluster mixins** (10 RFECV diagnostics, 9 boruta, 11 flat-torch, 20 shap-proxied, 23 base) — mechanical mixin moves; watch pickle on 11/23.
3. **Tier 3 extract-helper** in ascending risk: 25 (param oracle, clean) → 17 (composite-discovery, closure already isolated) → 22 (confirm) → 21 (DCD, watch `_DCD_STATE`) → 19 (post-xt) → 16/15 (FE-pairs / fe-step, watch lock + contracts) → 14 (train-one-target) → **12 `_fit_impl`** (biggest value, most care; one stage-cluster per commit) → 24 (hermite, last; kernel-state delicate).
4. **13/`mrmr.py`** mixin carves whenever convenient; accept it stays > 1000 LOC.
