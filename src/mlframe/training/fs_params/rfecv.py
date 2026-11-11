"""Strict parameters of `mlframe.feature_selection.wrappers.rfecv:RFECV`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.fs_params._generate`.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Sequence
from pydantic import BaseModel, ConfigDict


class RFECVParams(BaseModel):
    """Parameters of `mlframe.feature_selection.wrappers.rfecv:RFECV`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ("n_features_selection_rule",)
    __signature_skip_default__ = ("top_predictors_search_method", "votes_aggregation_method")

    search_config: Any = None
    fi_config: Any = None
    robustness_config: Any = None
    fit_params: dict | None = None
    max_nfeatures: int | None = None
    mean_perf_weight: float = 1.0
    std_perf_weight: float = 0.1
    feature_cost: float = 0.0
    smooth_perf: int = 0
    max_runtime_mins: float | None = None
    max_refits: int | None = None
    best_desired_score: float | None = None
    max_noimproving_iters: int = 30
    cv: object | int | None = 3
    cv_shuffle: bool = False
    min_train_size: int | None = None
    early_stopping_val_nsplits: int | None = 10
    early_stopping_rounds: int | None = None
    scoring: object | None = None
    nofeatures_dummy_scoring: bool = True
    top_predictors_search_method: Any = None
    votes_aggregation_method: Any = None
    use_all_fi_runs: bool = True
    use_last_fi_run_only: bool = False
    use_one_freshest_fi_run: bool = False
    use_fi_ranking: bool = False
    importance_getter: str | Callable | None = None
    random_state: int | None = None
    leave_progressbars: bool = True
    verbose: bool | int = 0
    show_plot: bool = False
    optimizer_plotting: str | None = None
    cat_features: Sequence | None = None
    keep_estimators: bool = False
    estimators_save_path: str | None = None
    frac: float | None = None
    skip_retraining_on_same_shape: bool = True
    stop_file: str = "stop"
    report_ndigits: int = 4
    special_feature_indices: list | None = None
    conduct_final_voting: bool = False
    must_include: Sequence | None = None
    n_jobs: int = 1
    force_parallel: bool = False
    must_exclude: Sequence | None = None
    leakage_corr_threshold: float | None = 0.95
    leakage_action: str = "warn"
    mbh_adaptive_threshold: int = 30
    feature_groups: dict | None = None
    n_features_selection_rule: Literal["auto", "argmax", "one_se_min", "one_se_max", "one_se_min_foldstd", "one_se_max_foldstd", "plateau"] = "auto"
    stability_selection: bool = False
    stability_n_bootstrap: int = 50
    stability_threshold: float = 0.6
    stability_top_k: int | None = None
    estimators: Sequence | None = None
    checkpoint_path: str | None = None
    swap_top_k: int = 0
    optimizer_config: dict | None = None
    keep_loser_subset_fi: bool = False
    fi_missing_policy: str = "worst"
    submit_dummy_to_optimizer: bool = True
    swap_top_k_allow_no_es: bool = False
    optimizer_target: str = "mean"
    convergence_tol: float | None = None
    convergence_tol_window: int = 10
    futility_stop: bool = True
    futility_min_iters: int = 5
    futility_alpha: float = 0.05
    futility_patience_frac: float = 0.1
    futility_anchor: str = "full"
    init_design_size: int | str | None = "auto"
    dichotomic_epsilon: float = 0.1
    dichotomic_step: str = "midpoint"
    fi_decay_rate: float = 0.0
    multiclass_coef_aggregation: str = "max"
    coef_scale_source: str = "train"
    cpi_max_depth: int | None = None
    cpi_min_samples_leaf: int = 10
    n_repeats: int = 5
    wide_data_fi_fallback: bool = True
    wide_data_fi_threshold: int = 200
    wide_data_fi_n_repeats: int = 2
    allow_unsafe_aggregation: bool = False
    drop_nan_score_fi: bool = True
    auto_tune: bool = False
    must_exclude_strict: bool = True
    noimprove_counts_revisit: bool = False
    cb_cached_borders: bool = True
    prescreen: str | Callable | None = None
    prescreen_top_k: int | None = None
    prescreen_fdr_level: float = 0.05
    prescreen_nested: bool = True
    multioutput_strategy: str | None = "union"
    drop_id_like_sequences: bool = True
    id_like_ratio_threshold: float = 0.999
    id_like_spacing_cv: float = 0.001
    drop_near_dup_corr: bool = True
    near_dup_corr_threshold: float = 0.999
    nan_in_X_policy: str = "impute"
    nan_indicator_cols: Sequence | None = ()
    importance_agg: str = "dispatched"
    importance_agg_k_cv: float = 1.0
    elimination_rule: str = "importance"
