"""ShapProxiedFS - feature selection by ranking subsets via summed SHAP values.

Idea (Mazzanti / TDS, extended): train one model on all features, compute SHAP values once, then
approximate the OOS prediction of a model trained on any feature subset ``S`` by the coalition value
``base + sum_{j in S} phi_j``. Rank subsets by a proper ML metric of ``(y, proxy_pred)`` instead of
honestly retraining 2^n models, then honestly re-validate the cheap top-N to pick the final subset.

Pipeline: fit big model -> OOF SHAP (per-fold base) -> proxy-rank subsets (exact numba/GPU brute
force for n<=~22, else beam/greedy/GA/annealing/gradient) -> proxy-trust guard -> honest re-validate
top-N on a disjoint holdout -> expose the sklearn selector contract.

Honest about limits: the proxy attributes the *full* model restricted to S, not a model retrained on
S, so it under-credits subsets that drop features whose signal correlated survivors could recover
(the empirical "<50% coverage breaks down" wall). Hence the trust guard, the disjoint-holdout
re-validation, the ``min_selected_ratio`` knob, and the importance-top-k ablation in the report.

sklearn contract mirrors BorutaShap: ``support_`` (bool mask in input-column order),
``selected_features_`` (names in input order), ``feature_names_in_``, ``n_features_in_``;
``transform`` uses name-based ``X.loc[:, selected]``; ``NotFittedError`` before fit.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import (
    _ADAPTIVE_PRESCREEN_FLOOR, _DEFAULT_ADAPTIVE_PRESCREEN_THRESHOLDS,
    _DEFAULT_BRUTE_FORCE_MAX_FEATURES, _DEFAULT_BRUTE_FORCE_N_SUB_GATE,
    _DEFAULT_CLUSTER_SU_AUTO_MAX_FEATURES, _EXACT_OPTIMIZERS, _HEURISTIC_OPTIMIZERS,
    _resolve_adaptive_prescreen_thresholds, _resolve_adaptive_prescreen_width,
    _resolve_adaptive_n_anchors, _resolve_knee_prescreen_cap,
    _resolve_brute_force_max_features, _resolve_brute_force_n_sub_gate,
    _resolve_cluster_su_auto_max_features, ShapProxiedNoCandidatesError,
)
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit import ShapProxiedFitMixin
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_init_params import assign_init_params
from mlframe.feature_selection.shap_proxied_fs._state_compat import ShapProxiedStateCompatMixin
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_methods import ShapProxiedMethodsMixin
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_precomputed import restrict_artifacts
from mlframe._shap_policy_constants import DEFAULT_TRUST_GUARD_FIDELITY_WEIGHTS

logger = logging.getLogger(__name__)


class ShapProxiedFS(ShapProxiedStateCompatMixin, ShapProxiedFitMixin, ShapProxiedMethodsMixin, TransformerMixin, BaseEstimator):
    """SHAP-coalition-proxy feature selector (sklearn transformer)."""

    def __init__(
        self,
        model: Any = None,
        classification: bool = True,
        metric: Optional[str] = None,
        optimizer: str = "auto",
        *,
        out_of_fold: bool = True,
        n_splits: int = 3,
        n_models: int = 1,
        min_features: int = 1,
        max_features: Optional[int] = None,
        top_n: int = 30,
        holdout_size: float = 0.25,
        report_holdout_fraction: float = 0.0,
        revalidate: bool = True,
        n_revalidation_models: int = 3,
        lambda_stab: float = 0.5,
        parsimony_tol: float = 0.02,
        min_selected_ratio: float = 0.0,
        trust_guard: bool = True,
        n_anchors: int | str = "auto",
        fidelity_floor: Optional[float] = None,
        spearman_floor: Optional[float] = None,
        run_importance_ablation: bool = True,
        use_bias_corrector: bool = True,
        active_learning: bool = False,
        active_learning_budget: int | None = None,
        config_jitter: bool = False,
        uncertainty_penalty: float = 0.0,
        interaction_aware: bool = False,
        max_interaction_features: int = 16,
        proxy_mode: str = "auto",
        interaction_proxy_top_k: int = 30,
        faith_n_coalitions: int = 2048,
        su_seeded_interactions: bool = False,
        su_seeded_top_k: int = 8,
        su_seeded_n_bins: int = 8,
        su_seeded_max_screen_cols: int = 120,
        su_seeded_snr_z: float = 3.0,
        su_seeded_snr_null_quantile: float = 0.99,
        su_seeded_snr_abs_floor: float = 1e-3,
        su_seeded_n_permutations: int = 3,
        residual_passes: int = 0,
        residual_merge: str = "rescue",
        residual_lambda: float = 1.0,
        residual_top_k: int | None = None,
        residual_exclude_top: int = 0,
        beam_width: int = 8,
        brute_force_max_features: int | None = None,
        adaptive_prescreen_by_stability: bool = False,
        prescreen_ladder_mode: str = "hardcoded",
        use_gpu: bool = False,
        prefilter_top: int | None = 2000,
        prefilter_method: str = "auto",
        prefilter_n_estimators: int | None = 100,
        oof_shap_n_estimators: int | None = 100,
        prefilter_stage1_keep: int | None = None,
        prefilter_univariate_batch_size: int | None = None,
        shap_prefilter_enabled: bool = True,
        shap_prefilter_top: int | None = None,
        shap_prefilter_safety_factor: int = 4,
        shap_prefilter_min_features: int = 40,
        shap_aware_stage1_keep: bool = True,
        shap_aware_stage1_cushion: int = 2,
        shap_aware_stage1_floor: int = 200,
        cluster_features: bool | str = "auto",
        cluster_corr_threshold: float = 0.7,
        cluster_weighting: str = "pca_pc1",
        cluster_use_gpu: bool | str = "auto",
        cluster_auto_threshold: int = 40,
        cluster_use_precomputed_bins: bool = True,
        cluster_su_threshold: float = 0.5,
        cluster_backend: str = "auto",
        cluster_su_auto_max_features: int | None = None,
        cluster_su_n_bins: int = 10,
        cluster_su_chance_correction: bool = True,
        prescreen_top: int | None = None,
        prescreen_ranking: str = "mean_abs_phi",
        banzhaf_n_coalitions: int = 4096,
        within_cluster_refine: bool = True,
        refine_n_estimators: int | None = 100,
        refine_mode: str = "auto",
        core_n_coalitions: int = 512,
        core_drop_threshold: float = 0.02,
        core_nucleolus: bool = False,
        refine_ucb_enabled: bool = True,
        refine_ucb_min_eval_size: int | None = None,
        refine_ucb_slack: float | None = None,
        refine_ucb_stdev_multiplier: float = 1.0,
        revalidation_n_estimators: int | None = 100,
        revalidation_ucb_enabled: bool = True,
        revalidation_ucb_min_eval_size: int | None = None,
        revalidation_ucb_slack: float | None = None,
        revalidation_ucb_stdev_multiplier: float | None = None,
        revalidation_adaptive_n_models: bool = True,
        revalidation_mmr_jaccard_threshold: float | None = None,
        trust_guard_n_estimators: int | None = 25,
        trust_guard_stratified_anchors: bool = False,
        trust_guard_uniform_tail_frac: float = 0.2,
        trust_guard_cardinality_dist: str = "zipf",
        trust_guard_zipf_alpha: float = 0.25,
        trust_guard_fidelity_weights: tuple[float, float] = DEFAULT_TRUST_GUARD_FIDELITY_WEIGHTS,
        trust_guard_metric: str = "proxy_fidelity_score",
        n_jobs: int = -1,
        inner_n_jobs_cap: bool = False,
        random_state: int = 0,
        verbose: bool = True,
        tqdm: bool = False,
        precomputed: dict | None = None,
        booster_kind: str | None = None,
        cat_features: list | None = None,
        cache_dir: str | None = None,
        max_runtime_mins: Optional[float] = None,
        stop_file: str = "stop",
    ):
        # Store all constructor arguments verbatim (design notes live in ``_shap_proxied_init_params``).
        params = dict(locals())
        del params["self"]
        assign_init_params(self, params)


__all__ = [
    "ShapProxiedFS",
    "restrict_artifacts",
    "ShapProxiedNoCandidatesError",
    "_resolve_brute_force_max_features",
    "_resolve_brute_force_n_sub_gate",
    "_resolve_cluster_su_auto_max_features",
    "_resolve_adaptive_prescreen_thresholds",
    "_resolve_adaptive_prescreen_width",
    "_resolve_adaptive_n_anchors",
    "_resolve_knee_prescreen_cap",
    "_DEFAULT_BRUTE_FORCE_MAX_FEATURES",
    "_DEFAULT_BRUTE_FORCE_N_SUB_GATE",
    "_DEFAULT_CLUSTER_SU_AUTO_MAX_FEATURES",
    "_DEFAULT_ADAPTIVE_PRESCREEN_THRESHOLDS",
    "_ADAPTIVE_PRESCREEN_FLOOR",
    "_EXACT_OPTIMIZERS",
    "_HEURISTIC_OPTIMIZERS",
    "pd",
]
