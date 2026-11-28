"""Transformer-style frozen feature engineering for tabular models (trees + linear).

Three blocks, all single-pass / stateless / reproducible-from-seed:

- ``compute_rff_features`` - Random Fourier Features (Rahimi-Recht 2007); approximates
  RBF-kernel feature maps for linear models, augments trees with smooth nonlinearities.
- ``compute_positional_encoding`` + ``positions_within_group`` - sinusoidal PE for
  within-group ordinal positions (time, session index, ticker-day order).
- ``compute_row_attention`` - multi-head softmax-weighted kNN-target-encoding over
  random-subspace projections. Strict OOF discipline on train, train-only key-bank
  for val / test / OOS / holdout. Mandatory random projection front-end at d > 256.

GPU acceleration (cupy) on the two compute-heavy bandwidth-bound stages only:
RFF matmul + row-attention stage-4 fused kernel. Other stages (projection in Mode A,
hnswlib ANN, standardisation, PE) are CPU-only by design - GPU loses on H2D overhead
at d=10k-20k, or has no GPU library on Windows (cuVS is Linux/WSL2 only).

Honest framing: without backprop, "attention" here is mathematically (a) random-projection
nonlinear feature maps (RFF), (b) softmax-weighted kNN-target-encoding (row-attn). These
are useful techniques for trees/linear, but the "transformer" name is structural, not
algorithmic - no learnable attention weights are involved.

Public APIs are wired in :mod:`mlframe.feature_engineering` so users can import directly:
``from mlframe.feature_engineering import compute_rff_features``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # static analysers and IDEs see the names; at runtime they resolve lazily through ``__getattr__`` below
    from .active_virtual import compute_active_virtual_features
    from .adaptive_bandwidth import compute_adaptive_bandwidth_attention
    from .adversarial_flip import compute_adversarial_flip_features
    from .adasyn_smote import compute_adasyn_smote_features
    from .anchor_attention import compute_anchor_attention
    from .aux_mlp import compute_aux_mlp_features
    from .autoencoder import compute_autoencoder_features
    from .denoising_autoencoder import compute_denoising_autoencoder_features
    from .swap_noise import swap_noise_augment
    from .band_conditional_anchor import compute_band_conditional_anchor_features
    from .baseline_disagreement import compute_baseline_disagreement_features
    from .baseline_disagreement_v2 import compute_baseline_disagreement_v2_features
    from .baseline_disagreement_balanced import compute_baseline_disagreement_balanced_features
    from .baseline_disagreement_smote import compute_baseline_disagreement_smote_features
    from .anomaly_score_features import compute_anomaly_score_features
    from .residual_stratified_distance import compute_residual_stratified_distance_features
    from .y_quintile_baseline_knn import compute_y_quintile_baseline_knn_features
    from .baseline_surprise import compute_baseline_surprise_features
    from .bgm_clustered_smote import compute_bgm_clustered_smote_features
    from .bidir_residual_band import compute_bidir_residual_band_features
    from .bgmm_density_ratio import compute_bgmm_density_ratio_features
    from .bgmm_dual_class import compute_bgmm_dual_class_features
    from .bgmm_multiscale import compute_bgmm_multiscale_features
    from .bgmm_quantile_bands import compute_bgmm_quantile_bands_features
    from .bgmm_virtual import compute_bgmm_virtual_features
    from .boosted_attention import compute_boosted_attention
    from .boosting_leaf import compute_boosting_leaf_features
    from .borderline_smote import compute_borderline_smote_features
    from .class_balanced_hard_row import compute_class_balanced_hard_row_features
    from .class_conditional_anchor import compute_class_conditional_anchor_attention
    from .class_distance import compute_class_distance_features
    from .class_mahalanobis import compute_class_mahalanobis_features
    from .cluster_smote import compute_cluster_smote_features
    from .counterfactual_substitution import compute_counterfactual_substitution_features
    from .cutmix import compute_cutmix_features
    from .decision_region_depth import compute_decision_region_depth_features
    from .density_ratio import compute_density_ratio_features
    from .density_weighted_smote import compute_density_weighted_smote_features
    from .diffusion_noise import compute_diffusion_noise_features
    from .disagreement_band import compute_disagreement_band_features
    from .fisher_weighted_residual import compute_fisher_weighted_residual_features
    from .focal_lgb import compute_focal_lgb_features
    from .geodesic_kgraph import compute_geodesic_kgraph_features
    from .gradient_direction_agreement import compute_gradient_direction_agreement_features
    from .hard_row_attention import compute_hard_row_attention_features
    from .ib_baseline_codes import compute_ib_baseline_codes_features
    from .inducing_attention import compute_inducing_attention_features
    from .ks_shift import compute_ks_shift_features
    from .lda_projection import compute_lda_projection_features
    from .local_classifier import compute_local_classifier_features
    from .local_curvature import compute_local_curvature_features
    from .local_density_gradient import compute_local_density_gradient_features
    from .local_intrinsic_dim import compute_local_intrinsic_dim_features
    from .local_lift import compute_local_lift_features
    from .mixup_boundary import compute_mixup_boundary_features
    from .local_linear import compute_local_linear_attention
    from .multi_aux_ensemble import compute_multi_aux_features
    from .multi_baseline_hard_row import compute_multi_baseline_hard_row_features
    from .multi_temp_band_attention import compute_multi_temp_band_attention_features
    from .multi_temp_cbhr import compute_multi_temp_cbhr_features
    from .multi_temp_residual_band import compute_multi_temp_residual_band_features
    from .multi_temperature import compute_multi_temperature_attention
    from .nca_projection import compute_nca_projection_features
    from .nn_oof_target_mean import compute_nn_oof_target_mean_features
    from .neighbor_aggregate_features import compute_neighbor_aggregate_features
    from .multiscale_rate import compute_multiscale_rate_features
    from .multiscale_smote import compute_multiscale_smote_features
    from .per_class_spectral import compute_per_class_spectral_attention
    from .persistence_diagram import compute_persistence_diagram_features
    from .per_column_rff import compute_per_column_rff
    from .performer_attention import compute_performer_attention_features
    from .pairwise_kl_divergence import compute_pairwise_kl_features
    from .pred_augmented import compute_pred_augmented_attention
    from .predictive_info_delta import compute_predictive_info_delta_features
    from .prediction_band_attention import compute_prediction_band_attention_features
    from .pseudo_smote import compute_pseudo_smote_features
    from .pure_pos_smote import compute_pure_pos_smote_features
    from .quantile_band_attention import compute_quantile_band_attention_features
    from .quantile_neighbours import compute_quantile_neighbours
    from .stacked_qnn import compute_stacked_quantile_neighbours
    from .random_features import (
        RFFState,
        compute_positional_encoding,
        compute_rff_features,
        positions_within_group,
        rff_apply_state,
    )
    from ._suite_adapter import ShortlistTransformerAdapter
    from .residual_attention import compute_residual_attention
    from .residual_band_attention import compute_residual_band_attention_features
    from .rf_proximity import compute_rf_proximity_attention
    from .robustness_budget import compute_robustness_budget_features
    from .signed_residual_band import compute_signed_residual_band_features
    from .smote_distance import compute_smote_distance_features
    from .row_attention import attend, build_key_bank, compute_row_attention
    from .spectral_attention import compute_spectral_attention
    from .stacked_attention import compute_stacked_row_attention
    from .apriori_itemsets import compute_apriori_itemsets_features
    from .conformal_coverage_failure import compute_conformal_coverage_failure_features
    from .conformal_locally_adaptive import compute_conformal_locally_adaptive_features
    from .distributional_moments import compute_distributional_moments_features
    from .fca_closed_concepts import compute_fca_closed_concepts_features
    from .jackknife_endpoint_stability import compute_jackknife_endpoint_stability_features
    from .mdl_binning_pairwise import compute_mdl_binning_pairwise_features
    from .cross_feature_reconstruction import compute_cross_feature_reconstruction_features
    from .multi_threshold_ordinal import compute_multi_threshold_ordinal_features
    from .target_kmeans_codebook import compute_target_kmeans_codebook_features
    from .tree_path_boolean import compute_tree_path_boolean_features
    from .quantile_spread_fan import compute_quantile_spread_fan_features
    from .sign_residual_baseline import compute_sign_residual_baseline_features
    from .target_quantile import compute_target_quantile_attention
    from .trust_score_oof import compute_trust_score_oof_features
    from .variance_baseline import compute_variance_baseline_features
    from ._knn_helper import knn_search
    from ._utils import is_gpu_available

__all__ = [
    "RFFState",
    "ShortlistTransformerAdapter",
    "attend",
    "build_key_bank",
    "compute_active_virtual_features",
    "compute_adaptive_bandwidth_attention",
    "compute_adasyn_smote_features",
    "compute_adversarial_flip_features",
    "compute_anchor_attention",
    "compute_autoencoder_features",
    "compute_denoising_autoencoder_features",
    "swap_noise_augment",
    "compute_aux_mlp_features",
    "compute_band_conditional_anchor_features",
    "compute_baseline_disagreement_features",
    "compute_baseline_disagreement_v2_features",
    "compute_baseline_disagreement_balanced_features",
    "compute_baseline_disagreement_smote_features",
    "compute_anomaly_score_features",
    "compute_residual_stratified_distance_features",
    "compute_y_quintile_baseline_knn_features",
    "compute_baseline_surprise_features",
    "compute_bgm_clustered_smote_features",
    "compute_bidir_residual_band_features",
    "compute_bgmm_density_ratio_features",
    "compute_bgmm_dual_class_features",
    "compute_bgmm_multiscale_features",
    "compute_bgmm_quantile_bands_features",
    "compute_bgmm_virtual_features",
    "compute_boosted_attention",
    "compute_boosting_leaf_features",
    "compute_borderline_smote_features",
    "compute_class_balanced_hard_row_features",
    "compute_class_conditional_anchor_attention",
    "compute_class_distance_features",
    "compute_class_mahalanobis_features",
    "compute_cluster_smote_features",
    "compute_counterfactual_substitution_features",
    "compute_cutmix_features",
    "compute_decision_region_depth_features",
    "compute_density_ratio_features",
    "compute_density_weighted_smote_features",
    "compute_diffusion_noise_features",
    "compute_disagreement_band_features",
    "compute_fisher_weighted_residual_features",
    "compute_focal_lgb_features",
    "compute_geodesic_kgraph_features",
    "compute_gradient_direction_agreement_features",
    "compute_hard_row_attention_features",
    "compute_ib_baseline_codes_features",
    "compute_inducing_attention_features",
    "compute_ks_shift_features",
    "compute_lda_projection_features",
    "compute_local_classifier_features",
    "compute_local_curvature_features",
    "compute_local_density_gradient_features",
    "compute_local_intrinsic_dim_features",
    "compute_local_lift_features",
    "compute_mixup_boundary_features",
    "compute_local_linear_attention",
    "compute_multi_aux_features",
    "compute_multi_baseline_hard_row_features",
    "compute_multi_temp_band_attention_features",
    "compute_multi_temp_cbhr_features",
    "compute_multi_temp_residual_band_features",
    "compute_multi_temperature_attention",
    "compute_multiscale_rate_features",
    "compute_multiscale_smote_features",
    "compute_nca_projection_features",
    "compute_nn_oof_target_mean_features",
    "compute_neighbor_aggregate_features",
    "compute_per_class_spectral_attention",
    "compute_per_column_rff",
    "compute_persistence_diagram_features",
    "compute_performer_attention_features",
    "compute_positional_encoding",
    "compute_pairwise_kl_features",
    "compute_pred_augmented_attention",
    "compute_predictive_info_delta_features",
    "compute_prediction_band_attention_features",
    "compute_pseudo_smote_features",
    "compute_pure_pos_smote_features",
    "compute_quantile_band_attention_features",
    "compute_quantile_neighbours",
    "compute_quantile_spread_fan_features",
    "compute_apriori_itemsets_features",
    "compute_conformal_coverage_failure_features",
    "compute_conformal_locally_adaptive_features",
    "compute_distributional_moments_features",
    "compute_fca_closed_concepts_features",
    "compute_jackknife_endpoint_stability_features",
    "compute_mdl_binning_pairwise_features",
    "compute_cross_feature_reconstruction_features",
    "compute_multi_threshold_ordinal_features",
    "compute_target_kmeans_codebook_features",
    "compute_tree_path_boolean_features",
    "compute_sign_residual_baseline_features",
    "compute_trust_score_oof_features",
    "compute_variance_baseline_features",
    "compute_residual_attention",
    "compute_residual_band_attention_features",
    "compute_rf_proximity_attention",
    "compute_rff_features",
    "compute_robustness_budget_features",
    "compute_signed_residual_band_features",
    "compute_smote_distance_features",
    "compute_row_attention",
    "compute_spectral_attention",
    "compute_stacked_quantile_neighbours",
    "compute_stacked_row_attention",
    "compute_target_quantile_attention",
    "positions_within_group",
    "rff_apply_state",
    "is_gpu_available",
    "knn_search",
]

# Importing this package no longer imports its ~130 transformer modules (and their numba / sklearn / torch dependencies). Each public name maps to the
# module that defines it and is imported on first attribute access (PEP 562), so ``from mlframe.feature_engineering.transformer import X`` and
# ``transformer.X`` keep working while ``import mlframe`` pays only for the transformers it uses (08-F16).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "compute_active_virtual_features": (".active_virtual", "compute_active_virtual_features"),
    "compute_adaptive_bandwidth_attention": (".adaptive_bandwidth", "compute_adaptive_bandwidth_attention"),
    "compute_adversarial_flip_features": (".adversarial_flip", "compute_adversarial_flip_features"),
    "compute_adasyn_smote_features": (".adasyn_smote", "compute_adasyn_smote_features"),
    "compute_anchor_attention": (".anchor_attention", "compute_anchor_attention"),
    "compute_aux_mlp_features": (".aux_mlp", "compute_aux_mlp_features"),
    "compute_autoencoder_features": (".autoencoder", "compute_autoencoder_features"),
    "compute_denoising_autoencoder_features": (".denoising_autoencoder", "compute_denoising_autoencoder_features"),
    "swap_noise_augment": (".swap_noise", "swap_noise_augment"),
    "compute_band_conditional_anchor_features": (".band_conditional_anchor", "compute_band_conditional_anchor_features"),
    "compute_baseline_disagreement_features": (".baseline_disagreement", "compute_baseline_disagreement_features"),
    "compute_baseline_disagreement_v2_features": (".baseline_disagreement_v2", "compute_baseline_disagreement_v2_features"),
    "compute_baseline_disagreement_balanced_features": (".baseline_disagreement_balanced", "compute_baseline_disagreement_balanced_features"),
    "compute_baseline_disagreement_smote_features": (".baseline_disagreement_smote", "compute_baseline_disagreement_smote_features"),
    "compute_anomaly_score_features": (".anomaly_score_features", "compute_anomaly_score_features"),
    "compute_residual_stratified_distance_features": (".residual_stratified_distance", "compute_residual_stratified_distance_features"),
    "compute_y_quintile_baseline_knn_features": (".y_quintile_baseline_knn", "compute_y_quintile_baseline_knn_features"),
    "compute_baseline_surprise_features": (".baseline_surprise", "compute_baseline_surprise_features"),
    "compute_bgm_clustered_smote_features": (".bgm_clustered_smote", "compute_bgm_clustered_smote_features"),
    "compute_bidir_residual_band_features": (".bidir_residual_band", "compute_bidir_residual_band_features"),
    "compute_bgmm_density_ratio_features": (".bgmm_density_ratio", "compute_bgmm_density_ratio_features"),
    "compute_bgmm_dual_class_features": (".bgmm_dual_class", "compute_bgmm_dual_class_features"),
    "compute_bgmm_multiscale_features": (".bgmm_multiscale", "compute_bgmm_multiscale_features"),
    "compute_bgmm_quantile_bands_features": (".bgmm_quantile_bands", "compute_bgmm_quantile_bands_features"),
    "compute_bgmm_virtual_features": (".bgmm_virtual", "compute_bgmm_virtual_features"),
    "compute_boosted_attention": (".boosted_attention", "compute_boosted_attention"),
    "compute_boosting_leaf_features": (".boosting_leaf", "compute_boosting_leaf_features"),
    "compute_borderline_smote_features": (".borderline_smote", "compute_borderline_smote_features"),
    "compute_class_balanced_hard_row_features": (".class_balanced_hard_row", "compute_class_balanced_hard_row_features"),
    "compute_class_conditional_anchor_attention": (".class_conditional_anchor", "compute_class_conditional_anchor_attention"),
    "compute_class_distance_features": (".class_distance", "compute_class_distance_features"),
    "compute_class_mahalanobis_features": (".class_mahalanobis", "compute_class_mahalanobis_features"),
    "compute_cluster_smote_features": (".cluster_smote", "compute_cluster_smote_features"),
    "compute_counterfactual_substitution_features": (".counterfactual_substitution", "compute_counterfactual_substitution_features"),
    "compute_cutmix_features": (".cutmix", "compute_cutmix_features"),
    "compute_decision_region_depth_features": (".decision_region_depth", "compute_decision_region_depth_features"),
    "compute_density_ratio_features": (".density_ratio", "compute_density_ratio_features"),
    "compute_density_weighted_smote_features": (".density_weighted_smote", "compute_density_weighted_smote_features"),
    "compute_diffusion_noise_features": (".diffusion_noise", "compute_diffusion_noise_features"),
    "compute_disagreement_band_features": (".disagreement_band", "compute_disagreement_band_features"),
    "compute_fisher_weighted_residual_features": (".fisher_weighted_residual", "compute_fisher_weighted_residual_features"),
    "compute_focal_lgb_features": (".focal_lgb", "compute_focal_lgb_features"),
    "compute_geodesic_kgraph_features": (".geodesic_kgraph", "compute_geodesic_kgraph_features"),
    "compute_gradient_direction_agreement_features": (".gradient_direction_agreement", "compute_gradient_direction_agreement_features"),
    "compute_hard_row_attention_features": (".hard_row_attention", "compute_hard_row_attention_features"),
    "compute_ib_baseline_codes_features": (".ib_baseline_codes", "compute_ib_baseline_codes_features"),
    "compute_inducing_attention_features": (".inducing_attention", "compute_inducing_attention_features"),
    "compute_ks_shift_features": (".ks_shift", "compute_ks_shift_features"),
    "compute_lda_projection_features": (".lda_projection", "compute_lda_projection_features"),
    "compute_local_classifier_features": (".local_classifier", "compute_local_classifier_features"),
    "compute_local_curvature_features": (".local_curvature", "compute_local_curvature_features"),
    "compute_local_density_gradient_features": (".local_density_gradient", "compute_local_density_gradient_features"),
    "compute_local_intrinsic_dim_features": (".local_intrinsic_dim", "compute_local_intrinsic_dim_features"),
    "compute_local_lift_features": (".local_lift", "compute_local_lift_features"),
    "compute_mixup_boundary_features": (".mixup_boundary", "compute_mixup_boundary_features"),
    "compute_local_linear_attention": (".local_linear", "compute_local_linear_attention"),
    "compute_multi_aux_features": (".multi_aux_ensemble", "compute_multi_aux_features"),
    "compute_multi_baseline_hard_row_features": (".multi_baseline_hard_row", "compute_multi_baseline_hard_row_features"),
    "compute_multi_temp_band_attention_features": (".multi_temp_band_attention", "compute_multi_temp_band_attention_features"),
    "compute_multi_temp_cbhr_features": (".multi_temp_cbhr", "compute_multi_temp_cbhr_features"),
    "compute_multi_temp_residual_band_features": (".multi_temp_residual_band", "compute_multi_temp_residual_band_features"),
    "compute_multi_temperature_attention": (".multi_temperature", "compute_multi_temperature_attention"),
    "compute_nca_projection_features": (".nca_projection", "compute_nca_projection_features"),
    "compute_nn_oof_target_mean_features": (".nn_oof_target_mean", "compute_nn_oof_target_mean_features"),
    "compute_neighbor_aggregate_features": (".neighbor_aggregate_features", "compute_neighbor_aggregate_features"),
    "compute_multiscale_rate_features": (".multiscale_rate", "compute_multiscale_rate_features"),
    "compute_multiscale_smote_features": (".multiscale_smote", "compute_multiscale_smote_features"),
    "compute_per_class_spectral_attention": (".per_class_spectral", "compute_per_class_spectral_attention"),
    "compute_persistence_diagram_features": (".persistence_diagram", "compute_persistence_diagram_features"),
    "compute_per_column_rff": (".per_column_rff", "compute_per_column_rff"),
    "compute_performer_attention_features": (".performer_attention", "compute_performer_attention_features"),
    "compute_pairwise_kl_features": (".pairwise_kl_divergence", "compute_pairwise_kl_features"),
    "compute_pred_augmented_attention": (".pred_augmented", "compute_pred_augmented_attention"),
    "compute_predictive_info_delta_features": (".predictive_info_delta", "compute_predictive_info_delta_features"),
    "compute_prediction_band_attention_features": (".prediction_band_attention", "compute_prediction_band_attention_features"),
    "compute_pseudo_smote_features": (".pseudo_smote", "compute_pseudo_smote_features"),
    "compute_pure_pos_smote_features": (".pure_pos_smote", "compute_pure_pos_smote_features"),
    "compute_quantile_band_attention_features": (".quantile_band_attention", "compute_quantile_band_attention_features"),
    "compute_quantile_neighbours": (".quantile_neighbours", "compute_quantile_neighbours"),
    "compute_stacked_quantile_neighbours": (".stacked_qnn", "compute_stacked_quantile_neighbours"),
    "RFFState": (".random_features", "RFFState"),
    "compute_positional_encoding": (".random_features", "compute_positional_encoding"),
    "compute_rff_features": (".random_features", "compute_rff_features"),
    "positions_within_group": (".random_features", "positions_within_group"),
    "rff_apply_state": (".random_features", "rff_apply_state"),
    "ShortlistTransformerAdapter": ("._suite_adapter", "ShortlistTransformerAdapter"),
    "compute_residual_attention": (".residual_attention", "compute_residual_attention"),
    "compute_residual_band_attention_features": (".residual_band_attention", "compute_residual_band_attention_features"),
    "compute_rf_proximity_attention": (".rf_proximity", "compute_rf_proximity_attention"),
    "compute_robustness_budget_features": (".robustness_budget", "compute_robustness_budget_features"),
    "compute_signed_residual_band_features": (".signed_residual_band", "compute_signed_residual_band_features"),
    "compute_smote_distance_features": (".smote_distance", "compute_smote_distance_features"),
    "attend": (".row_attention", "attend"),
    "build_key_bank": (".row_attention", "build_key_bank"),
    "compute_row_attention": (".row_attention", "compute_row_attention"),
    "compute_spectral_attention": (".spectral_attention", "compute_spectral_attention"),
    "compute_stacked_row_attention": (".stacked_attention", "compute_stacked_row_attention"),
    "compute_apriori_itemsets_features": (".apriori_itemsets", "compute_apriori_itemsets_features"),
    "compute_conformal_coverage_failure_features": (".conformal_coverage_failure", "compute_conformal_coverage_failure_features"),
    "compute_conformal_locally_adaptive_features": (".conformal_locally_adaptive", "compute_conformal_locally_adaptive_features"),
    "compute_distributional_moments_features": (".distributional_moments", "compute_distributional_moments_features"),
    "compute_fca_closed_concepts_features": (".fca_closed_concepts", "compute_fca_closed_concepts_features"),
    "compute_jackknife_endpoint_stability_features": (".jackknife_endpoint_stability", "compute_jackknife_endpoint_stability_features"),
    "compute_mdl_binning_pairwise_features": (".mdl_binning_pairwise", "compute_mdl_binning_pairwise_features"),
    "compute_cross_feature_reconstruction_features": (".cross_feature_reconstruction", "compute_cross_feature_reconstruction_features"),
    "compute_multi_threshold_ordinal_features": (".multi_threshold_ordinal", "compute_multi_threshold_ordinal_features"),
    "compute_target_kmeans_codebook_features": (".target_kmeans_codebook", "compute_target_kmeans_codebook_features"),
    "compute_tree_path_boolean_features": (".tree_path_boolean", "compute_tree_path_boolean_features"),
    "compute_quantile_spread_fan_features": (".quantile_spread_fan", "compute_quantile_spread_fan_features"),
    "compute_sign_residual_baseline_features": (".sign_residual_baseline", "compute_sign_residual_baseline_features"),
    "compute_target_quantile_attention": (".target_quantile", "compute_target_quantile_attention"),
    "compute_trust_score_oof_features": (".trust_score_oof", "compute_trust_score_oof_features"),
    "compute_variance_baseline_features": (".variance_baseline", "compute_variance_baseline_features"),
    "knn_search": ("._knn_helper", "knn_search"),
    "is_gpu_available": ("._utils", "is_gpu_available"),
}


def __getattr__(name: str) -> Any:
    """Import the module that defines ``name`` on first access and cache the attribute on the package."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise the lazily resolved public names alongside the module's own."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
