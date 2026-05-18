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

from .active_virtual import compute_active_virtual_features
from .adaptive_bandwidth import compute_adaptive_bandwidth_attention
from .adversarial_flip import compute_adversarial_flip_features
from .adasyn_smote import compute_adasyn_smote_features
from .anchor_attention import compute_anchor_attention
from .aux_mlp import compute_aux_mlp_features
from .autoencoder import compute_autoencoder_features
from .band_conditional_anchor import compute_band_conditional_anchor_features
from .baseline_disagreement import compute_baseline_disagreement_features
from .baseline_disagreement_v2 import compute_baseline_disagreement_v2_features
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
    compute_positional_encoding,
    compute_rff_features,
    positions_within_group,
)
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

__all__ = [
    "attend",
    "build_key_bank",
    "compute_active_virtual_features",
    "compute_adaptive_bandwidth_attention",
    "compute_adasyn_smote_features",
    "compute_adversarial_flip_features",
    "compute_anchor_attention",
    "compute_autoencoder_features",
    "compute_aux_mlp_features",
    "compute_band_conditional_anchor_features",
    "compute_baseline_disagreement_features",
    "compute_baseline_disagreement_v2_features",
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
]
