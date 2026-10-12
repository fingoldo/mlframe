"""Each transformer catalogue function is reachable through ``ShortlistTransformerAdapter`` (the opt-in switch).

The gate's ``_OPT_IN_SLOW`` set lists these names as valid-but-not-default paths. Dropping the export, or the constructor's
constructor lookup, makes the name unreachable again, so each name is checked against the constructor.
"""

import pytest

from mlframe.feature_engineering import transformer as pkg
from mlframe.feature_engineering.transformer._suite_adapter import ShortlistTransformerAdapter

_NAMES = [
    "compute_active_virtual_features",
    "compute_adaptive_bandwidth_attention",
    "compute_adasyn_smote_features",
    "compute_adversarial_flip_features",
    "compute_anchor_attention",
    "compute_anomaly_score_features",
    "compute_apriori_itemsets_features",
    "compute_autoencoder_features",
    "compute_aux_mlp_features",
    "compute_band_conditional_anchor_features",
    "compute_baseline_disagreement_balanced_features",
    "compute_baseline_disagreement_smote_features",
    "compute_baseline_disagreement_v2_features",
    "compute_baseline_surprise_features",
    "compute_bgm_clustered_smote_features",
    "compute_bgmm_density_ratio_features",
    "compute_bgmm_dual_class_features",
    "compute_bgmm_multiscale_features",
    "compute_bgmm_quantile_bands_features",
    "compute_bgmm_virtual_features",
    "compute_bidir_residual_band_features",
    "compute_boosted_attention",
    "compute_boosting_leaf_features",
    "compute_borderline_smote_features",
    "compute_class_balanced_hard_row_features",
    "compute_class_conditional_anchor_attention",
    "compute_class_distance_features",
    "compute_class_mahalanobis_features",
    "compute_cluster_smote_features",
    "compute_conformal_coverage_failure_features",
    "compute_conformal_locally_adaptive_features",
    "compute_counterfactual_substitution_features",
    "compute_cross_feature_reconstruction_features",
    "compute_cutmix_features",
    "compute_decision_region_depth_features",
    "compute_denoising_autoencoder_features",
    "compute_density_ratio_features",
    "compute_density_weighted_smote_features",
    "compute_diffusion_noise_features",
    "compute_disagreement_band_features",
    "compute_distributional_moments_features",
    "compute_fca_closed_concepts_features",
    "compute_fisher_weighted_residual_features",
    "compute_focal_lgb_features",
    "compute_geodesic_kgraph_features",
    "compute_gradient_direction_agreement_features",
    "compute_hard_row_attention_features",
    "compute_ib_baseline_codes_features",
    "compute_inducing_attention_features",
    "compute_jackknife_endpoint_stability_features",
    "compute_ks_shift_features",
    "compute_lda_projection_features",
    "compute_local_classifier_features",
    "compute_local_curvature_features",
    "compute_local_density_gradient_features",
    "compute_local_intrinsic_dim_features",
    "compute_local_lift_features",
    "compute_local_linear_attention",
    "compute_mdl_binning_pairwise_features",
    "compute_mixup_boundary_features",
    "compute_multi_baseline_hard_row_features",
    "compute_multi_temp_band_attention_features",
    "compute_multi_temp_cbhr_features",
    "compute_multi_temp_residual_band_features",
    "compute_multi_temperature_attention",
    "compute_multi_threshold_ordinal_features",
    "compute_multiscale_rate_features",
    "compute_multiscale_smote_features",
    "compute_nca_projection_features",
    "compute_neighbor_aggregate_features",
    "compute_pairwise_kl_features",
    "compute_per_class_spectral_attention",
    "compute_per_column_rff",
    "compute_performer_attention_features",
    "compute_persistence_diagram_features",
    "compute_positional_encoding",
    "compute_pred_augmented_attention",
    "compute_prediction_band_attention_features",
    "compute_predictive_info_delta_features",
    "compute_pseudo_smote_features",
    "compute_pure_pos_smote_features",
    "compute_quantile_band_attention_features",
    "compute_quantile_spread_fan_features",
    "compute_residual_attention",
    "compute_residual_band_attention_features",
    "compute_residual_stratified_distance_features",
    "compute_rf_proximity_attention",
    "compute_rff_features",
    "compute_robustness_budget_features",
    "compute_sign_residual_baseline_features",
    "compute_signed_residual_band_features",
    "compute_smote_distance_features",
    "compute_spectral_attention",
    "compute_stacked_quantile_neighbours",
    "compute_stacked_row_attention",
    "compute_target_kmeans_codebook_features",
    "compute_target_quantile_attention",
    "compute_tree_path_boolean_features",
    "compute_variance_baseline_features",
]


@pytest.mark.parametrize("name", _NAMES)
def test_catalogue_function_reachable_via_adapter(name):
    """The adapter built by name wraps the exported function itself."""
    adapter = ShortlistTransformerAdapter(name)
    assert isinstance(adapter, ShortlistTransformerAdapter)
    assert adapter.compute_fn is getattr(pkg, name)


def test_factory_rejects_non_catalogue_name():
    """Names outside the catalogue raise instead of silently building a wrapper."""
    with pytest.raises(KeyError):
        ShortlistTransformerAdapter("not_a_transformer")
