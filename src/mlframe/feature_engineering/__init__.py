"""Feature engineering helpers for tabular and time-series ML.

Submodules:
    anchor             - per-group anchor-based linear extrapolation features (sparse-truth case).
    basic              - date-part decomposition, symbolic regression entry-point.
    bayesian           - particle filter posterior features on 1-D state.
    bruteforce         - PySR-driven brute-force feature search.
    categorical        - aggregates for categorical / repeated-value series.
    ensemble_features  - per-row disagreement features across N stacked predictors.
    financial          - OHLCV ratios, lags, TA-Lib indicators, market-wide features.
    fuzzy_features     - fuzzification: encode a numeric column as fuzzy-partition (POSP) soft-membership features.
    graph_spectral_features - permutation-invariant per-graph spectral descriptor (for graph-classification tabular tasks).
    grouped            - per-group sliding-window iterator primitives.
    hurst              - Hurst exponent (R/S analysis) + rolling DFA / Higuchi FD.
    mps                - Maximum Profit System target/regions.
    numerical          - rich numerical aggregates for 1d vectors.
    recency_aggregation - per-entity recency-weighted mean/event-rate (Dyakonov weighted-scheme estimator).
    recency_density    - per-entity recency-weighted Parzen density: mode prediction + behavioral-stability score.
    nadaraya_watson    - Nadaraya-Watson kernel regression / smoothing (flat + per-entity), sample-weight composable.
    spatial            - XY / N-D Euclidean kNN aggregator features (geo / point-cloud).
    spectral           - FFT-based rolling band energies, spectral entropy, dominant freq.
    stationarity       - frac_diff (Lopez de Prado fractional differencing) + future ADF/KPSS.
    timeseries         - windowed feature aggregation and ACF helpers.
    transformer        - frozen transformer-style FE: RFF, positional encoding, row-attention (multi-head softmax-weighted kNN-TE).
    wavelet_dwt        - discrete + continuous wavelet transforms + denoising.
    windowed_shape     - rolling shape features (mean_abs_d2, n_peaks, n_troughs, extrema density, integral_above_baseline).
"""

from __future__ import annotations


from .anchor import (
    add_anchor_extrapolation_features,
    anchor_density_features,
    anchor_ewm_features,
    anchor_quadratic_extrapolation_features,
    anchor_residual_rmse_features,
    rows_until_next_anchor,
)
from .basic import create_date_features, run_pysr_fe
from .bayesian import (
    bocpd_features,
    kalman_filter_posterior_1d,
    kalman_smoother_posterior_1d,
    online_bayesian_linear_regression,
    particle_filter_posterior,
)
from .bruteforce import run_pysr_feature_engineering
from .categorical import compute_countaggs, get_countaggs_names
from .ensemble_features import (
    predictor_consensus_entropy,
    predictor_consensus_mean,
    predictor_consensus_trimmed_stats,
    predictor_disagreement_features,
    predictor_disagreement_iqr,
    predictor_disagreement_var,
    predictor_max_pairwise_distance,
    predictor_outlier_signature,
    predictor_pairwise_abs_diffs,
    predictor_quantile_spread,
    predictor_top2_mode_gap,
    predictor_weighted_consensus,
)
from .financial import (
    add_fast_rolling_stats,
    add_ohlcv_ratios_rlags,
    add_ohlcv_ta_indicators,
    create_ohlcv_wholemarket_features,
    merge_perticker_and_wholemarket_features,
)
from .grouped import (
    iter_group_segments,
    per_group_apply,
    per_group_cum_reduce,
    per_group_nth,
    per_group_rank,
    per_group_rolling_reduce,
    per_group_shift,
    per_group_sliding_window,
)
from .recency_aggregation import per_group_recency_weighted_agg, per_group_recency_weighted_mean
from .nested_ma_decompose import nested_ma_decompose
from .ma_crossover import ma_crossover_features
from .holiday_calendar_features import holiday_calendar_features
from .state_history import last_k_distinct_states_with_durations
from .panel_pivot import pivot_time_indexed_panel
from .ewma_multi_alpha_features import ewma_multi_alpha_features
from .row_wise_extremality import row_wise_extremality_index, row_wise_top_k_extreme_columns
from .recency_density import (
    per_group_behavioral_stability,
    per_group_recency_weighted_mode,
)
from .nadaraya_watson import (
    nadaraya_watson_smooth,
    per_group_nadaraya_watson_smooth,
)
from .hurst import (
    compute_hurst_exponent,
    compute_hurst_rs,
    dfa_alpha,
    dfa_alpha2_quadratic,
    higuchi_fd,
    multi_scale_hurst,
    multifractal_dfa,
    precompute_hurst_exponent,
    rolling_dfa_alpha,
    rolling_higuchi_fd,
    rolling_hurst,
)
from .spatial import (
    inverse_distance_weighted_aggregate,
    knn_aggregate,
    knn_gradient_features,
    knn_label_dispersion_features,
    knn_within_bucket_aggregate,
    local_density_features,
    radius_aggregate,
)
from .spectral import (
    rolling_dominant_freq_idx,
    rolling_hf_lf_ratio,
    rolling_periodicity_score,
    rolling_spectral_band_energies,
    rolling_spectral_bandwidth,
    rolling_spectral_centroid,
    rolling_spectral_entropy,
    rolling_spectral_flatness,
    rolling_spectral_flux,
    rolling_spectral_rolloff,
)
from .stationarity import (
    cusum_features,
    ewma_residual,
    frac_diff,
    frac_diff_weights,
    local_linear_detrend,
    quantile_normalize_per_group,
)
from .windowed_shape import (
    rolling_extrema_density,
    rolling_integral_above_baseline,
    rolling_longest_monotone_run,
    rolling_mean_abs_d2,
    rolling_n_peaks,
    rolling_n_troughs,
    rolling_quantile_spread,
    rolling_shannon_entropy_binned,
    rolling_total_variation,
    rolling_zero_crossings,
)
from .mps import (
    compute_mps_targets,
    find_best_mps_sequence,
    find_maximum_profit_system,
    safely_compute_mps,
    show_mps_regions,
)
from .numerical import (
    compute_numaggs,
    compute_numaggs_parallel,
    compute_simple_stats_numba,
    cont_entropy,
    get_basic_feature_names,
    get_moments_slope_mi_feature_names,
    get_numaggs_names,
    get_simple_stats_names,
    numaggs_over_matrix_rows,
    rolling_moving_average,
)
from .timeseries import create_aggregated_features, create_windowed_features, general_acf
from .transformer import (
    compute_positional_encoding,
    compute_rff_features,
    compute_row_attention,
    positions_within_group,
)
from .cat_cooccurrence_svd import (
    engineered_name_cooccur_svd,
    cat_cooccurrence_svd_fit,
    apply_cat_cooccurrence_svd,
    cat_cooccurrence_svd_with_recipes,
)
from .graph_features import (
    graph_neighbor_aggregate,
    graph_structural_features,
    link_prediction_features,
)
from .graph_construction import (
    knn_graph_edges,
    shared_attribute_edges,
)
from .fuzzy_features import (
    fuzzy_partition_encode,
    fuzzy_partition_fit,
    fuzzy_partition_transform,
    fuzzy_partition_names,
)
from .graph_spectral_features import (
    graph_spectral_features,
    graph_spectral_feature_names,
)
from .drift_remediation import remediate_drifting_features
from .entity_inter_event import entity_inter_event_features
from .fuzzy_entity import fuzzy_entity_group_features
from .state_duration import time_since_state_change
from .nearest_past_join import nearest_past_join
from .as_of_aggregate import leakage_safe_aggregate
from .random_lag_augmentation import randomize_as_of_lag
from .row_wise_summary import row_wise_summary_stats
from .rolling_target_correlation import rolling_target_correlation_tracker
from .auxiliary_feature_prediction import compute_auxiliary_feature_prediction_features
from .cross_sectional_neighbors import compute_cross_sectional_neighbor_features
from .latent_interaction_svd import latent_interaction_features
from .multi_window_aggregate import multi_window_aggregate
from .entity_diff_features import entity_diff_features
from .control_difference_augment import control_difference_augment
from .acf_lag_selection import select_significant_lags
from .two_step_target_encode import two_step_recency_weighted_target_encode
from .polars_dynamic_window import polars_dynamic_window_aggregate
from .sequence2vec_categorical import train_sequence2vec, sequence2vec_entity_features
from .relational_dfs import ChildTableSpec, compute_relational_features, stack_relational_features
from .windowed_edge_diff import windowed_edge_aggregate_diff
from .magnitude_sample_weight import magnitude_sample_weight
from .boolean_pair_interactions import boolean_pair_interactions, is_binary_column
from .sentinel_missing_count import add_sentinel_missing_count_feature
from .categorical_group_concat import concat_categorical_group
from .binned_unique_count import binned_unique_count
from .multi_decomposition_bank import multi_decomposition_feature_bank
from .variance_gated_pairwise_diff import variance_gated_pairwise_diff
from .event_proximity_decay import event_proximity_decay_features
from .tfidf_svd_entity_embedding import tfidf_svd_entity_embedding
from .gmm_bic_membership_features import gmm_bic_membership_features
from .latent_parameter_recovery import latent_parameter_recovery_features
from .panel_sequence_tensor import build_panel_sequence_tensor

__all__ = [
    "per_group_recency_weighted_mean",
    "per_group_recency_weighted_agg",
    "nested_ma_decompose",
    "ma_crossover_features",
    "holiday_calendar_features",
    "last_k_distinct_states_with_durations",
    "pivot_time_indexed_panel",
    "ewma_multi_alpha_features",
    "row_wise_extremality_index",
    "row_wise_top_k_extreme_columns",
    "per_group_recency_weighted_mode",
    "per_group_behavioral_stability",
    "nadaraya_watson_smooth",
    "per_group_nadaraya_watson_smooth",
    "add_anchor_extrapolation_features",
    "add_fast_rolling_stats",
    "add_ohlcv_ratios_rlags",
    "add_ohlcv_ta_indicators",
    "compute_countaggs",
    "compute_hurst_exponent",
    "compute_hurst_rs",
    "compute_mps_targets",
    "compute_numaggs",
    "compute_numaggs_parallel",
    "compute_positional_encoding",
    "compute_rff_features",
    "compute_row_attention",
    "compute_simple_stats_numba",
    "cont_entropy",
    "create_aggregated_features",
    "create_date_features",
    "create_ohlcv_wholemarket_features",
    "create_windowed_features",
    "dfa_alpha",
    "find_best_mps_sequence",
    "find_maximum_profit_system",
    "frac_diff",
    "frac_diff_weights",
    "general_acf",
    "get_basic_feature_names",
    "get_countaggs_names",
    "get_moments_slope_mi_feature_names",
    "get_numaggs_names",
    "get_simple_stats_names",
    "higuchi_fd",
    "iter_group_segments",
    "knn_aggregate",
    "knn_within_bucket_aggregate",
    "merge_perticker_and_wholemarket_features",
    "numaggs_over_matrix_rows",
    "particle_filter_posterior",
    "per_group_apply",
    "per_group_sliding_window",
    "positions_within_group",
    "precompute_hurst_exponent",
    "predictor_consensus_entropy",
    "predictor_consensus_mean",
    "predictor_disagreement_features",
    "predictor_disagreement_iqr",
    "predictor_disagreement_var",
    "predictor_pairwise_abs_diffs",
    "predictor_top2_mode_gap",
    "rolling_dfa_alpha",
    "rolling_dominant_freq_idx",
    "rolling_extrema_density",
    "rolling_hf_lf_ratio",
    "rolling_higuchi_fd",
    "rolling_hurst",
    "rolling_integral_above_baseline",
    "rolling_mean_abs_d2",
    "rolling_moving_average",
    "rolling_n_peaks",
    "rolling_n_troughs",
    "rolling_spectral_band_energies",
    "rolling_spectral_entropy",
    "run_pysr_fe",
    "run_pysr_feature_engineering",
    "safely_compute_mps",
    "show_mps_regions",
    "engineered_name_cooccur_svd",
    "cat_cooccurrence_svd_fit",
    "apply_cat_cooccurrence_svd",
    "cat_cooccurrence_svd_with_recipes",
    "graph_neighbor_aggregate",
    "graph_structural_features",
    "link_prediction_features",
    "knn_graph_edges",
    "shared_attribute_edges",
    "fuzzy_partition_encode",
    "fuzzy_partition_fit",
    "fuzzy_partition_transform",
    "fuzzy_partition_names",
    "graph_spectral_features",
    "graph_spectral_feature_names",
    "remediate_drifting_features",
    "entity_inter_event_features",
    "fuzzy_entity_group_features",
    "time_since_state_change",
    "nearest_past_join",
    "leakage_safe_aggregate",
    "randomize_as_of_lag",
    "row_wise_summary_stats",
    "rolling_target_correlation_tracker",
    "compute_auxiliary_feature_prediction_features",
    "compute_cross_sectional_neighbor_features",
    "latent_interaction_features",
    "multi_window_aggregate",
    "entity_diff_features",
    "control_difference_augment",
    "select_significant_lags",
    "two_step_recency_weighted_target_encode",
    "polars_dynamic_window_aggregate",
    "train_sequence2vec",
    "sequence2vec_entity_features",
    "ChildTableSpec",
    "compute_relational_features",
    "stack_relational_features",
    "windowed_edge_aggregate_diff",
    "magnitude_sample_weight",
    "boolean_pair_interactions",
    "is_binary_column",
    "add_sentinel_missing_count_feature",
    "concat_categorical_group",
    "binned_unique_count",
    "multi_decomposition_feature_bank",
    "variance_gated_pairwise_diff",
    "event_proximity_decay_features",
    "tfidf_svd_entity_embedding",
    "gmm_bic_membership_features",
    "latent_parameter_recovery_features",
    "build_panel_sequence_tensor",
]
