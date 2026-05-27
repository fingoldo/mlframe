"""Feature engineering helpers for tabular and time-series ML.

Submodules:
    anchor             - per-group anchor-based linear extrapolation features (sparse-truth case).
    basic              - date-part decomposition, symbolic regression entry-point.
    bayesian           - particle filter posterior features on 1-D state.
    bruteforce         - PySR-driven brute-force feature search.
    categorical        - aggregates for categorical / repeated-value series.
    ensemble_features  - per-row disagreement features across N stacked predictors.
    financial          - OHLCV ratios, lags, TA-Lib indicators, market-wide features.
    grouped            - per-group sliding-window iterator primitives.
    hurst              - Hurst exponent (R/S analysis) + rolling DFA / Higuchi FD.
    mps                - Maximum Profit System target/regions.
    numerical          - rich numerical aggregates for 1d vectors.
    spatial            - XY / N-D Euclidean kNN aggregator features (geo / point-cloud).
    spectral           - FFT-based rolling band energies, spectral entropy, dominant freq.
    stationarity       - frac_diff (Lopez de Prado fractional differencing) + future ADF/KPSS.
    timeseries         - windowed feature aggregation and ACF helpers.
    transformer        - frozen transformer-style FE: RFF, positional encoding, row-attention (multi-head softmax-weighted kNN-TE).
    wavelet_dwt        - discrete + continuous wavelet transforms + denoising.
    windowed_shape     - rolling shape features (mean_abs_d2, n_peaks, n_troughs, extrema density, integral_above_baseline).
"""

from __future__ import annotations


from .anchor import add_anchor_extrapolation_features
from .basic import create_date_features, run_pysr_fe
from .bayesian import particle_filter_posterior
from .bruteforce import run_pysr_feature_engineering
from .categorical import compute_countaggs, get_countaggs_names
from .ensemble_features import (
    predictor_consensus_entropy,
    predictor_consensus_mean,
    predictor_disagreement_features,
    predictor_disagreement_iqr,
    predictor_disagreement_var,
    predictor_pairwise_abs_diffs,
    predictor_top2_mode_gap,
)
from .financial import (
    add_fast_rolling_stats,
    add_ohlcv_ratios_rlags,
    add_ohlcv_ta_indicators,
    create_ohlcv_wholemarket_features,
    merge_perticker_and_wholemarket_features,
)
from .grouped import iter_group_segments, per_group_apply, per_group_sliding_window
from .hurst import (
    compute_hurst_exponent,
    compute_hurst_rs,
    dfa_alpha,
    higuchi_fd,
    precompute_hurst_exponent,
    rolling_dfa_alpha,
    rolling_higuchi_fd,
    rolling_hurst,
)
from .spatial import knn_aggregate, knn_within_bucket_aggregate
from .spectral import (
    rolling_dominant_freq_idx,
    rolling_hf_lf_ratio,
    rolling_spectral_band_energies,
    rolling_spectral_entropy,
)
from .stationarity import frac_diff, frac_diff_weights
from .windowed_shape import (
    rolling_extrema_density,
    rolling_integral_above_baseline,
    rolling_mean_abs_d2,
    rolling_n_peaks,
    rolling_n_troughs,
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

__all__ = [
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
]
