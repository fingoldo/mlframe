"""Feature engineering helpers for tabular and time-series ML.

Submodules:
    basic          - date-part decomposition, symbolic regression entry-point.
    bruteforce     - PySR-driven brute-force feature search.
    categorical    - aggregates for categorical / repeated-value series.
    financial      - OHLCV ratios, lags, TA-Lib indicators, market-wide features.
    hurst          - Hurst exponent (R/S analysis).
    mps            - Maximum Profit System target/regions.
    numerical      - rich numerical aggregates for 1d vectors.
    timeseries     - windowed feature aggregation and ACF helpers.
    transformer    - frozen transformer-style FE: RFF, positional encoding, row-attention (multi-head softmax-weighted kNN-TE).
"""

from __future__ import annotations


from .basic import create_date_features, run_pysr_fe
from .bruteforce import run_pysr_feature_engineering
from .categorical import compute_countaggs, get_countaggs_names
from .financial import (
    add_fast_rolling_stats,
    add_ohlcv_ratios_rlags,
    add_ohlcv_ta_indicators,
    create_ohlcv_wholemarket_features,
    merge_perticker_and_wholemarket_features,
)
from .hurst import compute_hurst_exponent, compute_hurst_rs, precompute_hurst_exponent
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
    "find_best_mps_sequence",
    "find_maximum_profit_system",
    "general_acf",
    "get_basic_feature_names",
    "get_countaggs_names",
    "get_moments_slope_mi_feature_names",
    "get_numaggs_names",
    "get_simple_stats_names",
    "merge_perticker_and_wholemarket_features",
    "numaggs_over_matrix_rows",
    "positions_within_group",
    "precompute_hurst_exponent",
    "rolling_moving_average",
    "run_pysr_fe",
    "run_pysr_feature_engineering",
    "safely_compute_mps",
    "show_mps_regions",
]
