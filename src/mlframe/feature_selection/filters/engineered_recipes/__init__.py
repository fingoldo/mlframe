"""Recipe-based replay of engineered features for ``MRMR.transform``.

A *recipe* is a frozen description of how to recompute one engineered column from the original feature matrix. ``MRMR.fit`` records one recipe per surviving
engineered feature; ``MRMR.transform`` replays each recipe against the test X and appends the resulting columns to the output.

Recipe kinds:
``"unary_binary"``: numeric pair FE -- ``binary(unary_a(X[a]), unary_b(X[b]))``, optionally discretized.
``"factorize"``:    cat-FE -- ``merge_vars`` of k ordinal-encoded categorical columns (XOR-style synergy capture).
``"target_encoding"``: ``E[Y | merged_class]`` per cell, with optional OOF smoothing.

The recipe is a small frozen dataclass (no behaviour bound to ``self``) so it round-trips cleanly through pickle and ``sklearn.base.clone``.

This is the package facade: the recipe dataclass + the inline ``unary_binary`` /
``factorize`` / ``hermite_pair`` / ``cluster_aggregate`` / ``target_encoding``
replay bodies live in cohesive ``_recipe_*`` submodules; the orthogonal-basis,
categorical-encoding, missingness / ratio, and grouped / temporal builders live
in their own siblings. Every public + private name external code imported from
``engineered_recipes`` is re-exported here unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

# Core dataclass + array-aware extra-equality helper (leaf submodule).
from ._recipe_core import EngineeredRecipe, _extra_equal

# Column extraction + int-coercion helpers shared across the replay paths.
from ._recipe_extract import (
    _coerce_to_int_with_nan_handling,
    _extract_column,
)

# The ``apply_recipe`` dispatcher.
from ._recipe_dispatch import apply_recipe

# Numeric pair-FE (unary_binary) builder + replay + the orjson preprocess helper.
from ._recipe_unary_binary import (
    _apply_unary_binary,
    _orjson_pp,
    build_unary_binary_recipe,
)

# Cat-FE factorize replay (pair + k-way chained).
from ._recipe_factorize import (
    _apply_factorize,
    _apply_factorize_kway,
)

# Hermite-pair / cluster-aggregate / target-encoding builders + replay.
from ._recipe_poly_cluster import (
    _apply_cluster_aggregate,
    _apply_hermite_pair,
    _apply_target_encoding,
    build_cluster_aggregate_recipe,
    build_hermite_pair_recipe,
)

# Orthogonal-triplet cross recipe builder + replay (sibling FE module).
from .._orthogonal_triplet_fe_recipes import (
    build_orth_triplet_cross_recipe,
    _apply_orth_triplet_cross,
)

# Orthogonal-quadruplet cross recipe builder + replay (sibling FE module).
from .._orthogonal_quadruplet_fe_recipes import (
    build_orth_quadruplet_cross_recipe,
    _apply_orth_quadruplet_cross,
)

# Numeric-decomposition recipe builders (sibling FE module); apply is dispatched inline.
from .._numeric_decompose_fe import (
    build_numeric_rounding_recipe,
    build_digit_extract_recipe,
)

# Temporal leak-safe aggregation recipe builders (sibling FE module); apply is dispatched inline.
from .._temporal_agg_fe import (
    build_temporal_expanding_recipe,
    build_temporal_rolling_recipe,
    build_temporal_lag_recipe,
)

# Periodic / modular decomposition recipe builder (sibling FE module); apply is dispatched inline.
from .._periodic_fe import build_modular_recipe

# Per-group distribution-distance recipe builder (sibling FE module); apply is dispatched inline.
from .._group_distance_fe import build_group_distance_recipe

# Hinge / piecewise-linear change-point basis recipe builder + replay (sibling
# FE module). Apply path dispatched lazily in the dispatcher;
# re-exported so external importers resolve the builder from this module.
from .._hinge_basis_fe import (
    build_hinge_basis_recipe,
    _apply_hinge_basis,
)

# Haar wavelet / localized multiresolution basis builder + replay helper (backlog
# #13, 2026-06-09) live in the sibling ``_wavelet_basis_fe`` one package level up;
# re-exported so external importers resolve the builder from this module (mirrors
# the hinge re-export above).
from .._wavelet_basis_fe import (
    build_orth_wavelet_recipe,
    _apply_orth_wavelet,
)

# Orthogonal-basis recipe builders + replay helpers live in the sibling
# ``_orth_basis_recipes`` (apply path dispatched lazily in the dispatcher);
# re-exported so external importers keep resolving them from this module.
from ._orth_basis_recipes import (
    _apply_orth_fourier,
    _apply_orth_pair_cross,
    _apply_orth_pre_transform,
    _apply_orth_spline,
    _apply_orth_univariate,
    _bspline_basis_values,
    _eval_orth_basis_column,
    _fit_spline_knots,
    _freeze_preprocess_params,
    build_orth_cluster_basis_recipe,
    build_orth_diff_basis_recipe,
    build_orth_fourier_recipe,
    build_orth_pair_cross_recipe,
    build_orth_spline_recipe,
    build_orth_univariate_recipe,
)

# Categorical-encoding recipe builders + replay helpers live in the sibling
# ``_encoding_recipes`` (apply path dispatched lazily in the dispatcher);
# re-exported so external importers keep resolving them from this module.
from ._encoding_recipes import (
    _apply_cat_num_residual,
    _apply_count_encoded,
    _apply_frequency_encoded,
    _apply_kfold_target_encoded,
    build_cat_num_residual_recipe,
    build_cat_pair_cross_recipe,
    build_cat_triple_cross_recipe,
    build_count_encoded_recipe,
    build_frequency_encoded_recipe,
    build_kfold_target_encoded_recipe,
)

# MI-greedy + missingness / pairwise-ratio recipe builders live in the sibling
# ``_missingness_ratio_recipes`` (mi-greedy apply dispatched lazily in the
# dispatcher); re-exported so external importers keep resolving them here.
from ._missingness_ratio_recipes import (
    _apply_mi_greedy_transform,
    build_missing_indicator_recipe,
    build_missingness_count_recipe,
    build_missingness_pattern_recipe,
    build_mi_greedy_transform_recipe,
    build_pairwise_ratio_recipe,
)

# Grouped / temporal recipe builders live in the sibling ``_grouped_recipes``
# (apply helpers dispatched lazily in the dispatcher); re-exported for importers.
from ._grouped_recipes import (
    build_composite_group_agg_recipe,
    build_grouped_agg_recipe,
    build_grouped_delta_recipe,
    build_grouped_quantile_recipe,
    build_lagged_diff_recipe,
    build_target_aware_group_bin_recipe,
)
