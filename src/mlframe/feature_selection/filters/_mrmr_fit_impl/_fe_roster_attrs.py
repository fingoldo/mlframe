"""The per-family engineered-feature rosters, in one place, so "it exists after fit" is a real contract.

Each FE family seeds its own ``<family>_features_`` list at the top of the cascade stage that owns it and
then appends to it. Tests read those rosters to assert a family stayed silent, and reading them through a
``getattr(..., [])`` default makes an absent attribute indistinguishable from an empty one -- a renamed
roster passes such a test forever. Dropping the default only works if the attribute is genuinely always
there, and it was NOT: the multioutput path fits a clone per target column and returns before the
single-target body runs, so none of these ever appeared on the outer estimator.

The names are collected here rather than repeated at each site, and ``tests/feature_selection`` has a meta
test asserting this tuple still matches what the cascades actually seed -- a family added without being
listed here would otherwise reintroduce exactly the gap this module closes.
"""

from __future__ import annotations

from typing import Any, Tuple

FE_ROSTER_ATTRS: Tuple[str, ...] = (
    "_adaptive_fourier_features_",
    "_hinge_features_",
    "cat_num_interaction_features_",
    "cat_pair_features_",
    "cat_triple_features_",
    "composite_group_agg_features_",
    "conditional_dispersion_features_",
    "conditional_gate_features_",
    "conditional_quantile_rank_features_",
    "conditional_residual_features_",
    "count_encoding_features_",
    "frequency_encoding_features_",
    "group_distance_features_",
    "grouped_agg_features_",
    "grouped_delta_features_",
    "grouped_quantile_features_",
    "hybrid_orth_features_",
    "integer_lattice_features_",
    "kfold_te_features_",
    "lagged_diff_features_",
    "lof_features_",
    "mahalanobis_density_features_",
    "mi_greedy_features_",
    "missingness_count_features_",
    "missingness_indicator_features_",
    "missingness_pattern_features_",
    "modular_features_",
    "numeric_decompose_features_",
    "ordinal_pattern_features_",
    "pairwise_log_ratio_features_",
    "pairwise_modular_features_",
    "pairwise_ratio_features_",
    "random_fourier_features_",
    "rankgauss_features_",
    "rare_category_features_",
    "row_argmax_features_",
    "sir_direction_features_",
    "temporal_agg_features_",
    "wavelet_features_",
)


def seed_empty_fe_rosters(estimator: Any) -> None:
    """Set every roster to an empty list on ``estimator``.

    For a fit path where no FE family ran on THIS object -- the multioutput fan-out, whose engineering
    happens inside per-target clones -- an empty roster is the truthful answer, and it is the answer callers
    have to be able to read without guessing whether the attribute exists.
    """
    for name in FE_ROSTER_ATTRS:
        setattr(estimator, name, [])
