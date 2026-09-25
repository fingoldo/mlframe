"""Shared state of one ``MRMR.fit``: the values every stage of ``_fit_impl`` reads, held in two namespaces.

``_fit_impl`` used to keep ~60 of these as plain locals, so every piece carved out of it needed them all as parameters.
They are grouped instead: ``FERecipes`` holds the per-family recipe registries the feature-engineering stages fill and
``transform`` replays; ``FEParams`` holds the feature-engineering parameters resolved once at the start of the fit.
"""

from __future__ import annotations

from types import SimpleNamespace


class FERecipes(SimpleNamespace):
    """Per-family recipe registries of one fit (``recipes.hybrid_orth``, ``recipes.kfold_te``, ...), each a dict the
    family's FE stage fills and ``transform`` replays."""


class FEParams(SimpleNamespace):
    """Feature-engineering parameters resolved at the start of one fit (``fe.max_steps``, ``fe.min_pair_mi``, ...)."""


# Recipe families seeded into ``engineered_recipes`` before the screening loop, in routing order (a later family wins a key clash).
ROUTED_RECIPE_FAMILIES = (
    "hybrid_orth",
    "mi_greedy",
    "kfold_te",
    "binned_agg",
    "count_enc",
    "freq_enc",
    "cat_num",
    "miss_ind",
    "miss_cnt",
    "miss_pat",
    "ratio",
    "log_ratio",
    "grouped_delta",
    "lagged_diff",
    "grouped_agg",
    "composite_group_agg",
    "grouped_quantile",
    "cat_pair",
    "cat_triple",
    "numeric_decompose",
    "modular",
    "pairwise_modular",
    "integer_lattice",
    "row_argmax",
    "conditional_gate",
    "group_distance",
    "rare_category",
    "conditional_residual",
    "conditional_dispersion",
    "conditional_quantile_rank",
    "ordinal_pattern",
    "random_fourier",
    "sir_direction",
    "lof",
    "mahalanobis_density",
    "wavelet",
    "rankgauss",
    "temporal_agg",
)
