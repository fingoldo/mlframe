"""Deprecated import path: everything lives in ``correlated_features`` now; ``GroupAwareMRMR`` is the old name of ``CorrelatedFeaturesSelector``.

The module and class names are kept so old imports and pickles (which reference ``mlframe.feature_selection.filters.group_aware.GroupAwareMRMR``) still resolve.
"""
from __future__ import annotations

import warnings
from typing import Any

from . import correlated_features as _correlated_features
from .correlated_features import CorrelatedFeaturesSelector, cluster_features_by_correlation

__all__ = ["CorrelatedFeaturesSelector", "GroupAwareMRMR", "cluster_features_by_correlation"]


class GroupAwareMRMR(CorrelatedFeaturesSelector):
    """Deprecated alias of :class:`CorrelatedFeaturesSelector` (it clusters by correlation, not by a groups column)."""

    def __init__(
        self,
        estimator,
        corr_threshold: float = 0.9,
        corr_method: str = "spearman",
        expand: bool = False,
        min_reduction: float = 0.05,
    ):
        warnings.warn(
            "GroupAwareMRMR is deprecated and will be removed; use mlframe.feature_selection.filters.correlated_features.CorrelatedFeaturesSelector.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            estimator,
            corr_threshold=corr_threshold,
            corr_method=corr_method,
            expand=expand,
            min_reduction=min_reduction,
        )


def __getattr__(name: str) -> Any:
    # Forward private helpers (``_cluster_medoids`` etc.) that old callers imported from this module.
    try:
        return getattr(_correlated_features, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
