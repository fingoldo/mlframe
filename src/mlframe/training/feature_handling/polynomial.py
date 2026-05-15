"""
Polynomial feature interactions (opt-in).

Round-3 future-proofing F19. Exposed via
``PreprocessingBackendConfig.polynomial_degree``; default ``None``
(disabled). Wraps either :class:`sklearn.preprocessing.PolynomialFeatures`
or polars-ds ``Blueprint.polynomial_features`` when available.

Memory safety: a degree-2 polynomial of 100 numeric features yields
~5050 cols (n*(n+1)/2 + n + 1). The wrapper logs an INFO line at fit
with the projected output size so the user can spot a mistake before
it OOMs the cache layer.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from mlframe.training.feature_handling.polars_capability import (
    PolarsNativeDispatcher,
)

logger = logging.getLogger(__name__)


def _projected_output_cols(n_in: int, degree: int, interaction_only: bool) -> int:
    """Combinatorial size estimate. ``interaction_only`` excludes
    pure-power terms (x^2, x^3) and keeps only cross-terms.
    """
    from math import comb
    if degree < 1:
        return n_in
    if interaction_only:
        # sum of C(n, k) for k = 1..degree
        return sum(comb(n_in, k) for k in range(1, degree + 1))
    # full polynomial: sum of multiset coefficients
    return sum(comb(n_in + k - 1, k) for k in range(1, degree + 1))


class PolynomialFeatureExpander:
    """Per-numeric-block polynomial expansion. Stateful: ``fit()`` on
    the train slice, ``transform()`` on any frame with matching schema.

    Interface mirrors sklearn so a downstream consumer can swap in
    ``sklearn.preprocessing.PolynomialFeatures`` directly.
    """

    def __init__(
        self,
        *,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        prefer_polarsds: bool = True,
    ):
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self._dispatcher = PolarsNativeDispatcher(prefer_polarsds=prefer_polarsds)
        self._impl: Optional[Any] = None
        self._n_features_in: Optional[int] = None
        self._fitted: bool = False
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------

    def fit(self, X: Any, feature_names: Optional[List[str]] = None) -> PolynomialFeatureExpander:
        """Fit on a numeric ndarray-like (n_samples, n_features)."""
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X_np.shape}")
        self._n_features_in = X_np.shape[1]

        projected = _projected_output_cols(
            self._n_features_in, self.degree, self.interaction_only,
        )
        if self.include_bias:
            projected += 1
        if projected > 5000:
            logger.warning(
                "[fhc] polynomial expansion: %d in -> %d out cols (degree=%d, "
                "interaction_only=%s). Memory cost ~%.1f MB at 1M rows. "
                "Consider degree=1 / interaction_only=True / smaller numeric block.",
                self._n_features_in, projected, self.degree,
                self.interaction_only, projected * 4 / 1e6,
            )
        else:
            logger.info(
                "[fhc] polynomial expansion: %d in -> %d out cols (degree=%d)",
                self._n_features_in, projected, self.degree,
            )

        # TODO(phase upstream): dispatcher.has("blueprint.polynomial_features")
        # path lands when polars-ds wires the equivalent method.
        from sklearn.preprocessing import PolynomialFeatures
        self._impl = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        self._impl.fit(X_np)

        # Cache feature names
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self._n_features_in)]
        try:
            self._feature_names = list(self._impl.get_feature_names_out(feature_names))
        except Exception:  # pragma: no cover
            self._feature_names = [f"poly_{i}" for i in range(projected)]

        self._fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "PolynomialFeatureExpander not fitted -- call .fit() first"
            )
        X_np = np.asarray(X, dtype=np.float32)
        return self._impl.transform(X_np).astype(np.float32, copy=False)

    def fit_transform(self, X: Any, feature_names: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(X, feature_names=feature_names).transform(X)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_features_in(self) -> int:
        if self._n_features_in is None:
            raise RuntimeError("not fitted yet")
        return self._n_features_in

    @property
    def feature_names_out(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError("not fitted yet")
        return list(self._feature_names) if self._feature_names else []

    def __repr__(self) -> str:
        return (
            f"PolynomialFeatureExpander(degree={self.degree}, "
            f"interaction_only={self.interaction_only}, "
            f"fitted={self._fitted})"
        )


__all__ = ["PolynomialFeatureExpander"]
