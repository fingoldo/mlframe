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
        max_features_out: Optional[int] = None,
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
        # Cap on the projected column count. ``None`` / ``0`` disables the auto-tune (legacy behaviour);
        # otherwise ``fit`` adjusts ``interaction_only``, then ``degree``, then skips the expansion entirely
        # to stay under the cap. Wired from ``PreprocessingBackendConfig.polynomial_max_features``.
        self.max_features_out = max_features_out
        # Set to True by ``fit`` when the cap forces a complete skip (no impl built). ``transform`` then
        # returns the input unchanged so the downstream pipeline can no-op safely.
        self._skipped: bool = False
        # Records the actually-fitted (degree, interaction_only) after auto-tune; differs from the
        # constructor values when the cap kicked in. Useful for diagnostics / regression tests.
        self.effective_degree: Optional[int] = None
        self.effective_interaction_only: Optional[bool] = None

    # ------------------------------------------------------------------

    def fit(self, X: Any, feature_names: Optional[List[str]] = None) -> PolynomialFeatureExpander:
        """Fit on a numeric ndarray-like (n_samples, n_features). When ``max_features_out`` is set, the
        expansion config (interaction_only, degree) is auto-tuned downward until ``projected <= cap``;
        if the unary case (degree=1) still exceeds the cap, the whole step is skipped and ``transform``
        returns input untouched. Each tuning step is WARN-logged. Never raises on cap exceedance."""
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X_np.shape}")
        self._n_features_in = X_np.shape[1]

        cap = self.max_features_out
        eff_degree = int(self.degree)
        eff_interaction = bool(self.interaction_only)

        def _project(d: int, io: bool) -> int:
            p = _projected_output_cols(self._n_features_in, d, io)
            if self.include_bias:
                p += 1
            return p

        projected = _project(eff_degree, eff_interaction)

        # Auto-tune ONLY when the cap is positive. None / 0 leaves the historical warn-only behaviour.
        if cap not in (None, 0):
            if projected > cap and not eff_interaction:
                logger.warning(
                    "[fhc] polynomial auto-tune: projected=%d > cap=%d at degree=%d, interaction_only=False; "
                    "flipping interaction_only=True (drops pure-power terms).",
                    projected, cap, eff_degree,
                )
                eff_interaction = True
                projected = _project(eff_degree, eff_interaction)
            while projected > cap and eff_degree > 1:
                logger.warning(
                    "[fhc] polynomial auto-tune: projected=%d > cap=%d at degree=%d, interaction_only=%s; "
                    "decrementing degree -> %d.",
                    projected, cap, eff_degree, eff_interaction, eff_degree - 1,
                )
                eff_degree -= 1
                projected = _project(eff_degree, eff_interaction)
            if projected > cap:
                logger.warning(
                    "[fhc] polynomial auto-tune: even at degree=1 the projected=%d > cap=%d; skipping "
                    "polynomial expansion entirely. transform() will return the input unchanged.",
                    projected, cap,
                )
                self._skipped = True
                self.effective_degree = eff_degree
                self.effective_interaction_only = eff_interaction
                if feature_names is None:
                    feature_names = [f"x{i}" for i in range(self._n_features_in)]
                self._feature_names = list(feature_names)
                self._fitted = True
                return self

        self.effective_degree = eff_degree
        self.effective_interaction_only = eff_interaction

        if projected > 5000:
            logger.warning(
                "[fhc] polynomial expansion: %d in -> %d out cols (degree=%d, interaction_only=%s). "
                "Memory cost ~%.1f MB at 1M rows.",
                self._n_features_in, projected, eff_degree, eff_interaction, projected * 4 / 1e6,
            )
        else:
            logger.info(
                "[fhc] polynomial expansion: %d in -> %d out cols (degree=%d)",
                self._n_features_in, projected, eff_degree,
            )

        from sklearn.preprocessing import PolynomialFeatures
        self._impl = PolynomialFeatures(
            degree=eff_degree,
            interaction_only=eff_interaction,
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
            # Wave 37 P1 fix (2026-05-20): sklearn convention is
            # NotFittedError. Pipeline / cross-val machinery catches
            # NotFittedError, not RuntimeError; pre-fix unfitted-state
            # failures leaked past expected handlers.
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("PolynomialFeatureExpander not fitted -- call .fit() first")
        X_np = np.asarray(X, dtype=np.float32)
        if self._skipped:
            return X_np
        return self._impl.transform(X_np).astype(np.float32, copy=False)

    def fit_transform(self, X: Any, feature_names: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(X, feature_names=feature_names).transform(X)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_features_in(self) -> int:
        if self._n_features_in is None:
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("PolynomialFeatureExpander not fitted yet")
        return self._n_features_in

    @property
    def feature_names_out(self) -> List[str]:
        if not self._fitted:
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("PolynomialFeatureExpander not fitted yet")
        return list(self._feature_names) if self._feature_names else []

    def __repr__(self) -> str:
        return f"PolynomialFeatureExpander(degree={self.degree}, " f"interaction_only={self.interaction_only}, " f"fitted={self._fitted})"


__all__ = ["PolynomialFeatureExpander"]
