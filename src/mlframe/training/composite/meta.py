"""CompositeOrRawStacker -- learn WHEN a composite target helps vs a raw-y model.

Motivation
----------
``CompositeTargetEstimator`` transforms the target (e.g. residual / ratio / log) and inverts at predict.
When the transform matches the data-generating process it is a large win; when the base feature is
misspecified the transform can HURT vs a plain model on raw ``y``. You usually do not know in advance
which regime you are in. This estimator removes the guess: it fits BOTH paths and lets the data decide.

How it works
------------
1. Fit a ``CompositeTargetEstimator`` (transform-and-invert) on full ``(X, y)``.
2. Fit a plain ``raw`` inner regressor on full ``(X, y)``.
3. Generate leakage-free OUT-OF-FOLD predictions for each path via KFold (each fold predicted by a model
   that never saw it), giving an ``(n, 2)`` OOF matrix ``[composite_oof, raw_oof]``.
4. Fit a small NON-NEGATIVE meta-blender (NNLS) of those two OOF columns against ``y``. The weights are
   normalised to sum to 1 when their sum is positive, so the final prediction is a convex blend.

The blend recovers the composite when it dominates (composite weight ~1), the raw model when the
transform hurts (raw weight ~1), and an honest mixture in between -- all chosen on leakage-free OOF
signal rather than a hand-picked flag.

Leakage contract
----------------
OOF predictions come from KFold clones that never train on the rows they predict, so the meta-blender
sees only honest out-of-fold signal (mirrors the ensemble NNLS stacker's contract). The two FULL-data
models (steps 1-2) are used ONLY at predict time, never to fit the weights.

Biz_value: see ``tests/training/test_biz_val_composite_meta.py`` -- on a target where the composite
clearly wins the blend ~= composite; on a misspecified-base target the blend falls back to raw and beats
the standalone composite on OOS RMSE.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold

from .estimator import CompositeTargetEstimator

__all__ = ["CompositeOrRawStacker"]


def _as_1d_y(y: Any) -> np.ndarray:
    """Narrow target pull to a contiguous float ndarray (no frame copy of X)."""
    arr = np.asarray(y, dtype=float)
    return arr.reshape(-1)


def _row_subset(X: Any, idx: np.ndarray) -> Any:
    """Format-native row subset of X (pandas .iloc / polars indexing / ndarray) -- never to_pandas()."""
    # pandas / anything exposing .iloc
    iloc = getattr(X, "iloc", None)
    if iloc is not None:
        return iloc[idx]
    # polars DataFrame
    if hasattr(X, "to_numpy") and hasattr(X, "columns") and not isinstance(X, np.ndarray):
        try:
            return X[idx.tolist()]
        except Exception:  # pragma: no cover - fallback for exotic frames  # nosec B110 - best-effort/optional path, no module logger
            pass
    return np.asarray(X)[idx]


def _fit_nnls_2col(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NNLS fit of a 2-column OOF matrix against y; returns weights summing to 1 (convex) when positive.

    Reuses ``scipy.optimize.nnls``. Rows with any non-finite OOF / y are dropped before the solve so a
    single domain-violation prediction cannot poison the weights. Degenerate fallbacks:
      * no finite rows / both columns constant -> equal split [0.5, 0.5];
      * NNLS returns all-zero (both columns anti-correlated with y) -> equal split.
    """
    from scipy.optimize import nnls

    finite = np.isfinite(oof).all(axis=1) & np.isfinite(y)
    if finite.sum() < 2:
        return np.array([0.5, 0.5])
    A = oof[finite]
    b = y[finite]
    try:
        w, _residual = nnls(A, b)
    except Exception as exc:  # pragma: no cover - solver edge
        logger.debug("NNLS blend weight solve failed, falling back to an equal split: %s", exc)
        return np.array([0.5, 0.5])
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.array([0.5, 0.5])
    return np.asarray(w / total)


class CompositeOrRawStacker(BaseEstimator, RegressorMixin):
    """Blend a composite-target model with a raw-``y`` model, weighted by leakage-free OOF performance.

    Parameters
    ----------
    base_estimator
        Unfitted sklearn-compatible regressor prototype. Cloned for BOTH the composite inner and the raw
        inner (and for every OOF fold). ``None`` defers to :class:`CompositeTargetEstimator`'s own default.
    transform_name
        Target transform for the composite path (one of :func:`list_transforms`); e.g. ``"diff"``.
    base_column / base_columns / group_column
        Plumbing columns forwarded to the composite path (see :class:`CompositeTargetEstimator`).
    n_splits
        KFold splits used to generate the OOF prediction matrix. Default 5.
    shuffle / random_state
        KFold shuffle controls. Default shuffle=True for i.i.d. data; pass shuffle=False for ordered series.
    composite_kwargs
        Extra kwargs passed verbatim to the inner :class:`CompositeTargetEstimator` constructor.

    Attributes set at fit
    ---------------------
    composite_ / raw_
        The two FULL-data fitted models used at predict time.
    weights_
        ``np.ndarray([w_composite, w_raw])`` -- non-negative, summing to 1 when positive.
    oof_matrix_
        The ``(n, 2)`` leakage-free OOF prediction matrix the weights were fit on.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        transform_name: str = "diff",
        base_column: str = "",
        base_columns: Sequence[str] | None = None,
        group_column: str | None = None,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = 0,
        composite_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.transform_name = transform_name
        self.base_column = base_column
        self.base_columns = base_columns
        self.group_column = group_column
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.composite_kwargs = composite_kwargs

    # -- builders -----------------------------------------------------------

    def _make_composite(self) -> CompositeTargetEstimator:
        """Build a fresh, unfitted ``CompositeTargetEstimator`` from this stacker's constructor params."""
        kw = dict(self.composite_kwargs or {})
        return CompositeTargetEstimator(
            base_estimator=clone(self.base_estimator) if self.base_estimator is not None else None,
            transform_name=self.transform_name,
            base_column=self.base_column,
            base_columns=self.base_columns,
            group_column=self.group_column,
            **kw,
        )

    def _make_raw(self) -> Any:
        """Build a fresh, unfitted raw (non-composite) estimator sharing the same learner family as the composite path."""
        if self.base_estimator is None:
            # Mirror CompositeTargetEstimator's default inner so both paths share a learner family.
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(random_state=0)
        return clone(self.base_estimator)

    # -- fit ----------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> "CompositeOrRawStacker":
        """Fit both the composite and raw estimators, computing OOF predictions from each and NNLS-blending them into ``weights_``."""
        y_arr = _as_1d_y(y)
        n = y_arr.shape[0]

        # Effective split count: cannot exceed n, must be >= 2 for a meaningful OOF.
        n_splits = int(max(2, min(self.n_splits, n)))
        kf = KFold(n_splits=n_splits, shuffle=self.shuffle, random_state=self.random_state if self.shuffle else None)

        oof = np.full((n, 2), np.nan, dtype=float)
        for train_idx, test_idx in kf.split(np.arange(n)):
            X_tr = _row_subset(X, train_idx)
            X_te = _row_subset(X, test_idx)
            y_tr = y_arr[train_idx]

            comp = self._make_composite()
            comp.fit(X_tr, y_tr)
            oof[test_idx, 0] = np.asarray(comp.predict(X_te), dtype=float).reshape(-1)

            raw = self._make_raw()
            raw.fit(X_tr, y_tr)
            oof[test_idx, 1] = np.asarray(raw.predict(X_te), dtype=float).reshape(-1)

        self.oof_matrix_ = oof
        self.weights_ = _fit_nnls_2col(oof, y_arr)

        # FULL-data models for prediction (NOT used to fit the weights -> no leakage).
        self.composite_ = self._make_composite()
        self.composite_.fit(X, y_arr)
        self.raw_ = self._make_raw()
        self.raw_.fit(X, y_arr)

        n_feat = getattr(self.composite_, "n_features_in_", None)
        if n_feat is None:
            cols = getattr(X, "columns", None)
            if cols is not None:
                n_feat = len(cols)
            elif getattr(X, "shape", None) is not None and len(X.shape) >= 2:
                n_feat = int(X.shape[1])
        self.n_features_in_ = n_feat
        return self

    # -- predict ------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Predict as the NNLS-weighted blend of the full-data composite and raw estimators' predictions."""
        if not hasattr(self, "weights_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("CompositeOrRawStacker is not fitted; call fit first.")
        comp_pred = np.asarray(self.composite_.predict(X), dtype=float).reshape(-1)
        raw_pred = np.asarray(self.raw_.predict(X), dtype=float).reshape(-1)
        w_c, w_r = self.weights_

        # If the composite produced a non-finite prediction on a row (domain violation that slipped the
        # fallback), lean fully on the raw model there rather than poisoning the blended output with NaN.
        blended = w_c * comp_pred + w_r * raw_pred
        bad = ~np.isfinite(blended)
        if bad.any():
            blended[bad] = raw_pred[bad]
        return np.asarray(blended)
