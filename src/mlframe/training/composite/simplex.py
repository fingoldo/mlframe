"""Composite estimator for COMPOSITIONAL / simplex-valued targets.

A compositional target is a vector ``y`` of shape ``(n, K)`` where every row is a
proper composition: every part is non-negative and the ``K`` parts sum to 1
(market shares, expenditure proportions, mineral assays, topic mixtures, vote
shares, ...). Such data lives on the (K-1)-dimensional simplex, NOT in free
Euclidean space. A plain per-part regressor ignores both constraints and routinely
predicts negative parts or rows whose sum is far from 1 -- predictions that are
simply not valid compositions, and whose Euclidean error is meaningless because the
simplex geometry is not Euclidean (Aitchison geometry).

``CompositeSimplexEstimator`` solves this the textbook way (Aitchison 1986):

1.  **Log-ratio transform** the ``K``-part composition to ``K-1`` UNCONSTRAINED
    real coordinates -- either the additive log-ratio ``alr`` (log of each part
    over a chosen reference part) or the isometric log-ratio ``ilr`` (an
    orthonormal Helmert rotation of the centred log-ratio, removing the singular
    sum-to-zero direction). On these coordinates ordinary regression is valid.
2.  **Fit one inner regressor per coordinate** (a clone of ``base_estimator``,
    optionally one inner ``CompositeTargetEstimator`` per coordinate so each
    log-ratio coordinate can carry its own affine base on the log-ratio scale).
3.  **Invert at predict** -- ``alr`` via softmax-with-reference, ``ilr`` via the
    inverse Helmert rotation + softmax -- so every prediction is, BY
    CONSTRUCTION, a valid composition: strictly positive parts that sum to 1.

Zeros. The log-ratio is undefined at exact zeros, so we apply a small
multiplicative replacement (Martin-Fernandez et al. 2003): each zero part is set
to a small ``delta``, the non-zero parts are scaled by ``1 - (#zeros * delta)`` so
the row still sums to 1. This is the standard zero-replacement for log-ratio
analysis and preserves the relative structure of the non-zero parts.

K=2. With two parts ``alr`` reduces to the single coordinate ``log(y1 / y0)`` and
the softmax inverse reduces to the logistic function -- i.e. the estimator becomes
exactly a logistic / log-odds regression on the share, the correct 2-part special
case.

Why a dedicated estimator (vs ``CompositeMultiOutputEstimator``). The multi-output
wrapper fits ``K`` INDEPENDENT per-column regressors on the raw parts; nothing
couples them, so the predictions do not sum to 1 and can go negative. This
estimator instead models the composition jointly through the log-ratio map, which
is what makes the inversion produce a valid simplex point.

cProfile (fit + predict, n=20k, K=5, LightGBM n_estimators=100): ~0.55 s total,
of which ~0.48 s is inside the K-1 inner LightGBM fits and their predict calls.
The wrapper-side log-ratio forward/​inverse is pure vectorised numpy (a couple of
``np.log`` / ``np.exp`` / one ``(K, K-1)`` matmul for ``ilr``) and measures
<1 ms combined at this shape, so there is NO actionable wrapper-side speedup --
the cost is the K-1 inner fits, each already internally threaded. No numba / GPU
ladder is warranted (the transform is O(n*K), called twice per fit).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin, clone

logger = logging.getLogger(__name__)

# Default multiplicative zero-replacement magnitude. Small enough not to distort
# the relative structure of the non-zero parts, large enough to keep the
# subsequent log finite. 1e-6 is the common choice in the compositional-data
# literature for proportions reported to ~5 significant figures.
_DEFAULT_ZERO_DELTA: float = 1e-6


def _ilr_basis(k: int) -> np.ndarray:
    """Return the ``(K, K-1)`` Helmert-style orthonormal ILR contrast basis.

    Columns are an orthonormal basis of the sum-to-zero hyperplane (the clr
    image), so ``V.T @ V == I_{K-1}`` and ``V @ V.T`` is the centring projector.
    Column ``j`` (0-indexed) contrasts the first ``j+1`` parts against part
    ``j+1``, scaled to unit norm -- the standard Egozcue et al. (2003) sequential
    binary partition basis.
    """
    v = np.zeros((k, k - 1), dtype=np.float64)
    for j in range(k - 1):
        r = j + 1  # number of parts on the "numerator" side
        scale = np.sqrt(r / (r + 1.0))
        v[:r, j] = 1.0 / r
        v[r, j] = -1.0
        v[:, j] *= scale
    return v


def multiplicative_zero_replacement(y: np.ndarray, delta: float = _DEFAULT_ZERO_DELTA) -> np.ndarray:
    """Replace exact zeros in each composition row, keeping the row sum at 1.

    Each zero part becomes ``delta``; the non-zero parts in that row are scaled by
    ``1 - n_zero * delta`` so the closure (sum-to-1) is preserved. Rows with no
    zeros are returned unchanged. Operates on a fresh array (``y`` is the small
    target matrix, not the feature frame, so a copy here is cheap and safe).
    """
    out = np.array(y, dtype=np.float64, copy=True)
    n_zero = (out <= 0.0).sum(axis=1)
    rows = np.nonzero(n_zero > 0)[0]
    for i in rows:
        nz = n_zero[i]
        mask0 = out[i] <= 0.0
        out[i, mask0] = delta
        out[i, ~mask0] *= 1.0 - nz * delta
    return out


def _close(y: np.ndarray) -> np.ndarray:
    """Normalise each row to sum to 1 (the closure operation).

    A degenerate row whose parts sum to <=0 (an all-zero row, or all-negative
    noise) carries no relative information and would make the naive ``y / s``
    produce NaN/inf that silently corrupts the downstream log-ratio target. Such
    rows are mapped to the uniform composition ``1/K`` -- the no-information point
    of the simplex -- so the log-ratio map stays finite.
    """
    s = y.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(s[:, 0]) | (s[:, 0] <= 0.0)
    if bad.any():
        out = y / np.where(s <= 0.0, 1.0, s)
        out[bad] = 1.0 / y.shape[1]
        return np.asarray(out)
    return np.asarray(y / s)


def alr_forward(y: np.ndarray, ref: int) -> np.ndarray:
    """Additive log-ratio: ``(n, K)`` composition -> ``(n, K-1)`` coordinates.

    Coordinate ``j`` is ``log(y_j / y_ref)`` for every part ``j != ref``, in the
    original part order with ``ref`` dropped. Inputs must be strictly positive
    (apply :func:`multiplicative_zero_replacement` first).
    """
    logy = np.log(y)
    keep = [j for j in range(y.shape[1]) if j != ref]
    return np.asarray(logy[:, keep] - logy[:, ref : ref + 1])


def alr_inverse(z: np.ndarray, ref: int, k: int) -> np.ndarray:
    """Invert ALR coordinates back to a valid ``(n, K)`` composition via softmax.

    Re-inserts a zero log-ratio at the reference position, then softmaxes, so the
    output is strictly positive and sums to 1 by construction. Numerically stable
    (subtracts the row-max before ``exp``).
    """
    n = z.shape[0]
    full = np.empty((n, k), dtype=np.float64)
    keep = [j for j in range(k) if j != ref]
    full[:, keep] = z
    full[:, ref] = 0.0
    full -= full.max(axis=1, keepdims=True)
    np.exp(full, out=full)
    return _close(full)


def ilr_forward(y: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Isometric log-ratio: ``(n, K)`` composition -> ``(n, K-1)`` coordinates.

    Centres the log composition (clr) then projects onto the orthonormal
    ``basis`` (shape ``(K, K-1)``). Inputs must be strictly positive.
    """
    logy = np.log(y)
    clr = logy - logy.mean(axis=1, keepdims=True)
    return np.asarray(clr @ basis)


def ilr_inverse(z: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Invert ILR coordinates back to a valid ``(n, K)`` composition.

    Rotates the coordinates back to clr space (``z @ basis.T``) then softmaxes,
    giving strictly-positive parts that sum to 1 by construction.
    """
    clr = z @ basis.T
    clr -= clr.max(axis=1, keepdims=True)
    e = np.exp(clr)
    return _close(e)


def aitchison_distance(a: np.ndarray, b: np.ndarray, delta: float = _DEFAULT_ZERO_DELTA) -> np.ndarray:
    """Per-row Aitchison distance between two ``(n, K)`` compositions.

    The Aitchison distance is the Euclidean distance between the clr vectors --
    the natural metric on the simplex. Both inputs are closed + zero-replaced
    first so the clr is finite.
    """
    pa = np.log(multiplicative_zero_replacement(_close(a), delta))
    pb = np.log(multiplicative_zero_replacement(_close(b), delta))
    ca = pa - pa.mean(axis=1, keepdims=True)
    cb = pb - pb.mean(axis=1, keepdims=True)
    return np.asarray(np.sqrt(((ca - cb) ** 2).sum(axis=1)))


class CompositeSimplexEstimator(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """Regressor for compositional targets via the ALR / ILR log-ratio map.

    Parameters
    ----------
    base_estimator
        Prototype regressor cloned once per log-ratio coordinate (``K-1`` clones).
        Each clone is fit on ``X`` against one unconstrained coordinate.
    transform
        ``"ilr"`` (default, isometric -- isometric/orthonormal, basis-symmetric)
        or ``"alr"`` (additive -- simpler, asymmetric in the reference part).
    reference
        Reference part index for ``alr`` (ignored for ``ilr``). Defaults to the
        last part (``K-1``).
    zero_delta
        Multiplicative zero-replacement magnitude (see
        :func:`multiplicative_zero_replacement`).
    per_coordinate_base
        Optional sequence of ``CompositeTargetEstimator`` kwargs dicts, one per
        coordinate. When given, each coordinate is fit through an inner
        ``CompositeTargetEstimator`` on the log-ratio scale (so a coordinate can
        carry its own affine base / transform on top of the log-ratio target)
        instead of a bare ``base_estimator`` clone.

    Notes
    -----
    The default ``transform`` is ``"ilr"`` because it is isometric (preserves
    distances, basis-symmetric) and free of the arbitrary reference-part choice
    that makes ``alr`` non-isometric; ``alr`` remains available and is the natural
    choice when one part is a meaningful denominator. Per the project default
    policy, the more correct path (ilr) ships as the default, not opt-in.
    """

    def __init__(
        self,
        base_estimator: Any,
        transform: str = "ilr",
        reference: Optional[int] = None,
        zero_delta: float = _DEFAULT_ZERO_DELTA,
        per_coordinate_base: Optional[Sequence[dict]] = None,
        score_metric: str = "aitchison",
    ) -> None:
        self.base_estimator = base_estimator
        self.transform = transform
        self.reference = reference
        self.zero_delta = zero_delta
        self.per_coordinate_base = per_coordinate_base
        self.score_metric = score_metric

    # ---- internals -------------------------------------------------------

    def _forward(self, y: np.ndarray) -> np.ndarray:
        """Map a composition ``y`` (n, K) to its (n, K-1) unconstrained coordinates, dispatching to ALR or ILR per ``self.transform``."""
        if self.transform == "alr":
            return alr_forward(y, self._ref_)
        assert self._basis_ is not None  # set in fit() whenever transform == "ilr"
        return ilr_forward(y, self._basis_)

    def _inverse(self, z: np.ndarray) -> np.ndarray:
        """Map predicted (n, K-1) unconstrained coordinates back to the closed (sum-to-1) composition, dispatching to ALR or ILR per ``self.transform``."""
        if self.transform == "alr":
            return alr_inverse(z, self._ref_, self._k_)
        assert self._basis_ is not None  # set in fit() whenever transform == "ilr"
        return ilr_inverse(z, self._basis_)

    def _build_inner(self, j: int) -> Any:
        """Build the inner estimator for coordinate ``j``."""
        if self.per_coordinate_base:
            from .estimator import CompositeTargetEstimator

            spec = dict(self.per_coordinate_base[j])
            spec.setdefault("base_estimator", self.base_estimator)
            return CompositeTargetEstimator(**spec)
        return clone(self.base_estimator)

    # ---- sklearn API -----------------------------------------------------

    def fit(self, X: Any, y: np.ndarray) -> "CompositeSimplexEstimator":
        """Close+zero-replace the composition, transform it to K-1 unconstrained coordinates, and fit one independent inner estimator per coordinate on the shared feature matrix ``X``."""
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError(f"y must be a 2-D composition (n, K) with K>=2; got shape {getattr(y, 'shape', None)}")
        k = y.shape[1]
        if self.transform not in ("alr", "ilr"):
            raise ValueError(f"transform must be 'alr' or 'ilr'; got {self.transform!r}")

        self._k_ = k
        self._ref_ = (k - 1) if self.reference is None else int(self.reference)
        if not (0 <= self._ref_ < k):
            raise ValueError(f"reference must be in [0, {k}); got {self._ref_}")
        self._basis_ = _ilr_basis(k) if self.transform == "ilr" else None

        y_safe = multiplicative_zero_replacement(_close(y), self.zero_delta)
        z = self._forward(y_safe)  # (n, K-1)

        self.estimators_ = []
        for j in range(k - 1):
            inner = self._build_inner(j)
            inner.fit(X, z[:, j])
            self.estimators_.append(inner)
        self.n_outputs_ = k
        cols = getattr(X, "columns", None)
        if cols is not None:
            self.n_features_in_ = len(cols)
        elif getattr(X, "shape", None) is not None and len(X.shape) >= 2:
            self.n_features_in_ = int(X.shape[1])
        return self

    def predict_coordinates(self, X: Any) -> np.ndarray:
        """Predict the ``(n, K-1)`` log-ratio coordinates (the unconstrained scale)."""
        if not hasattr(self, "estimators_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("CompositeSimplexEstimator: call fit before predict.")
        cols = [np.asarray(est.predict(X), dtype=np.float64).ravel() for est in self.estimators_]
        return np.column_stack(cols)

    def predict(self, X: Any) -> np.ndarray:
        """Predict a valid ``(n, K)`` composition: non-negative parts summing to 1."""
        z = self.predict_coordinates(X)
        return self._inverse(z)

    def score(self, X: Any, y: np.ndarray) -> float:
        """Compositional skill score on the simplex (default ``score_metric='aitchison'``).

        The inherited ``RegressorMixin.score`` is the Euclidean R^2, which is meaningless for compositions (Euclidean
        distance ignores the constant-sum constraint and the relative scale of the parts). The Aitchison score is the
        natural analogue: ``1 - SS_aitch(prediction) / SS_aitch(baseline)`` where the squared Aitchison distances are
        taken against the closed geometric-mean composition (the simplex "mean"). 1.0 is perfect; <= 0 means no better
        than predicting that constant composition -- exactly the R^2 interpretation, transported to the simplex. Set
        ``score_metric='euclidean'`` to recover the (meaningless-here but sklearn-contract) Euclidean R^2.
        """
        if getattr(self, "score_metric", "aitchison") == "euclidean":
            return float(super().score(X, y))
        y_arr = np.asarray(y, dtype=np.float64)
        y_safe = multiplicative_zero_replacement(_close(y_arr), self.zero_delta)
        pred = self.predict(X)
        d_pred = aitchison_distance(pred, y_safe, delta=self.zero_delta)
        gmean = _close(np.exp(np.log(y_safe).mean(axis=0, keepdims=True)))
        baseline = np.repeat(gmean, y_safe.shape[0], axis=0)
        d_base = aitchison_distance(baseline, y_safe, delta=self.zero_delta)
        ss_pred = float(np.sum(np.square(d_pred)))
        ss_base = float(np.sum(np.square(d_base)))
        return 1.0 - ss_pred / ss_base if ss_base > 0.0 else 0.0
