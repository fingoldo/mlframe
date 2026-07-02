"""LENKOR: a learned metric as a coordinate-descent-tuned deformed combination of per-block similarities (PZAD recsys).

Dyakonov's LENKOR technique (Бизнес-Информатика 2012 №1(19) pp.32-39; ECML-PKDD 2011 Discovery Challenge, 1st
place) builds a similarity between two objects as a DEFORMED weighted combination of per-attribute-block
sub-similarities: ``sim(x, z) = f( Σ_b c_b · sim_b(x, z) )``, where each ``sim_b`` compares one block (authors,
category, title, co-view counts, ...), the weights ``c_b`` are tuned by coordinate descent DIRECTLY on the target
functional, and ``f`` is an optional nonlinear deformation (nesting sqrt's, slide 96). Unlike Mahalanobis metric
learning (NCA/LMNN, which learn a linear map on raw features), this learns to COMBINE already-computed heterogeneous
similarity matrices — the natural tool when "how to compare projects is unclear, but how to compare their authors /
sponsors / fields is clear" (slide 84).

`fit_composite_similarity(block_sims, y)` coordinate-descent-tunes the block weights to maximize a leave-one-out
kNN prediction metric over the combined similarity, and returns the weights plus a `combine` to build the metric for
new query-vs-bank blocks. Reusable well beyond recsys: any task with several precomputed similarity views to blend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

try:
    import numba

    _HAS_NUMBA = True
except Exception:  # numba is an optional accelerator
    _HAS_NUMBA = False

logger = logging.getLogger(__name__)

__all__ = ["fit_composite_similarity", "combine_block_similarities", "CompositeSimilarityResult", "DEFORMATIONS"]

DEFORMATIONS = ("linear", "sqrt")


def _knn_loo_predict_impl(S, y, k, classification):
    """Leave-one-out kNN prediction from a similarity matrix: each row predicted from its top-k most-similar OTHERS."""
    n = S.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # find the k largest similarities among j != i (simple selection; k and n are modest here)
        best_idx = np.full(k, -1, dtype=np.int64)
        best_val = np.full(k, -np.inf, dtype=np.float64)
        for j in range(n):
            if j == i:
                continue
            v = S[i, j]
            # insert v into the running top-k (find current minimum slot)
            m = 0
            for t in range(1, k):
                if best_val[t] < best_val[m]:
                    m = t
            if v > best_val[m]:
                best_val[m] = v
                best_idx[m] = j
        wsum = 0.0
        acc = 0.0
        for t in range(k):
            j = best_idx[t]
            if j < 0:
                continue
            w = best_val[t]
            if w <= 0.0:
                w = 1e-12  # keep non-positive similarities from zeroing the vote
            wsum += w
            acc += w * y[j]
        if wsum > 0.0:
            out[i] = acc / wsum
    if classification:
        for i in range(n):
            out[i] = 1.0 if out[i] >= 0.5 else 0.0
    return out


if _HAS_NUMBA:
    _knn_loo_predict_impl = numba.njit(cache=True, nogil=True)(_knn_loo_predict_impl)


@dataclass
class CompositeSimilarityResult:
    """Result of :func:`fit_composite_similarity`."""

    weights: np.ndarray
    score: float
    deformation: str
    k: int
    classification: bool
    history: list = field(default_factory=list)

    def combine(self, block_sims) -> np.ndarray:
        """Build the combined similarity from per-block similarity matrices using the fitted weights."""
        return combine_block_similarities(block_sims, self.weights, self.deformation)


def combine_block_similarities(block_sims, weights, deformation: str = "linear") -> np.ndarray:
    """Weighted (optionally sqrt-deformed) combination of per-block similarity matrices."""
    w = np.asarray(weights, dtype=np.float64)
    S = np.zeros_like(np.asarray(block_sims[0], dtype=np.float64))
    for b in range(len(block_sims)):
        S = S + w[b] * np.asarray(block_sims[b], dtype=np.float64)
    if deformation == "sqrt":
        S = np.sqrt(np.clip(S, 0.0, None))
    elif deformation != "linear":
        raise ValueError(f"deformation must be one of {DEFORMATIONS}, got {deformation!r}.")
    return S


def _default_metric(classification):
    if classification:
        def acc(y, yhat):
            return float(np.mean(y == yhat))
        return acc, True
    def neg_mse(y, yhat):
        return -float(np.mean((y - yhat) ** 2))
    return neg_mse, True


def fit_composite_similarity(
    block_sims,
    y,
    *,
    k: int = 10,
    metric=None,
    greater_is_better: bool = True,
    classification: bool | None = None,
    deformation: str = "linear",
    grid=None,
    max_iter: int = 15,
) -> CompositeSimilarityResult:
    """Tune per-block similarity weights by coordinate descent to maximize a leave-one-out kNN prediction metric.

    Parameters
    ----------
    block_sims : sequence of ``(n, n)`` similarity matrices (higher = more similar), one per attribute block.
    y : ``(n,)`` target aligned to the similarity rows.
    k : neighbours used in the leave-one-out kNN prediction.
    metric : ``metric(y_true, y_pred) -> float``; default is accuracy (classification) or negative MSE (regression).
    classification : force task type; auto-detected from ``y`` (<=10 unique integer-like values) when ``None``.
    deformation : ``'linear'`` or ``'sqrt'`` (elementwise sqrt of the weighted sum, LENKOR's diminishing-returns form).
    grid : candidate values tried for each weight per coordinate sweep; default a geometric grid ``[0,.25,.5,1,2,4,8]``.

    Returns a :class:`CompositeSimilarityResult` with the fitted ``weights``, the achieved ``score``, and ``combine``.
    """
    sims = [np.ascontiguousarray(s, dtype=np.float64) for s in block_sims]
    yy = np.ascontiguousarray(y).astype(np.float64).ravel()
    n = yy.shape[0]
    if any(s.shape != (n, n) for s in sims):
        raise ValueError("fit_composite_similarity: every block similarity must be an (n, n) matrix matching y.")
    if deformation not in DEFORMATIONS:
        raise ValueError(f"deformation must be one of {DEFORMATIONS}, got {deformation!r}.")
    if k < 1 or k >= n:
        raise ValueError("fit_composite_similarity: require 1 <= k < n.")
    if classification is None:
        uniq = np.unique(yy)
        classification = uniq.shape[0] <= 10 and np.all(uniq == np.round(uniq))
    if metric is None:
        metric, greater_is_better = _default_metric(classification)
    grid = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] if grid is None else grid, dtype=np.float64)
    sign = 1.0 if greater_is_better else -1.0

    def score(w):
        S = combine_block_similarities(sims, w, deformation)
        pred = _knn_loo_predict_impl(S, yy, k, classification)
        return sign * metric(yy, pred)

    K = len(sims)
    w = np.ones(K, dtype=np.float64)
    best = score(w)
    history = [best]
    for _ in range(max_iter):
        improved = False
        for j in range(K):
            for cand in grid:
                w2 = w.copy()
                w2[j] = cand
                if w2.sum() <= 0:
                    continue
                s = score(w2)
                if s > best + 1e-12:
                    best = s
                    w = w2
                    improved = True
        history.append(best)
        if not improved:
            break
    return CompositeSimilarityResult(
        weights=w, score=sign * best, deformation=deformation, k=k, classification=classification, history=history
    )
