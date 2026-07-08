"""Weighted Cohen's kappa for ordinal / ranked multiclass targets (PZAD err_multirankcluster).

The multiclass lecture (Дьяконов 2020, slides 2-5) opens with Weighted Kappa: when the classes are
ORDERED (severity grade, star rating, ordinal survey), confusing class 1 with class 5 is worse than
confusing class 1 with class 2, so each off-diagonal confusion carries a weight ``w_ij``. Quadratic
weights ``w_ij = (i-j)^2 / (l-1)^2`` give the Quadratic Weighted Kappa (QWK) that ordinal Kaggle
competitions optimize. mlframe's `cohen_kappa_binary` handles only the unweighted 2-class case.

    kappa = 1 - sum(w_ij * O_ij) / sum(w_ij * E_ij)

where O is the confusion matrix and E is the expected (chance) matrix from the marginal products.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["quadratic_weighted_kappa", "weighted_kappa", "KAPPA_WEIGHTS"]

KAPPA_WEIGHTS = ("quadratic", "linear")


@njit(fastmath=False, cache=True, nogil=True)
def _confusion_matrix(y_true, y_pred, n):
    obs = np.zeros((n, n), dtype=np.float64)
    for k in range(y_true.shape[0]):
        obs[y_true[k], y_pred[k]] += 1.0
    return obs


def weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, *, weights: str = "quadratic", n_classes: int | None = None) -> float:
    """Weighted Cohen's kappa for integer-labeled ordinal classes ``0..n_classes-1``.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Integer class labels (ordinal). Values must lie in ``[0, n_classes)``.
    weights : {'quadratic', 'linear'}
        Penalty growth with class distance: quadratic ``(i-j)^2`` (QWK) or linear ``|i-j|``.
    n_classes : int, optional
        Number of ordinal classes. Defaults to ``max(labels)+1``.

    Returns
    -------
    float
        Kappa in ``[-1, 1]``; 1 = perfect agreement, 0 = chance, negative = worse than chance. Returns 1.0 when
        both raters are constant and agree, 0.0 when the expected-agreement denominator is degenerate.
    """
    if weights not in KAPPA_WEIGHTS:
        raise ValueError(f"weighted_kappa: weights must be one of {KAPPA_WEIGHTS}, got {weights!r}.")
    yt = np.ascontiguousarray(y_true).astype(np.int64)
    yp = np.ascontiguousarray(y_pred).astype(np.int64)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("weighted_kappa: y_true and y_pred length mismatch.")
    if yt.shape[0] == 0:
        return np.nan
    if yt.min() < 0 or yp.min() < 0:
        raise ValueError("weighted_kappa: labels must be non-negative integers.")
    n = int(n_classes) if n_classes is not None else int(max(yt.max(), yp.max())) + 1
    if n < 2:
        return 1.0  # single class, ratings trivially agree

    obs = _confusion_matrix(yt, yp, n)
    total = obs.sum()
    hist_true = obs.sum(axis=1)
    hist_pred = obs.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / total  # expected counts under independence, same total as obs

    idx = np.arange(n, dtype=np.float64)
    diff = np.abs(idx[:, None] - idx[None, :])
    W = (diff / (n - 1)) ** 2 if weights == "quadratic" else diff / (n - 1)

    num = float((W * obs).sum())
    den = float((W * E).sum())
    if den == 0.0:
        return 0.0
    return 1.0 - num / den


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int | None = None) -> float:
    """Quadratic Weighted Kappa (QWK): ``weighted_kappa(..., weights='quadratic')``. The standard ordinal-agreement metric."""
    return weighted_kappa(y_true, y_pred, weights="quadratic", n_classes=n_classes)
