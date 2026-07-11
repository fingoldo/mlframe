"""``magnitude_sample_weight``: derive a per-row sample weight from multiple correlated regression targets.

Source: 1st_jane-street-market-prediction.md -- used the mean of absolute values of multiple ``resp_*``
regression targets as ``sample_weight`` when training a multi-label classifier on BINARIZED action labels, to
focus training on high-magnitude/high-conviction samples. Whenever a business proxy label is a threshold/sign
binarization of an underlying continuous signal, the binarization throws away magnitude information that was
still genuinely informative about how much that row's outcome mattered -- reintroducing it as a sample weight
lets the classifier prioritize getting high-conviction rows right over borderline near-zero ones (where the
binary label is close to a coin flip regardless of what the model predicts).
"""
from __future__ import annotations

import numpy as np


def magnitude_sample_weight(y_multi: np.ndarray, norm: str = "mean_abs", robust: bool = False, winsor_quantile: float = 0.99) -> np.ndarray:
    """Per-row magnitude summary across multiple correlated regression target columns.

    Parameters
    ----------
    y_multi
        ``(n_samples,)`` (single target) or ``(n_samples, n_targets)`` (multiple correlated targets, e.g.
        Jane Street's ``resp_1..resp_4``).
    norm
        ``"mean_abs"`` (mean of ``|y|`` across target columns -- the source's own choice), ``"max_abs"``
        (worst-case single-target conviction), or ``"l2"`` (Euclidean norm across target columns).
    robust
        If ``True``, winsorize each target column's ``|y|`` at ``winsor_quantile`` before applying ``norm``,
        so a handful of extreme-magnitude rows (data glitches, fat-tailed events) cannot dominate the total
        weight mass -- a plain ``mean_abs``/``max_abs``/``l2`` norm is unbounded and a single huge-magnitude
        row can outweigh hundreds of genuinely-informative high-conviction rows combined. Default ``False``
        keeps the original unbounded behavior bit-identical for existing callers.
    winsor_quantile
        Per-column clip quantile applied to ``|y|`` when ``robust=True`` (default ``0.99`` -- clip the top 1%
        of each column's magnitude down to the 99th percentile value). Ignored when ``robust=False``.

    Returns
    -------
    np.ndarray
        ``(n_samples,)`` non-negative weights, one per row.
    """
    arr = np.atleast_2d(np.asarray(y_multi, dtype=np.float64))
    if np.asarray(y_multi).ndim == 1:
        arr = arr.T  # atleast_2d on a 1D array makes it (1, n) -- transpose to (n, 1)

    abs_arr = np.abs(arr)
    if robust:
        if not 0.0 < winsor_quantile <= 1.0:
            raise ValueError(f"magnitude_sample_weight: winsor_quantile must be in (0, 1], got {winsor_quantile!r}")
        caps = np.quantile(abs_arr, winsor_quantile, axis=0, keepdims=True)  # per-column cap, shape (1, n_targets)
        abs_arr = np.minimum(abs_arr, caps)

    if norm == "mean_abs":
        return np.asarray(np.mean(abs_arr, axis=1))
    elif norm == "max_abs":
        return np.asarray(np.max(abs_arr, axis=1))
    elif norm == "l2":
        return np.asarray(np.sqrt(np.sum(abs_arr**2, axis=1)))
    raise ValueError(f"magnitude_sample_weight: unsupported norm {norm!r}, expected 'mean_abs', 'max_abs', or 'l2'")


__all__ = ["magnitude_sample_weight"]
