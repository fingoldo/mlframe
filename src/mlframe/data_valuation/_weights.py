"""Transform a per-row data-valuation vector (KNN-Shapley / TMC / Banzhaf) into a training ``sample_weight``.

Mirrors the shape/contract of ``feature_engineering/magnitude_sample_weight.py``: a plain ``(n,)``
non-negative float64 vector a caller feeds straight into ``model.fit(sample_weight=...)``.
"""

from __future__ import annotations

import numpy as np


def valuation_sample_weight(values: np.ndarray, *, mode: str = "clip_negative", floor: float = 0.0, temperature: float = 1.0) -> np.ndarray:
    """Convert raw per-row valuations into a ``(n,)`` non-negative ``sample_weight`` vector, mean ~= 1.

    Parameters
    ----------
    values
        ``(n,)`` raw valuations (e.g. from :func:`knn_shapley`) -- can be negative (harmful rows).
    mode
        ``"clip_negative"`` (RECOMMENDED default): ``w = max(values, floor)``, renormalized to mean 1.
        Least aggressive -- harmful rows drop toward ``floor`` (default 0, i.e. effectively excluded),
        everyone else keeps roughly sample-size-preserving weight near 1, so overall training dynamics
        (learning rate schedules, regularization tuned against ``n``) are not disrupted.
        ``"rank"``: weight by each row's PERCENTILE rank among all values (0..1, then renormalized to
        mean 1) -- ignores the raw value's magnitude/scale, useful when valuations are noisy and only
        the relative ordering is trustworthy.
        ``"softmax"``: ``w = softmax(values / temperature)`` renormalized to mean 1 -- most aggressive,
        can concentrate almost all weight on a few top rows at low ``temperature``; experimental.
    floor
        Minimum weight for ``mode="clip_negative"`` before renormalization (default 0.0 -- harmful rows
        get exactly excluded, not just down-weighted).
    temperature
        Softmax temperature for ``mode="softmax"`` (ignored otherwise); lower = more concentrated.

    Returns
    -------
    np.ndarray
        ``(n,)`` non-negative float64, mean ~= 1.0, no NaN -- verified before returning.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        return values

    if mode == "clip_negative":
        w = np.maximum(values, floor)
    elif mode == "rank":
        order = np.argsort(np.argsort(values))
        w = (order.astype(np.float64) + 1.0) / n  # percentile in (0, 1]
    elif mode == "softmax":
        z = values / temperature
        z = z - z.max()  # overflow-safe
        ez = np.exp(z)
        w = ez / ez.sum() * n  # mean 1 before the final renormalization below (already exact, but kept uniform)
    else:
        raise ValueError(f"valuation_sample_weight: unsupported mode {mode!r}, expected 'clip_negative', 'rank', or 'softmax'")

    total = w.sum()
    if total <= 0.0:
        # Degenerate: every value was <= floor (or all-zero softmax input) -- fall back to uniform
        # weight rather than dividing by zero / returning an all-NaN vector.
        return np.ones(n, dtype=np.float64)
    w = w / total * n  # renormalize to mean 1

    if np.any(np.isnan(w)) or np.any(w < 0.0):
        raise RuntimeError("valuation_sample_weight: produced NaN or negative weight -- this is a bug, not a data issue.")
    return np.asarray(w, dtype=np.float64)


__all__ = ["valuation_sample_weight"]
