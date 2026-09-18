"""Detect composite-target specs whose fitted T column is redundant before the expensive per-spec training.

Two specs (or a spec and raw y) whose T columns are affine maps of each other train equivalent models under squared loss:
the inner model learns the same function up to a fixed scale/shift, and the inverse maps it back to the same y-scale
prediction. Production examples: ``addres`` with fitted slope 1 vs ``diff`` on the same base; ``medres`` / ``linresR``
whose fitted base term is ~0 on a zero-inflated target (T == y - const). Each such spec costs a full model-zoo fit.

The check is generic (works on the materialised T, not on per-transform parameters): standardise every column on the
train rows and flag a pair when the squared Pearson correlation is within ``r2_tol`` of 1.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_R2_TOL = 1e-4


def t_train_envelope(t_train: np.ndarray) -> tuple[float, float] | None:
    """Same T-clip envelope ``CompositeTargetEstimator.fit`` derives: median(T) +/- 10*MAD, widened to the observed range."""
    t = np.asarray(t_train, dtype=np.float64).reshape(-1)
    t = t[np.isfinite(t)]
    if t.size < 10:
        return None
    med = float(np.median(t))
    mad = float(np.median(np.abs(t - med)))
    lo, hi = float(t.min()), float(t.max())
    if mad > 0:
        lo, hi = min(med - 10.0 * mad, lo), max(med + 10.0 * mad, hi)
    if hi <= lo:
        return None
    return lo, hi


def _standardise(col: np.ndarray) -> np.ndarray | None:
    c = col - col.mean()
    sd = float(np.sqrt(np.mean(c * c)))
    if not np.isfinite(sd) or sd <= 1e-12 * max(1.0, float(np.max(np.abs(col)))):
        return None
    return c / sd


def find_equivalent_composite_specs(
    y_train: np.ndarray,
    t_by_name: dict[str, np.ndarray],
    priority: Sequence[str],
    *,
    r2_tol: float = DEFAULT_R2_TOL,
    max_rows: int = 200_000,
    seed: int = 0,
) -> dict[str, str]:
    """Return ``{spec_name: reason}`` for specs to drop.

    ``t_by_name`` holds each spec's T on the train rows (aligned with ``y_train``); ``priority`` lists spec names best-first,
    so of an equivalent pair the earlier one is kept. A spec is dropped when its T is constant, affine in raw y, or affine
    in the T of an already-kept spec.
    """
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    names = [n for n in priority if n in t_by_name]
    if not names or y.size == 0:
        return {}
    mask = np.isfinite(y)
    cols = {}
    for n in names:
        t = np.asarray(t_by_name[n], dtype=np.float64).reshape(-1)
        if t.shape != y.shape:
            continue
        cols[n] = t
        mask &= np.isfinite(t)
    rows = np.flatnonzero(mask)
    if rows.size < 10:
        return {}
    if rows.size > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=max_rows, replace=False))
    thr = 1.0 - float(r2_tol)
    z_y = _standardise(y[rows])
    drops: dict[str, str] = {}
    kept: list[tuple[str, np.ndarray]] = []
    for n in names:
        if n not in cols:
            continue
        z = _standardise(cols[n][rows])
        if z is None:
            drops[n] = "fitted T is constant on the train rows (prediction would be a constant)"
            continue
        if z_y is not None:
            r = float(np.mean(z * z_y))
            if r * r >= thr:
                drops[n] = f"fitted T is an affine map of raw y (r={r:+.6f}); equivalent to training on the raw target"
                continue
        dup = None
        for kn, kz in kept:
            r = float(np.mean(z * kz))
            if r * r >= thr:
                dup = (kn, r)
                break
        if dup is not None:
            drops[n] = f"fitted T is an affine map of kept spec '{dup[0]}' (r={dup[1]:+.6f}); equivalent model"
            continue
        kept.append((n, z))
    return drops
