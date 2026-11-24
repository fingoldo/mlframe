"""Quantile binning and plug-in mutual information (nats) on rank-based bins; the numpy reference used by the statistics scripts."""

from __future__ import annotations

import numpy as np


def qbin(x, k=10):
    """Equal-frequency bin codes in ``[0, k)`` from the stable rank of the finite values of ``x``.

    Non-finite values (NaN) are not ranked, so they cannot take a real bin: they get the dedicated code ``k`` (11-F4). With no non-finite
    values the codes are exactly the ``[0, k)`` equal-frequency bins.
    """
    x = np.asarray(x, dtype=np.float64)
    codes = np.full(x.shape[0], k, np.int64)
    idx = np.flatnonzero(np.isfinite(x))
    m = idx.size
    if m:
        r = np.empty(m, np.int64)
        r[np.argsort(x[idx], kind="stable")] = np.arange(m)  # ordinal-rank-ok: equal-frequency bins need distinct ordinal ranks
        codes[idx] = np.minimum((r * k) // m, k - 1)
    return codes


def mi_b(xb, yb, kx=None, ky=None, mm=False):
    """Plug-in MI of two integer code vectors; ``mm`` subtracts the Miller-Madow bias ``(kx-1)(ky-1)/(2n)``."""
    kx = kx if kx is not None else xb.max() + 1
    ky = ky if ky is not None else yb.max() + 1
    jc = np.bincount(xb * ky + yb, minlength=kx * ky).reshape(kx, ky).astype(float)
    n = jc.sum()
    p = jc / n
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    v = float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
    if mm:
        v -= ((px > 0).sum() - 1) * ((py > 0).sum() - 1) / (2 * n)
    return v


def mi(x, yb, k=10, mm=False):
    """MI between the ``k``-quantile bins of a continuous ``x`` and target codes ``yb``."""
    return mi_b(qbin(x, k), yb, k, yb.max() + 1, mm)
