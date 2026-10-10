"""Quantile binning and plug-in mutual information (nats) on rank-based bins; the numpy reference used by the statistics scripts."""

from __future__ import annotations

import numpy as np


def qbin(x, k=10):
    """Equal-frequency bin codes in ``[0, k)`` from the stable rank of ``x``."""
    n = len(x)
    r = np.empty(n, np.int64)
    r[np.argsort(x, kind="stable")] = np.arange(n)
    return np.minimum((r * k) // n, k - 1)


def mi_b(xb, yb, kx=None, ky=None, mm=False):
    """Plug-in MI of two integer code vectors; ``mm`` subtracts the Miller-Madow bias ``(kx-1)(ky-1)/(2n)``."""
    kx = kx or xb.max() + 1
    ky = ky or yb.max() + 1
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
