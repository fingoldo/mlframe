"""Statistics shared by the FE operators that accept a candidate on a held-out MI gain over a baseline.

A fixed margin on the gain cannot follow n, noise or the shape of the target. The bar used instead is the standard error of the PAIRED difference of two plug-in MIs on the same rows: the
plug-in MI is the mean of the per-row pointwise MI ``log p(x, y) / (p(x) p(y))``, so the difference of two features' MIs is the mean of the per-row differences and its standard error is the
spread of those differences over sqrt(n) (delta method). Bin edges for the held-out rows come from the other half of the rows, never from the rows scored.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["has_few_distinct", "quantile_codes", "pointwise_mi", "held_out_codes", "paired_gain_se", "plugin_mi_of_codes"]

N_BINS = 10


@njit(cache=True, nogil=True)
def _codes_right(x, edges):  # pragma: no cover - numba body
    """``np.searchsorted(edges, x, side="right")`` for ascending ``edges``: the number of edges that are <= each value (same comparisons, no promotion copy of ``x``)."""
    out = np.empty(x.shape[0], dtype=np.int64)
    m = edges.shape[0]
    for i in range(x.shape[0]):
        v = x[i]
        lo = 0
        hi = m
        while lo < hi:
            mid = (lo + hi) >> 1
            if edges[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo
    return out


@njit(cache=True, nogil=True)
def _more_than_distinct(v, cap):  # pragma: no cover - numba body
    """True when the finite array ``v`` holds more than ``cap`` distinct values; stops at the ``cap + 1``-th distinct value, so a continuous column costs ``cap + 1`` reads, not a sort."""
    seen = np.empty(cap + 1, dtype=np.float64)
    k = 0
    for i in range(v.shape[0]):
        x = v[i]
        dup = False
        for j in range(k):
            if seen[j] == x:
                dup = True
                break
        if not dup:
            if k == cap:
                return True
            seen[k] = x
            k += 1
    return False


def has_few_distinct(v: np.ndarray, cap: int) -> bool:
    """``np.unique(v).size <= cap`` for an array without NaN, without sorting it (the early exit of :func:`_more_than_distinct`)."""
    return not bool(_more_than_distinct(np.ascontiguousarray(v), int(cap)))


def quantile_codes(x: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-frequency bin codes of ``x`` in ``[0, n_bins)`` (ties share a bin; non-finite values go to the lowest bin)."""
    x = np.where(np.isfinite(x), x, -np.inf)
    finite = x[np.isfinite(x)]
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]) if finite.size else np.zeros(n_bins - 1)
    codes: np.ndarray = _codes_right(np.ascontiguousarray(x), np.ascontiguousarray(edges))
    return codes


@njit(cache=True, nogil=True)
def _pointwise_mi_njit(codes, n_codes, y_codes, ky, out):
    """Fill ``out`` with the per-row pointwise MI; ``False`` (nothing usable in ``out``) when a code lies outside its range, so the caller takes the numpy form and its own error."""
    n_rows = codes.shape[0]
    joint = np.zeros((n_codes, ky), dtype=np.float64)  # Intentional float64: joint-count matrix, exact integer-scale accumulation
    for i in range(n_rows):
        c = codes[i]
        t = y_codes[i]
        if c < 0 or c >= n_codes or t < 0 or t >= ky:
            return False
        joint[c, t] += 1.0
    px = np.zeros(n_codes, dtype=np.float64)  # Intentional float64: marginal counts
    py = np.zeros(ky, dtype=np.float64)  # Intentional float64: marginal counts
    for a in range(n_codes):
        for b in range(ky):
            px[a] += joint[a, b]
            py[b] += joint[a, b]
    n = 0.0
    for a in range(n_codes):
        n += px[a]
    table = np.zeros((n_codes, ky), dtype=np.float64)  # Intentional float64: the log-ratio of each occupied cell
    for a in range(n_codes):
        for b in range(ky):
            if joint[a, b] > 0.0:
                table[a, b] = np.log(joint[a, b] * n / (px[a] * py[b]))
    for i in range(n_rows):
        out[i] = table[codes[i], y_codes[i]]
    return True


def _pointwise_mi_numpy(codes: np.ndarray, n_codes: int, y_codes: np.ndarray, ky: int) -> np.ndarray:
    """The numpy form of :func:`pointwise_mi` (the reference of the compiled one, and the route of inputs it does not take)."""
    joint = np.bincount(codes * ky + y_codes, minlength=n_codes * ky).reshape(n_codes, ky).astype(np.float64)  # Intentional float64: plug-in MI joint-count matrix; exact integer-scale accumulation feeding entropy
    n = joint.sum()
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        table = np.log(joint * n / (px @ py))
    return np.asarray(np.where(joint > 0, table, 0.0)[codes, y_codes])


def pointwise_mi(codes: np.ndarray, n_codes: int, y_codes: np.ndarray, ky: int) -> np.ndarray:
    """Per-row pointwise mutual information of integer ``codes`` (in ``[0, n_codes)``) and ``y_codes`` (in ``[0, ky)``); its mean is the plug-in MI in nats.

    One compiled pass (counts, the table of occupied cells, the gather) without the interpreter lock; the numpy form made five full-length temporaries per call and a fit calls this about a
    thousand times from the column pools. Same counts and the same per-cell arithmetic; the values agree with the numpy form to the last bit or two of the logarithm."""
    c = np.asarray(codes)
    t = np.asarray(y_codes)
    if c.ndim == 1 and c.shape == t.shape and c.dtype.kind in "iu" and t.dtype.kind in "iu" and c.size:
        out = np.empty(c.shape[0], dtype=np.float64)  # Intentional float64: per-row pointwise MI (statistic vector)
        if _pointwise_mi_njit(np.ascontiguousarray(c), int(n_codes), np.ascontiguousarray(t), int(ky), out):
            return out
    return _pointwise_mi_numpy(codes, n_codes, y_codes, ky)


def plugin_mi_of_codes(codes: np.ndarray, n_codes: int, y_codes: np.ndarray, ky: int) -> float:
    """Plug-in MI (nats) of integer ``codes`` and ``y_codes``."""
    return float(pointwise_mi(codes, n_codes, y_codes, ky).mean())


def held_out_codes(feature: np.ndarray, fit_rows: np.ndarray, score_rows: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bin codes of ``feature[score_rows]`` using equal-frequency edges taken from ``feature[fit_rows]`` only."""
    edges = np.quantile(feature[fit_rows], np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.asarray(np.searchsorted(edges, feature[score_rows], side="right"), dtype=np.int64)


def paired_gain_se(cand_codes: np.ndarray, base_codes: np.ndarray, y_codes: np.ndarray, ky: int, n_codes: int = N_BINS) -> float:
    """``(gain, standard error)`` is returned as the standard error alone: the spread of the per-row difference of the candidate's and the baseline's pointwise MIs over sqrt(n). The gain itself is
    ``plugin_mi_of_codes(cand) - plugin_mi_of_codes(base)`` on the same rows."""
    d = pointwise_mi(cand_codes, n_codes, y_codes, ky) - pointwise_mi(base_codes, n_codes, y_codes, ky)
    return float(d.std() / np.sqrt(len(d)))
