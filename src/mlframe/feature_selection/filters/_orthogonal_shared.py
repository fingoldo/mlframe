"""Shared helpers for the ``_orthogonal_*_fe.py`` layer family (adaptive-degree, bootstrap-MI,
cluster-basis, CMIM, diff-basis, JMIM, quadruplet, routing, three-gate-MI, total-correlation,
triplet, ...): small utilities independently duplicated across those modules, consolidated here
so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np

_CODE_TO_BASIS = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}


def quantile_bin_batched(arr: np.ndarray, nbins: int) -> np.ndarray:
    """Vectorised equi-frequency bin of a 2-D (n, k) all-finite float array.

    Computes ``np.quantile(arr, qs, axis=0)`` ONCE for the whole batch (the underlying
    partition-based selector amortises across columns much better than ``k`` separate
    ``np.quantile(col, qs)`` calls). Then a per-column dedup + ``np.searchsorted`` produces
    dense int64 bin codes matching the contract of ``_quantile_bin`` on the all-finite path:
    the same edges (after ``np.unique`` dedup) and the same ``side='right'`` searchsorted
    convention. Bit-equivalent to ``_quantile_bin`` on all-finite numeric input; the
    per-column fallback handles mixed-NaN / Inf data via the original per-column path."""
    n, k = arr.shape
    out = np.zeros((n, k), dtype=np.int64)
    if n == 0 or k == 0:
        return out
    qs = np.linspace(0.0, 1.0, int(nbins) + 1)
    edges_all = np.quantile(arr, qs, axis=0)  # shape (nbins+1, k)
    for j in range(k):
        col_edges = np.unique(edges_all[:, j])
        if col_edges.size <= 2:
            if col_edges.size == 2:
                out[:, j] = (arr[:, j] >= col_edges[1]).astype(np.int64)
            continue
        inner = col_edges[1:-1]
        out[:, j] = np.searchsorted(inner, arr[:, j], side="right").astype(np.int64)
    return out


def noise_aware_floor(values: np.ndarray, sigma_thresh: float) -> float:
    """Median + sigma * 1.4826 * MAD noise floor used by the Layer 22/56 pair pipelines.
    Returns 0 when too few values to estimate robustly."""
    if values.size < 4:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med + sigma_thresh * 1.4826 * mad


def parse_code_deg_with_basis(s: str):
    """Parse a leg-code token like ``"He3"``/``"LL2"``/``"T1"``/``"L4"`` into ``(basis_name, degree)``,
    checking two-letter codes before single-letter ones so ``"LL"`` isn't mis-parsed as ``"L"``; returns
    ``(None, None)`` when ``s`` doesn't match any known code prefix.

    Distinct from ``_orthogonal_univariate_fe/_gpu_resident_cross_basis.py``'s ``_parse_code_deg``, which
    returns only the degree (int) and discards the basis code -- that device-builder variant deliberately
    ignores the code since the GPU leg spec re-routes basis via ``basis_route_by_moments``. This one is for
    the host-side generators that need the parsed basis name too.
    """
    for code in ("LL", "He", "T", "L"):
        if s.startswith(code):
            rest = s[len(code) :]
            if rest.isdigit():
                return _CODE_TO_BASIS[code], int(rest)
    return None, None


def coerce_y_classif(y) -> np.ndarray:
    """Dense int64 class labels for MI estimators.

    Integer dtypes pass straight through. Non-integer y (float / continuous /
    categorical) is DENSIFIED via ``np.unique(return_inverse=...)`` rather than
    truncated with ``.astype(int64)``: plain truncation merges distinct labels
    (1.2 and 1.8 -> 1) and destroys continuous-y signal entirely (every value
    in [0, 1) collapses to 0). The dense-rank mapping preserves every distinct
    value as its own class, which is the contract the MI estimator expects.
    """
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64, copy=False)
