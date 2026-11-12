"""Shared helpers for the ``_orthogonal_*_fe.py`` layer family (adaptive-degree, bootstrap-MI,
cluster-basis, CMIM, diff-basis, JMIM, quadruplet, routing, three-gate-MI, total-correlation,
triplet, ...): small utilities independently duplicated across those modules, consolidated here
so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

from ._basis_name_codes import BASIS_NAME_CODES, CODE_TO_BASIS

from typing import cast
from typing import Any

import numpy as np

_CODE_TO_BASIS = CODE_TO_BASIS


def _full_edge_codes(arr: np.ndarray, nbins: int, out: np.ndarray) -> np.ndarray:
    """Fill ``out`` with the bin codes of the columns whose ``nbins + 1`` quantile edges are all distinct, from the compiled (device when large float32) quantile-codes block, and return which columns that was.

    With every edge distinct the dedup and the two-edge special case below do nothing, so the codes equal the numpy path's; a column with tied edges is left to it."""
    from ._numba_utils import quantile_codes_block
    from ._quantile_codes_gpu import quantile_codes_block_auto

    v = np.ascontiguousarray(arr.T)
    codes = np.empty(v.shape, dtype=np.int16)
    nx = np.empty(v.shape[0], dtype=np.int64)
    quantile_codes_block_auto(v, nbins, codes, nx, host_kernel=quantile_codes_block)
    full: np.ndarray = nx == nbins
    out[:, full] = codes[full].T
    return full


def mean_fill_nonfinite(x: np.ndarray) -> None:
    """Replace the non-finite entries of ``x`` by the mean of its finite ones (0 when there are none), in place."""
    finite_mask = np.isfinite(x)
    if not finite_mask.all():
        fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
        np.copyto(x, np.where(finite_mask, x, fill))


def leg_eng_col_name(cols: "tuple[str, ...]", code: str, degs: "tuple[int, ...]") -> str:
    """``"{col_1}*...*{col_n}__{code}{deg_1}_..._{code}{deg_n}"``: the stable name of an engineered product of ``n`` basis legs (``code`` is the leg-1 basis code)."""
    return "*".join(cols) + "__" + "_".join(f"{code}{d}" for d in degs)


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
    done = _full_edge_codes(arr, int(nbins), out)
    if done.all():
        return out
    cols = np.flatnonzero(~done)
    arr = arr[:, cols]
    out_all = out
    out = np.zeros((n, cols.size), dtype=np.int64)
    k = cols.size
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
    out_all[:, cols] = out
    return out_all


def noise_aware_floor(values: np.ndarray, sigma_thresh: float) -> float:
    """Median + sigma * 1.4826 * MAD noise floor used by the Layer 22/56 pair pipelines.
    Returns 0 when too few values to estimate robustly."""
    if values.size < 4:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med + sigma_thresh * 1.4826 * mad


def parse_code_deg_with_basis(s: str) -> "tuple[str | None, int | None]":
    """Parse a leg-code token like ``"He3"``/``"LL2"``/``"T1"``/``"L4"`` into ``(basis_name, degree)``,
    checking two-letter codes before single-letter ones so ``"LL"`` isn't mis-parsed as ``"L"``; returns
    ``(None, None)`` when ``s`` doesn't match any known code prefix.

    Distinct from ``_orthogonal_univariate_fe/_gpu_resident_cross_basis.py``'s ``_parse_code_deg``, which
    returns only the degree (int) and discards the basis code -- that device-builder variant deliberately
    ignores the code since the GPU leg spec re-routes basis via ``basis_route_by_moments``. This one is for
    the host-side generators that need the parsed basis name too.
    """
    for code in BASIS_NAME_CODES:
        if s.startswith(code):
            rest = s[len(code) :]
            if rest.isdigit():
                return _CODE_TO_BASIS[code], int(rest)
    return None, None


def coerce_y_classif(y: "np.ndarray | Any") -> np.ndarray:
    """Dense int64 class labels for MI estimators.

    Integer dtypes pass straight through. A float y with few distinct values (at most ``FEW_CLASSES_MAX``) is categorical and is DENSIFIED via
    ``np.unique(return_inverse=...)`` rather than truncated with ``.astype(int64)`` (truncation merges 1.2 and 1.8 and collapses every value in [0, 1)
    to one class). A float y with more distinct values is a continuous target and is quantile-binned into 10 classes, as every other gate does
    (``_unified_fe_gate._coerce_y_classes``): giving each distinct value its own class makes nearly every row a class of its own, and the plug-in MI of
    ANY binned feature with such a label equals the feature's own entropy (ln of the bin count), so the bootstrap-MI uplift of every engineered
    column over its source was about 1 and the families scored this way (modular, decomposition, orthogonal bases) saw no signal on a regression target.
    Other dtypes (strings, categoricals) are densified per distinct value.
    """
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        from ._unified_fe_gate import _coerce_y_classes

        return cast("np.ndarray", _coerce_y_classes(arr).astype(np.int64, copy=False))
    _, inv = np.unique(arr, return_inverse=True)
    return cast("np.ndarray", inv.astype(np.int64, copy=False))
