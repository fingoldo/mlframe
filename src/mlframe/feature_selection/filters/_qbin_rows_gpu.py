"""Row-batched equi-frequency binning on the device: R resident columns of equal length binned in one pass.

``_sync_free_qbin_codes`` (``_mi_greedy_cmi_fe_binning``) bins ONE resident column with about 200 tiny launches (sort, edge interpolation, duplicate / extreme-edge exclusion, ``searchsorted``).
A caller that bins many short columns (the cross-fold stability vote: recipes x folds columns of n / K rows) paid that per column and was launch bound. ``qbin_codes_rows`` makes the same decisions
for a ``(R, n)`` matrix with every step along axis 1: one segmented sort, the numpy symmetric-lerp edges, the same exclusion rule (edges equal to the row minimum, adjacent duplicates, edges equal
to the row maximum when an interior distinct value exists -> +inf), and the bin of a value is the count of surviving edges not above it (``searchsorted(..., side="right")``), so the codes equal
those of the column-at-a-time binner. Rows are processed in blocks of at most ``ROW_BLOCK_BYTES`` of the ``(R, n, nbins + 1)`` comparison tensor, so the working set is bounded.
"""

from __future__ import annotations

from typing import Any

from ._device_sort_safe import device_sort

ROW_BLOCK_BYTES = 64 * 1024 * 1024


def _edges_rows(cp: Any, xs: Any, nbins: int) -> Any:
    """Sorted surviving interior edges of every row of the row-sorted ``xs`` (excluded edges are +inf, at the tail)."""
    n = int(xs.shape[1])
    qs = cp.linspace(0.0, 100.0, int(nbins) + 1)
    pos = qs / 100.0 * (n - 1)
    lo = cp.floor(pos).astype(cp.int64)
    hi = cp.minimum(lo + 1, n - 1)
    frac = pos - lo
    xlo, xhi = xs[:, lo], xs[:, hi]
    diff = xhi - xlo
    e = cp.where(frac >= 0.5, xhi - diff * (1 - frac), xlo + diff * frac)  # (R, nbins + 1), ascending per row
    dup = cp.zeros(e.shape, dtype=bool)
    dup[:, 1:] = e[:, 1:] == e[:, :-1]
    emin = e[:, :1]
    emax = e[:, -1:]
    has_interior = cp.any((e > emin) & (e < emax), axis=1, keepdims=True)
    excl = (e == emin) | dup | (has_interior & (e == emax))
    return device_sort(cp.where(excl, cp.inf, e), axis=1)


def qbin_codes_rows(cp: Any, x: Any, nbins: int) -> Any:
    """Equi-frequency bin codes (R, n) int64 of the finite resident ``(R, n)`` float matrix ``x``, row by row, equal to ``_sync_free_qbin_codes`` of each row."""
    r, n = int(x.shape[0]), int(x.shape[1])
    per_row = max(1, n * (int(nbins) + 1))
    block = max(1, ROW_BLOCK_BYTES // per_row)
    out = cp.empty((r, n), dtype=cp.int64)
    for s in range(0, r, block):
        xb = x[s : s + block]
        e2 = _edges_rows(cp, device_sort(xb, axis=1), nbins)
        # numpy compares the values with the float64 edges in float64: the comparison is made in the edge dtype
        out[s : s + block] = (e2[:, None, :] <= xb.astype(e2.dtype, copy=False)[:, :, None]).sum(axis=2)
    return out
