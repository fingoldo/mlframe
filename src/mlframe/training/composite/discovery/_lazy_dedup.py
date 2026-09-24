"""Near-collinear dedup of ``x_remaining`` without the (rows x features) float plane, for the lazy prebin path.

The lazy prebin path pulls one sampled column at a time so the float matrix is never built, but the per-base dedup walked
that matrix, so the path was switched off whenever dedup was on (the default). Here the pairwise correlations come from a
Gram matrix accumulated over row blocks (``block x features`` float64 at a time), and every base's keep mask is the same
left-to-right walk as ``near_collinear_keep_mask`` run on those correlations. A pair whose estimate lies within
``_BAND`` of the threshold, or that touches a column with non-finite values (the walk masks rows per pair), is decided
exactly: its two columns are pulled again and scored by ``_ref_pair_corr`` in float64 on the jointly finite rows, the
arithmetic of the numba walk's borderline re-check. Exact decisions are memoised across bases.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._collinear_numba import _ref_pair_corr
from .screening import _extract_column_array

# The block Gram's |corr| differs from the two-pass per-pair value by about n x 1e-16; pairs closer than this to the
# threshold are re-decided exactly, so the band only trades exact re-checks for safety.
_BAND = 1e-7
_ROW_BLOCK = 8192


class StreamedCollinearity:
    """Pairwise near-collinearity of the sampled feature columns, computed without holding them all."""

    def __init__(self, df: Any, cols: Sequence[str], rows: np.ndarray, *, corr_threshold: float):
        self._df, self._cols, self._rows = df, list(cols), np.asarray(rows)
        self.thr = float(corr_threshold)
        n, f = self._rows.shape[0], len(self._cols)
        self.trivial = f < 2 or n < 3 or not (self.thr < 1.0)
        self._exact: dict = {}
        if self.trivial:
            return
        means = np.empty(f)
        finite = np.ones(f, dtype=bool)
        for j, c in enumerate(self._cols):
            col = np.asarray(_extract_column_array(df, c, rows=self._rows), dtype=np.float64)
            finite[j] = bool(np.isfinite(col).all())
            means[j] = col.mean() if finite[j] else 0.0
        gram = np.zeros((f, f))
        for start in range(0, n, _ROW_BLOCK):
            block_rows = self._rows[start:start + _ROW_BLOCK]
            block = np.empty((block_rows.shape[0], f))
            for j, c in enumerate(self._cols):
                block[:, j] = np.asarray(_extract_column_array(df, c, rows=block_rows), dtype=np.float64) - means[j]
            block[:, ~finite] = 0.0
            gram += block.T @ block
            del block
        var = np.diag(gram).copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.abs(gram) / np.sqrt(np.outer(var, var))
        # sure: the pair is decided by the estimate; exceeds: its decision when sure.
        low_var = var < 1e-12 * max(float(var.max()), 1.0)  # near-constant: the exact walk decides its skip
        self._sure = np.isfinite(corr) & (np.abs(corr - self.thr) > _BAND)
        self._sure &= finite[:, None] & finite[None, :]
        self._sure &= ~(low_var[:, None] | low_var[None, :])
        self._exceeds = np.where(self._sure, corr > self.thr, False)

    def _column(self, j: int) -> np.ndarray:
        return np.asarray(_extract_column_array(self._df, self._cols[j], rows=self._rows), dtype=np.float64)

    def _exceeds_exact(self, j: int, k: int) -> bool:
        key = (min(j, k), max(j, k))
        hit = self._exact.get(key)
        if hit is None:
            a, b = self._column(j), self._column(k)
            pair = np.isfinite(a) & np.isfinite(b)
            corr = _ref_pair_corr(a[pair], b[pair]) if int(pair.sum()) >= 3 else None
            hit = self._exact[key] = corr is not None and corr > self.thr
        return hit

    def keep_mask(self, drop_idx: Any) -> np.ndarray:
        """The dedup keep mask over the columns left after dropping ``drop_idx``, in their original order."""
        remaining = np.delete(np.arange(len(self._cols)), drop_idx)
        keep = np.ones(remaining.size, dtype=bool)
        if self.trivial or remaining.size < 2:
            return keep
        kept: list[int] = []
        for pos, j in enumerate(remaining):
            if kept:
                ks = np.asarray(kept)
                drop = bool(self._exceeds[j, ks].any())
                if not drop:
                    drop = any(self._exceeds_exact(int(j), int(k)) for k in ks[~self._sure[j, ks]])
                if drop:
                    keep[pos] = False
                    continue
            kept.append(int(j))
        return keep


__all__ = ["StreamedCollinearity"]
