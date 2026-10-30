"""Re-binning of a frame under permuted targets for a supervised ``nbins_strategy`` whose edges come from a small set per column (``optimal_joint``).

``categorize_dataset`` repeats, for every permutation of the target, the target-free work (the float copy of the frame, the NaN scan and fill) and re-encodes every column from its edges. The edges
of a column take only a few distinct values over the permutations (one per candidate bin count), so :class:`PermutedTargetBinner` prepares the frame once, memoises the code column of each
(column, edge set) it meets, and builds the code matrix of a permutation by one parallel gather. The result equals that of ``categorize_dataset``.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry

_TABLE_BYTES = 1536 * 1024 * 1024  # memoised code columns kept per binner; beyond it the caller falls back to ``categorize_dataset``


@njit(cache=True, parallel=True)
def _gather_code_columns(table: np.ndarray, slots: np.ndarray, out: np.ndarray) -> None:
    """``out[j] = table[slots[j]]`` for every row ``j`` of ``out``, parallel over columns; each is a contiguous copy."""
    for j in prange(slots.shape[0]):
        src = table[slots[j]]
        for r in range(src.shape[0]):
            out[j, r] = src[r]


class PermutedTargetBinner:
    """Codes of a numeric frame under many targets with the same result as ``categorize_dataset``, for strategies whose per-column edges repeat across targets.

    Args:
        df: pandas frame of numeric columns (anything else makes :meth:`supported` false).
        missing_strategy: the ``missing_strategy`` of ``categorize_dataset``.
        nbins_strategy: the supervised strategy name.
        nbins_strategy_kwargs: its kwargs (they may carry a plan cache).
        dtype: the code dtype of ``categorize_dataset``.
        out_cols: the names of the columns of the returned matrix, in order (default: the columns of ``df``); each is a column of ``df`` or a key of ``fixed``.
        fixed: ``{name: (codes, bin count)}`` of columns that do not depend on the target; they fill their place of the matrix unchanged.
    """

    def __init__(
        self, df: Any, missing_strategy: str, nbins_strategy: str, nbins_strategy_kwargs: Optional[dict], dtype: type, out_cols: Optional[list] = None, fixed: Optional[dict] = None,
    ) -> None:
        from . import _handle_missing
        from ._discretization_dataset import _discretize_input_dtype

        self.cols: list = df.head(5).select_dtypes(exclude=("category", "object", "string", "bool")).columns.values.tolist()
        arr = df[self.cols].to_numpy(dtype=_discretize_input_dtype(), na_value=np.nan)
        nan_mask = np.isnan(arr) if (missing_strategy in ("separate_bin", "propagate") and arr.size > 0) else None
        nan_cols = np.flatnonzero(nan_mask.any(axis=0)) if nan_mask is not None else np.empty(0, dtype=np.int64)
        arr = _handle_missing(arr, strategy=missing_strategy, nan_mask=nan_mask)
        self._arr = np.asfortranarray(arr)  # a column is one contiguous run
        self._nan = {int(j): nan_mask[:, j].copy() for j in nan_cols} if nan_mask is not None else {}
        self._strategy = nbins_strategy
        self._kwargs = dict(nbins_strategy_kwargs or {})
        self._dtype = dtype
        self._lock = threading.Lock()
        self._slots: "dict[tuple, int]" = {}
        self._table = np.empty((0, arr.shape[0]), dtype=dtype)
        self._col_max = np.empty(0, dtype=np.int64)
        self._used = 0
        # place of each output column: a position among the re-binned columns, or the table row of a fixed column
        position = {str(c): j for j, c in enumerate(self.cols)}
        names = [str(c) for c in (self.cols if out_cols is None else out_cols)]
        fixed = {str(k): v for k, v in (fixed or {}).items()}
        self.ok = all(c in position or c in fixed for c in names)
        self._out_names = names
        self._out_dep = np.array([position.get(c, -1) for c in names], dtype=np.int64)
        self._out_fixed = np.full(len(names), -1, dtype=np.int64)
        if self.ok:
            for i, c in enumerate(names):
                if self._out_dep[i] < 0:
                    codes, nb = fixed[c]
                    self._out_fixed[i] = self._add_row(np.asarray(codes), int(nb) - 1)

    @staticmethod
    def supported(df: Any) -> bool:
        """Whether ``df`` is a pandas frame with numeric columns only, at least one row and one column."""
        import pandas as pd

        if not isinstance(df, pd.DataFrame) or df.shape[0] == 0 or df.shape[1] == 0:
            return False
        return df.select_dtypes(include=("category", "object", "string", "bool")).shape[1] == 0

    def _slot(self, j: int, edges: np.ndarray, nan_code: int) -> Optional[int]:
        """Table row holding the code column of column ``j`` under ``edges``, encoded on first use; ``None`` when the table is at its budget."""
        key = (j, edges.tobytes())
        with self._lock:
            slot = self._slots.get(key)
            if slot is not None:
                return slot
            col = self._arr[:, j]
            codes = np.searchsorted(edges, col, side="right") if edges.size else np.zeros(col.shape[0], dtype=np.int64)
            nan = self._nan.get(j)
            if nan is not None:
                codes[nan] = nan_code
            slot = self._add_row(codes, int(codes.max()))
            if slot is not None:
                self._slots[key] = slot
            return slot

    def _add_row(self, codes: np.ndarray, top: int) -> Optional[int]:
        """Append a code column (highest code ``top``) to the table, growing it by doubling; ``None`` when that would pass the byte budget."""
        slot = self._used
        if slot >= self._table.shape[0]:
            grown = max(8, 2 * self._table.shape[0])
            if grown * self._table.shape[1] * self._table.itemsize > _TABLE_BYTES:
                return None
            table = np.empty((grown, self._table.shape[1]), dtype=self._dtype)
            table[: self._used] = self._table[: self._used]
            col_max = np.zeros(grown, dtype=np.int64)
            col_max[: self._used] = self._col_max[: self._used]
            self._table, self._col_max = table, col_max
        self._table[slot] = codes
        self._col_max[slot] = top
        self._used += 1
        return slot

    def codes(self, y: np.ndarray) -> "Optional[tuple[np.ndarray, list, np.ndarray]]":
        """``(codes, columns, bin counts)`` as ``categorize_dataset`` returns them for the target ``y`` (codes as int64, column-contiguous), or ``None`` when the caller should use ``categorize_dataset`` (budget, dtype range)."""
        from .._adaptive_nbins import per_feature_edges

        edges_list = per_feature_edges(self._arr, y=np.asarray(y).ravel(), method=self._strategy, cache_dir=None, **self._kwargs)
        if max((int(e.size) + 1 for e in edges_list), default=1) > np.iinfo(self._dtype).max:
            return None
        dep_slots = np.empty(len(edges_list), dtype=np.int64)
        for j, e in enumerate(edges_list):
            s = self._slot(j, np.asarray(e), int(e.size) + 1)
            if s is None:
                return None
            dep_slots[j] = s
        slots = np.where(self._out_dep >= 0, dep_slots[np.maximum(self._out_dep, 0)], self._out_fixed)
        out_t = np.empty((slots.shape[0], self._arr.shape[0]), dtype=np.int64)  # one contiguous run per column; the caller reads columns
        with self._lock:
            table, col_max = self._table, self._col_max
        with parallel_kernel_entry():
            _gather_code_columns(table, slots, out_t)
        return out_t.T, self._out_names, col_max[slots] + 1
