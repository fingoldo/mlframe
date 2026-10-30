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
        # a column is one contiguous run, in the float64 the edge builders convert each column to (the float32 values widen exactly, so every comparison is the same)
        self._arr = np.asfortranarray(arr, dtype=np.float64)
        self._nan = {int(j): nan_mask[:, j].copy() for j in nan_cols} if nan_mask is not None else {}
        self._strategy = nbins_strategy
        self._kwargs = dict(nbins_strategy_kwargs or {})
        self._dtype = dtype
        self._lock = threading.Lock()
        self._slots: "dict[tuple, int]" = {}
        self._table: np.ndarray = np.empty((0, arr.shape[0]), dtype=dtype)
        self._col_max = np.empty(0, dtype=np.int64)
        self._used = 0
        self._plans: Optional[list] = None  # the cached optimal-joint plan of every re-binned column, once they all exist
        self._packed: Optional[dict] = None
        self._by_count: "dict[tuple, int]" = {}  # (column, winning bin count) -> table row
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
        return bool(df.select_dtypes(include=("category", "object", "string", "bool")).shape[1] == 0)

    def _edges_slot(self, j: int, y: np.ndarray) -> Optional[int]:
        """Table row of column ``j`` under the edges the strategy gives it for ``y``; ``None`` when the codes do not fit the dtype or the table."""
        from .._adaptive_nbins import per_feature_edges

        edges = np.asarray(per_feature_edges(self._arr[:, j : j + 1], y=y, method=self._strategy, cache_dir=None, **self._kwargs)[0])
        if edges.size + 1 > np.iinfo(self._dtype).max:
            return None
        return self._slot(j, edges, int(edges.size) + 1)

    def column(self, y: np.ndarray, out_index: int) -> "Optional[tuple[np.ndarray, int]]":
        """``(codes, bin count)`` of output column ``out_index`` alone under the target ``y`` (the column's edges depend on that column and ``y`` only), or ``None`` when the caller should bin the frame."""
        from .._adaptive_nbins import per_feature_edges

        slot = int(self._out_fixed[out_index])
        if slot < 0:
            j = int(self._out_dep[out_index])
            edges = per_feature_edges(self._arr[:, j : j + 1], y=np.asarray(y).ravel(), method=self._strategy, cache_dir=None, **self._kwargs)[0]
            s = self._slot(j, np.asarray(edges), int(edges.size) + 1)
            if s is None:
                return None
            slot = s
        with self._lock:
            return self._table[slot], int(self._col_max[slot]) + 1

    def _prescore(self, y: np.ndarray) -> "Optional[list]":
        """Score the candidate bin counts of every column for ``y`` in one parallel kernel (``optimal_joint`` with a plan cache), once the plans of all columns are cached."""
        cache = self._kwargs.get("optimal_joint_plan_cache")
        if cache is None or str(self._strategy).lower() not in ("optimal_joint", "cv") or (y.dtype.kind == "f" and not np.isfinite(y).all()):
            return None
        from .._optimal_joint_edges import pack_plans, prescore_best_m

        candidates = tuple(self._kwargs.get("candidates", (4, 8, 16, 32)))
        n_splits, seed = self._kwargs.get("n_splits", 3), self._kwargs.get("random_state", 0)
        if self._packed is None:
            if self._plans is False:  # type: ignore[comparison-overlap]
                return None
            plans = [cache.get(self._arr[:, j], candidates, n_splits, "quantile", seed) for j in range(self._arr.shape[1])]
            if any(plan is None for plan in plans):
                return None  # the first permutation builds the plans on the normal route
            packed = pack_plans(plans, candidates)
            if packed is None:
                self._plans = False  # type: ignore[assignment]
                return None
            self._plans, self._packed = plans, packed
        assert self._plans is not None
        return prescore_best_m(self._packed, self._plans, y, cache._target_key(y), cache.fold_target_provider(y, self._plans[0]["shape_key"], self._kwargs.get("optimal_joint_max_y_classes", 64)))

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
            table: np.ndarray = np.empty((grown, self._table.shape[1]), dtype=self._dtype)
            table[: self._used] = self._table[: self._used]
            col_max = np.zeros(grown, dtype=np.int64)
            col_max[: self._used] = self._col_max[: self._used]
            self._table, self._col_max = table, col_max
        self._table[slot] = codes
        self._col_max[slot] = top
        self._used += 1
        return slot

    def codes(self, y: np.ndarray) -> "Optional[tuple[np.ndarray, list, np.ndarray]]":
        """``(codes, columns, bin counts)`` as ``categorize_dataset`` returns them for the target ``y`` (codes in the code dtype, column-contiguous), or ``None`` when the caller should use ``categorize_dataset`` (budget, dtype range)."""
        from .._adaptive_nbins import per_feature_edges

        y = np.asarray(y).ravel()
        winners = self._prescore(y)
        n_cols = self._arr.shape[1]
        dep_slots = np.empty(n_cols, dtype=np.int64)
        if winners is not None and all(m is not None for m in winners):
            # the edges of a column are a function of the column and the winning bin count alone, so a repeated (column, count) is a table row already
            for j, m in enumerate(winners):
                s = self._by_count.get((j, m))
                if s is None:
                    s = self._edges_slot(j, y)
                    if s is None:
                        return None
                    self._by_count[(j, m)] = s
                dep_slots[j] = s
        else:
            edges_list = per_feature_edges(self._arr, y=y, method=self._strategy, cache_dir=None, **self._kwargs)
            if max((int(e.size) + 1 for e in edges_list), default=1) > np.iinfo(self._dtype).max:
                return None
            for j, e in enumerate(edges_list):
                s = self._slot(j, np.asarray(e), int(e.size) + 1)
                if s is None:
                    return None
                dep_slots[j] = s
        slots = np.where(self._out_dep >= 0, dep_slots[np.maximum(self._out_dep, 0)], self._out_fixed)
        out_t: np.ndarray = np.empty((slots.shape[0], self._arr.shape[0]), dtype=self._dtype)  # one contiguous run per column, in the code dtype; the caller reads columns
        with self._lock:
            table, col_max = self._table, self._col_max
        with parallel_kernel_entry():
            _gather_code_columns(table, slots, out_t)
        return out_t.T, self._out_names, col_max[slots] + 1
