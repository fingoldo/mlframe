"""Single-pass per-cell sum/count accumulation over training rows of one fold."""

from __future__ import annotations

import numba
import numpy as np

from mlframe.metrics import NUMBA_NJIT_PARAMS


@numba.njit(**NUMBA_NJIT_PARAMS)
def _cell_sum_cnt_kernel(codes, y, fold_ids, skip_fold, valid, use_valid, n_cells):
    """Accumulate per-cell sum and count over rows with fold_ids != skip_fold (and valid[i] when use_valid), in row order."""
    cell_sum = np.zeros(n_cells, dtype=np.float64)
    cell_cnt = np.zeros(n_cells, dtype=np.float64)
    for i in range(codes.shape[0]):
        if fold_ids[i] == skip_fold:
            continue
        if use_valid and not valid[i]:
            continue
        c = codes[i]
        cell_sum[c] += y[i]
        cell_cnt[c] += 1.0
    return cell_sum, cell_cnt


def fold_cell_sum_cnt(codes, y, fold_ids, skip_fold, n_cells, valid=None):
    """Per-cell (sum, count) of y over rows outside fold ``skip_fold`` (``skip_fold=-1`` keeps every row), optionally restricted to ``valid``.

    Summation runs in row order exactly like ``np.add.at`` on the boolean-masked arrays, so the result is bit-identical.
    """
    use_valid = valid is not None
    v = valid if use_valid else np.ones(1, dtype=np.bool_)
    return _cell_sum_cnt_kernel(
        np.ascontiguousarray(codes, dtype=np.int64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(fold_ids, dtype=np.int64),
        int(skip_fold),
        v,
        use_valid,
        int(n_cells),
    )
