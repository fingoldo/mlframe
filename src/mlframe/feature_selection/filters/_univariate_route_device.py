"""Batched device routing of the columns of the univariate basis generator.

``_generate_univariate_basis_features_impl`` chose the basis of every column on the host, one column at a time (``basis_route_by_signal``: a fit and a one-hot evaluation per candidate basis and degree).
With eight candidate bases that is the largest host cost of the stage. The columns that pass the generator's own filters (finite, not an integer-coded group key, no unlabeled pool) are uploaded as one
block and routed by the batched device router; a column the device cannot route is left out of the result, and the host router decides it exactly as before.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _native_column(series: Any) -> np.ndarray:
    """The column as a host array of its own floating dtype (float32 stays float32); other dtypes become float32 only when exact there is not required, i.e. never float64 by default."""
    a = np.asarray(series.to_numpy())
    return a if a.dtype.kind == "f" else a.astype(np.float32)


#: Share of the free device memory one whole-block upload (the row-major block and its transposed copy) may take; a larger block is filled column by column.
_BLOCK_UPLOAD_FRACTION = 0.25


def _upload_columns(cp: Any, columns: list, dtype: Any) -> Any:
    """The ``(n, k)`` device matrix of the host columns.

    The columns are copied as the rows of one host block (a contiguous copy each, no strided ``column_stack`` write), uploaded in one transfer and transposed on the device: two launches
    for the block. Filling the device matrix column by column was one transfer and one assignment launch per column (420 transfers of 120 kB, 0.48 s of a 100000 x 100 fit). A block whose two
    device copies do not fit the free-memory share keeps the column-by-column fill."""
    n, k = int(columns[0].shape[0]), len(columns)
    need = 2 * n * k * int(np.dtype(dtype).itemsize)
    try:
        from ._device_sort_safe import device_room

        fits = need <= _BLOCK_UPLOAD_FRACTION * device_room(cp)
    except Exception as exc:
        log_throttle(logger, "univariate_route_device_room", logging.DEBUG, "device room unknown (%s: %s); column-by-column upload", type(exc).__name__, exc)
        fits = False
    if fits and k > 1:
        rows = np.empty((k, n), dtype=dtype)
        for j, col in enumerate(columns):
            rows[j] = col
        return cp.ascontiguousarray(cp.asarray(rows).T)
    out = cp.empty((n, k), dtype=dtype)
    for j, col in enumerate(columns):
        out[:, j] = cp.asarray(col)
    return out


def route_columns_on_device(X: Any, cols: Sequence[str], y: Any, degrees: Sequence[int], cols_auto: bool, aux_pool: Any) -> "dict[str, str]":
    """``{column: basis}`` of the columns routed on the device; empty when the device path does not apply."""
    try:
        import cupy as cp

        from ._basis_cells_ktc import use_device_cells
        from ._fe_gpu_batch._devices import crit_float_dtype
        from ._gpu_resident_basis import _gpu_route_bases_batched, device_bases
        from ._gpu_resident_fe import fe_gpu_resident_basis_mi_enabled, fe_gpu_routing_enabled
        from ._orthogonal_univariate_fe import _is_int_as_cat_axis, evaluable_bases
        from .hermite_fe.shared import robust_axis_enabled as _robust_axis_enabled
    except Exception as exc:
        log_throttle(logger, "univariate_route_device_imports", logging.DEBUG, "device routing of the univariate columns unavailable (%s: %s)", type(exc).__name__, exc)
        return {}
    n = len(X)
    if y is None or aux_pool is not None or not fe_gpu_resident_basis_mi_enabled() or not fe_gpu_routing_enabled() or not use_device_cells(n):
        return {}
    bases = list(evaluable_bases())
    if any(b not in device_bases() for b in bases):
        return {}
    try:
        yv = np.asarray(y, dtype=np.float64).ravel()  # Intentional float64: target statistic input
        if n < 30 or yv.size != n or not np.isfinite(yv).all() or float(yv.std()) < 1e-12:
            return {}  # the host router falls back to the moment router there
        eligible, columns = [], []
        for col in cols:
            x = _native_column(X[col])
            if (cols_auto and _is_int_as_cat_axis(x)) or not np.isfinite(x).all():
                continue
            eligible.append(col)
            columns.append(x)
        if not eligible:
            return {}
        from ._device_blocks import first_block, run_column_blocks

        dt = crit_float_dtype()
        y_dev = cp.asarray(yv)
        robust = bool(_robust_axis_enabled())
        degs = tuple(int(d) for d in degrees)

        def route(lo: int, hi: int) -> list:
            """The routed bases of the columns ``lo:hi``."""
            return list(_gpu_route_bases_batched(cp, _upload_columns(cp, columns[lo:hi], dt), y_dev, bases, degs, robust_axis=robust))

        multiple = (len(bases) * max(1, len(degs))) + 4
        chosen = [b for piece in run_column_blocks(cp, len(columns), first_block(cp, n, len(columns), np.dtype(dt).itemsize, multiple), route) for b in piece]
    except Exception as exc:
        log_throttle(logger, "univariate_route_device_failed", logging.WARNING, "device routing of the univariate columns failed (%s: %s); the host router decides", type(exc).__name__, exc)
        return {}
    return {c: b for c, b in zip(eligible, chosen) if b is not None}


def build_cells_on_device(X: Any, chosen: "dict[str, str]", degrees: Sequence[int], code: "dict[str, str]") -> "dict[str, dict[str, np.ndarray]]":
    """``{column: {cell name: values}}`` of the columns routed on the device, with the cells of every (column, degree) built there too and returned as host columns of the criterion dtype.

    Owner decision (2026-11-16): bit identity with the host cells and float64 are not required while the selected features stay the same. The host built each cell from a float64 fit axis one column at a time
    (the polynomial evaluators, and for rbf / sigmoid / pade / fourier their own njit kernels); here a block of columns is uploaded once and ``_gpu_evaluate_basis_matrix`` builds all of the block's
    cells in a few launches. The recipes of the winners are rebuilt from the cell NAMES by the host path, which replays the same function to float32 rounding. Empty when the device cannot build them."""
    if not chosen:
        return {}
    try:
        import cupy as cp

        from ._device_blocks import first_block, run_column_blocks
        from ._fe_gpu_batch._devices import crit_float_dtype
        from ._gpu_resident_basis import _gpu_evaluate_basis_matrix
        from .hermite_fe.shared import robust_axis_enabled as _robust_axis_enabled

        names = list(chosen)
        dt = crit_float_dtype()
        columns = [_native_column(X[c]) for c in names]  # the host column in its own dtype; the cast to the criterion dtype happens on the device
        robust = bool(_robust_axis_enabled())
        degs = [int(d) for d in degrees]
        n = len(X)

        def block(lo: int, hi: int) -> "dict[str, dict[str, np.ndarray]]":
            """The cells of the columns ``lo:hi``."""
            out: "dict[str, dict[str, np.ndarray]]" = {}
            cand, meta = _gpu_evaluate_basis_matrix(cp, _upload_columns(cp, columns[lo:hi], dt), [chosen[c] for c in names[lo:hi]], degs, robust_axis=robust)
            if cand is None:
                return out
            host = cp.asnumpy(cp.ascontiguousarray(cand.T))  # (cells, n): every cell is a contiguous host row, no strided copy per cell
            for j, (ci, basis, deg) in enumerate(meta):
                out.setdefault(names[lo + ci], {})[f"{names[lo + ci]}__{code.get(basis, basis)}{deg}"] = host[j]
            return out

        multiple = len(degs) + 6
        merged: "dict[str, dict[str, np.ndarray]]" = {}
        for piece in run_column_blocks(cp, len(names), first_block(cp, n, len(names), np.dtype(dt).itemsize, multiple), block):
            merged.update(piece)
        return {c: cells for c, cells in merged.items() if len(cells) == len(degs)}  # a column missing a cell is left to the host path
    except Exception as exc:
        log_throttle(logger, "univariate_cells_device_failed", logging.WARNING, "device build of the univariate cells failed (%s: %s); the host builds them", type(exc).__name__, exc)
        return {}
