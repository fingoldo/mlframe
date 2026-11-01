"""Device twin of the (column, degree) cell enumeration + MI of the adaptive-degree basis FE.

The host path routes a basis per source column (``basis_route_by_signal``), builds every degree of the chosen basis as its own vector and scores the stacked matrix. Here all columns are uploaded
once, the bases are routed by the batched device router of ``_gpu_resident_basis``, every column's chosen basis x degree comes from the batched Clenshaw kernels and the plug-in MI is computed on the
device. Only the per-cell MI returns; the one winning degree per source is rebuilt by the host path, so engineered values and recipes are the host ones.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

class LazyAdaptiveValues:
    """``cand_values``-like sequence: element ``j`` is rebuilt on the host when asked for (only the winning cell of each source is)."""

    def __init__(self, X: Any, cand_to_source: list, dtype: Any) -> None:
        self._X = X
        self._cells = cand_to_source
        self._dtype = dtype

    def __getitem__(self, j: int) -> np.ndarray:
        """Host values of cell ``j``, built as ``_build_degree_candidates`` builds them."""
        from ._orthogonal_univariate_fe import _evaluate_basis_column

        src, deg, basis = self._cells[j]
        x = np.asarray(self._X[src].to_numpy(), dtype=self._dtype)
        finite = np.isfinite(x)
        if not finite.all():
            x = np.where(finite, x, float(np.nanmean(x[finite])) if finite.any() else 0.0)
        return _evaluate_basis_column(x, basis, int(deg))

    def __len__(self) -> int:
        return len(self._cells)


def adaptive_degree_cells_on_device(
    X: Any, cols: Sequence[str], y_arr: np.ndarray, degree_range: tuple, basis: str, nbins: int
) -> "tuple[list, list, np.ndarray] | None":
    """``(cand_cols, cand_to_source, eng_mi)`` of every usable (column, degree) cell in the host order, or ``None`` when the device path does not apply.

    ``basis`` is ``"auto"`` (routed per column on the device, over the bases that have an evaluator: see ``evaluable_bases``) or a pinned basis name. A column whose routing is degenerate sends
    the whole call to the host path.
    """
    try:
        import cupy as cp

        from ._basis_cells_ktc import use_device_cells
        from ._fe_gpu_batch._devices import crit_float_dtype
        from ._fe_resident_operands import resident_operand
        from ._gpu_resident_basis import _PREWARP_CLENSHAW_GPU, _gpu_evaluate_basis_matrix, _gpu_route_bases_batched
        from ._gpu_resident_fe import fe_gpu_resident_basis_mi_enabled
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
        from ._orthogonal_adaptive_degree_fe import _BASIS_CODE
        from ._orthogonal_univariate_fe import evaluable_bases
        from .hermite_fe._hermite_robust import _robust_axis_enabled
    except Exception as exc:
        log_throttle(logger, "adaptive_degree_device_imports", logging.DEBUG, "adaptive-degree device path unavailable (%s: %s)", type(exc).__name__, exc)
        return None
    n = len(X)
    if not fe_gpu_resident_basis_mi_enabled() or not use_device_cells(n):
        return None
    if basis != "auto" and basis not in _PREWARP_CLENSHAW_GPU:
        return None
    if basis == "auto" and any(b not in _PREWARP_CLENSHAW_GPU for b in evaluable_bases()):
        return None  # a base without a device twin would be skipped by the device router while the host router scores it
    try:
        dt = crit_float_dtype()
        columns = []
        for col in cols:
            x = np.asarray(X[col].to_numpy(), dtype=dt)
            finite = np.isfinite(x)
            if not finite.all():
                x = np.where(finite, x, float(np.nanmean(x[finite])) if finite.any() else 0.0)
            columns.append(x)
        M = cp.asarray(np.column_stack(columns), dtype=dt)
        robust = bool(_robust_axis_enabled())
        y_int = np.ascontiguousarray(y_arr, dtype=np.int64).ravel()
        if basis == "auto":
            y_f = y_int.astype(np.float64)  # Intentional float64: target statistic input
            if n < 30 or float(y_f.std()) < 1e-12:
                return None  # the host router falls back to the moment router there
            chosen = _gpu_route_bases_batched(cp, M, cp.asarray(y_f), list(evaluable_bases()), tuple(degree_range), robust_axis=robust)
            if any(c is None for c in chosen):
                return None
        else:
            chosen = [basis] * len(cols)
        cand, meta = _gpu_evaluate_basis_matrix(cp, M, chosen, list(degree_range), robust_axis=robust)
        if cand is None:
            return None
        usable = cp.isfinite(cand).all(axis=0) & (cand.std(axis=0) > 1e-12)
        keep = np.flatnonzero(cp.asnumpy(usable))
        if keep.size == 0:
            return [], [], np.empty(0, dtype=np.float64)  # Intentional float64: MI statistic vector
        y_gpu = resident_operand(y_int, "adaptive_degree_y_int", dtype=np.int64)
        n_classes = int(y_int.max()) - int(y_int.min()) + 1
        mi = np.asarray(
            _plugin_mi_classif_batch_cuda_resident(cand[:, cp.asarray(keep)], y_gpu, int(nbins), y_min=int(y_int.min()), n_classes=n_classes, relax_binning=True),
            dtype=np.float64,  # Intentional float64: MI statistic vector
        )
    except Exception as exc:
        log_throttle(logger, "adaptive_degree_device_failed", logging.WARNING, "device adaptive-degree cells failed (%s: %s); using the host path", type(exc).__name__, exc)
        return None
    degree_pos = {int(d): i for i, d in enumerate(degree_range)}
    cells = []
    for pos, j in enumerate(keep):
        ci, b, d = meta[int(j)]
        cells.append((ci, degree_pos[int(d)], cols[ci], int(d), b, float(mi[pos])))
    cells.sort(key=lambda r: r[:2])  # host order: source column, then degree
    cand_cols = [f"{c[2]}__{_BASIS_CODE.get(c[4], c[4])}{c[3]}" for c in cells]
    cand_to_source = [(c[2], c[3], c[4]) for c in cells]
    eng_mi = np.array([c[5] for c in cells], dtype=np.float64)  # Intentional float64: MI statistic vector
    return cand_cols, cand_to_source, eng_mi
