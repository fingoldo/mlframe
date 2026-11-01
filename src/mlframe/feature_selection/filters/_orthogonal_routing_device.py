"""Device twin of the cell enumeration + scoring of the conditional basis routing FE.

The host path builds every ``(source column, pre-transform, basis, degree)`` cell as its own float64 vector (about 90 ms per cell at 100000 rows), stacks them into one matrix and scores the
matrix. Here the pre-transformed columns of a block of source columns are uploaded once, every basis x degree is built by the batched Clenshaw kernels of ``_gpu_resident_basis``, and the cells
are scored (plug-in MI and |Pearson corr| with the target) without leaving the device. Only the per-cell scores come back; the few winning columns are rebuilt on the host by the host path, so the
engineered values and the recipes are the host ones.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence, cast

import numpy as np

from ._fe_usability_signal import _crit_np_dtype
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

class LazyRoutingValues:
    """``cand_values``-like sequence whose element ``j`` is rebuilt on the host when asked for (only the winning cells are asked for)."""

    def __init__(self, X: Any, cand_meta: list) -> None:
        self._X = X
        self._meta = cand_meta

    def __getitem__(self, j: int) -> np.ndarray:
        """Host values of cell ``j`` exactly as the host enumeration builds them."""
        from ._orthogonal_routing_fe import _routing_cells_of_column

        src, pt, basis, deg = self._meta[j]
        _, values, _ = _routing_cells_of_column(self._X, src, (pt,), (basis,), (int(deg),))
        return cast("np.ndarray", values[0])

    def __len__(self) -> int:
        return len(self._meta)


def _cells_per_source(n_variants: int, n_bases: int, n_degrees: int) -> int:
    """Upper bound of the cells one source column adds."""
    return max(1, n_variants * n_bases * n_degrees)


def routing_scores_on_device(
    X: Any,
    cols: Sequence[str],
    transform_variants: tuple,
    candidate_bases: tuple,
    degrees: tuple,
    y_arr: np.ndarray,
    nbins: int,
    routing_criterion: str,
    force: bool = False,
) -> "tuple[list, list, np.ndarray, np.ndarray] | None":
    """``(cand_cols, cand_meta, eng_mi, route_score)`` of every usable cell in the host enumeration order, or ``None`` when the device path does not apply (the host path then runs).

    A cell is usable when it is finite and not constant, like the host filter. ``eng_mi`` is the plug-in MI of the cell with ``y_arr`` and ``route_score`` its |Pearson corr| (or the MI under
    ``routing_criterion != "corr"`` / a constant target), as ``_routing_scores`` defines them.
    """
    try:
        import cupy as cp

        from ._basis_cells_ktc import use_device_cells
        from ._cell_validity_gpu import cell_validity_gpu
        from ._fe_gpu_batch._devices import crit_float_dtype
        from ._fe_resident_operands import resident_operand
        from ._gpu_resident_fe import _gpu_k_chunk, fe_gpu_resident_basis_mi_enabled
        from ._gpu_resident_basis import _gpu_batched_abs_corr, _gpu_detect_heavy_tail_batched, _gpu_evaluate_basis_matrix, device_bases
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
        from ._orthogonal_routing_fe import _pre_transform_variants_for_column, _routing_col_name
        from .hermite_fe._hermite_robust import _robust_axis_enabled
    except Exception as exc:
        log_throttle(logger, "routing_device_imports", logging.DEBUG, "routing device path unavailable (%s: %s)", type(exc).__name__, exc)
        return None
    n = len(X)
    if not fe_gpu_resident_basis_mi_enabled() or not (force or use_device_cells(n)):
        return None
    if any(b not in device_bases() for b in candidate_bases):
        return None  # a basis without a device port would silently lose its cells
    try:
        y_int = np.ascontiguousarray(y_arr, dtype=np.int64).ravel()
        y_gpu = resident_operand(y_int, "routing_y_int", dtype=np.int64)
        y_min = int(y_int.min())
        n_classes = int(y_int.max()) - y_min + 1
        y_f = y_int.astype(np.float64)  # Intentional float64: target statistic input
        use_corr = str(routing_criterion).lower() == "corr" and float(y_f.std()) > 1e-12
        y_f_gpu = cp.asarray(y_f)
        robust = bool(_robust_axis_enabled())
        dt = _crit_np_dtype()
        block_sources = max(1, int(_gpu_k_chunk(n)) // _cells_per_source(len(transform_variants), len(candidate_bases), len(degrees)))
        records: list = []  # (entry index, basis index, degree index, source, pre-transform, basis, degree, mi, score)
        entry_no = 0
        for b0 in range(0, len(cols), block_sources):
            entries: list = []
            for col in cols[b0 : b0 + block_sources]:
                x_raw = np.asarray(X[col].to_numpy(), dtype=dt)
                finite = np.isfinite(x_raw)
                if not finite.all():
                    fill = float(np.nanmean(x_raw[finite])) if finite.any() else 0.0
                    x_raw = np.where(finite, x_raw, fill)
                pt_cache: dict = {}
                _pre_transform_variants_for_column(transform_variants, x_raw, col, pt_cache)
                for pt, xt in pt_cache.items():
                    entries.append((col, pt, xt))
            if not entries:
                continue
            M = cp.asarray(np.column_stack([e[2] for e in entries]), dtype=crit_float_dtype())  # the values keep the criterion dtype (float32 by default), as the resident univariate path
            heavy = cp.asnumpy(_gpu_detect_heavy_tail_batched(cp, M)) if robust else np.zeros(M.shape[1], dtype=bool)
            for bi, basis in enumerate(candidate_bases):
                cand, meta = _gpu_evaluate_basis_matrix(cp, M, [basis] * M.shape[1], list(degrees), robust_axis=robust, heavy_host=heavy)
                if cand is None:
                    continue
                usable, col_mean = cell_validity_gpu(cp, cand)
                keep = np.flatnonzero(usable)
                if keep.size == 0:
                    continue
                # the whole block when every cell is usable (the usual case), else only the usable columns
                sel = cand if keep.size == cand.shape[1] else cand[:, cp.asarray(keep)]
                mi = np.asarray(_plugin_mi_classif_batch_cuda_resident(sel, y_gpu, int(nbins), y_min=y_min, n_classes=n_classes, relax_binning=True), dtype=np.float64)  # Intentional float64: MI statistic
                if use_corr:
                    ac = cp.asnumpy(_gpu_batched_abs_corr(cp, sel, y_f_gpu, mu=col_mean[keep]))
                    score = np.where(ac < 0.0, 0.0, ac)  # the kernel marks a degenerate column with -1
                else:
                    score = mi
                deg_index = {int(d): i for i, d in enumerate(degrees)}
                for pos, j in enumerate(keep):
                    ci, _basis, d = meta[int(j)]
                    col, pt, _ = entries[ci]
                    records.append((entry_no + ci, bi, deg_index[int(d)], col, pt, basis, int(d), float(mi[pos]), float(score[pos])))
                del sel, cand
            entry_no += len(entries)
            del M
        records.sort(key=lambda r: r[:3])
        cand_cols = [_routing_col_name(r[3], r[5], r[6], r[4]) for r in records]
        cand_meta = [(r[3], r[4], r[5], r[6]) for r in records]
        eng_mi = np.array([r[7] for r in records], dtype=np.float64)  # Intentional float64: MI statistic vector
        route_score = np.array([r[8] for r in records], dtype=np.float64)  # Intentional float64: routing score vector
        return cand_cols, cand_meta, eng_mi, route_score
    except Exception as exc:
        log_throttle(logger, "routing_device_failed", logging.WARNING, "device routing cells failed (%s: %s); using the host path", type(exc).__name__, exc)
        return None
