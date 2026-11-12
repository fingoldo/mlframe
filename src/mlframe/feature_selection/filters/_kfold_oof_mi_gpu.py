"""Device twin of the k-fold OOF MI scorer's fold step (``_orthogonal_three_gate_mi_fe.score_features_by_kfold_oof_mi``).

Per fold and column: equi-frequency edges at ``linspace(0, 1, nbins + 1)[1:-1]`` from the TRAIN rows (numpy's linear quantile, the radix select's values),
the TEST rows binned against them (``searchsorted(side="right")``; the host deduplicates the edges, which only renumbers bins, so the partition and every
entropy and observed support are the same), and the Miller-Madow MI of ``_mi_from_binned_xy`` with ``clip_zero=False``: ``H(X) + H(Y) - H(X, Y) -
(k_x k_y - k_x - k_y + 1) / (2 n)`` with the observed supports of the held-out fold. Only the summation order differs from the host (round-off).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _sweep_kfold() -> list:
    """Host-vs-device sweep of the k-fold OOF MI scorer for its kernel-tuning entry."""
    from ._orthogonal_three_gate_mi_fe import score_features_by_kfold_oof_mi
    from ._scorer_device_dispatch import sweep_scorer, synthetic_scorer_frame

    return sweep_scorer(lambda inp: score_features_by_kfold_oof_mi(*inp)["engineered_mi_oof"].to_numpy(), synthetic_scorer_frame)


def kfold_oof_mi_device_enabled(n_rows: int, n_cols: int) -> bool:
    """Whether the folds run on the device: a usable GPU, ``MLFRAME_KFOLD_OOF_MI_DEVICE`` not ``0`` and this scorer's own tuned choice for this size."""
    if os.environ.get("MLFRAME_KFOLD_OOF_MI_DEVICE", "1").strip() == "0":
        return False
    try:
        from ._gpu_policy import cuda_available_for_run

        if not cuda_available_for_run():
            return False
        from ._scorer_device_dispatch import scorer_device_choice

        return scorer_device_choice("kfold_oof_mi", int(n_rows), int(n_cols), sweep=_sweep_kfold)
    except Exception as exc:
        log_throttle(logger, "_kfold_oof_mi_gpu.except_40", logging.WARNING, "k-fold OOF MI device check failed (%s); host", exc)
        return False


def fold_mi_device(arr: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, y_test_bin: np.ndarray, nbins: int) -> Optional[np.ndarray]:
    """``(n_cols,)`` OOF MI of every column of ``arr`` for one fold (the host ``_one_fold`` values up to round-off), or ``None`` on any device fault."""
    try:
        import cupy as cp

        from ._fe_batched_mi_cmi import _batched_joint_counts2
        from ._gpu_resident_discretize import _searchsorted_codes
        from ._gpu_resident_select import radix_quantiles_grouped

        n_cols = int(arr.shape[1])
        if n_cols == 0:
            return np.zeros(0, dtype=np.float64)  # Intentional float64: score statistic output contract
        from ._fe_gpu_vram import fe_gpu_has_vram_cushion

        # Admission before the copy: the fold's two row slices (and about as much again in temporaries) must fit the VRAM cushion,
        # otherwise the host runs the fold. Checked first so a fold too large for the card is never copied to it.
        fold_bytes = int(arr.itemsize) * n_cols * (len(train_idx) + len(test_idx))
        if not fe_gpu_has_vram_cushion(2 * fold_bytes):
            return None
        tr = cp.asarray(np.ascontiguousarray(arr[train_idx, :]))
        te = cp.asarray(np.ascontiguousarray(arr[test_idx, :]))
        qs = np.linspace(0.0, 1.0, int(nbins) + 1)[1:-1]
        if qs.size == 0:
            return None
        edges = radix_quantiles_grouped(tr, qs)  # (nbins - 1, K) double, numpy's linear quantile of the train rows
        del tr
        if edges is None:
            return None
        codes = _searchsorted_codes(te, edges).astype(cp.int64)  # (n_test, K), side="right"
        del te
        yb = cp.asarray(np.ascontiguousarray(y_test_bin, dtype=np.int64))
        ky = int(y_test_bin.max()) + 1 if y_test_bin.size else 1
        kx = int(nbins)
        n = float(len(test_idx))
        counts = _batched_joint_counts2(codes, yb, kx, ky).astype(cp.float64).reshape(n_cols, kx, ky)  # Intentional float64: entropy accumulators

        def _h(c: Any, axis: "tuple[int, ...]") -> "tuple[Any, Any]":
            """Plug-in entropy (nats) and observed support of the counts summed over everything but ``axis``-complement."""
            p = c / n
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = cp.where(c > 0, p * cp.log(cp.where(c > 0, p, 1.0)), 0.0)
            return -terms.sum(axis=axis), (c > 0).sum(axis=axis)

        cx = counts.sum(axis=2)
        h_x, k_x = _h(cx, (1,))
        h_xy, _ = _h(counts, (1, 2))
        cy = counts[0].sum(axis=0)
        h_y, k_y = _h(cy, (0,))
        k_x = k_x.astype(cp.float64)  # Intentional float64: Miller-Madow support term
        k_y = float(k_y)
        mi = h_x + float(h_y) - h_xy
        ok = (k_x > 0) & (k_y > 0)
        bias = (k_x * k_y - k_x - k_y + 1.0) / (2.0 * n)
        mi = cp.where(ok, mi - bias, mi)
        return np.asarray(cp.asnumpy(mi), dtype=np.float64)  # Intentional float64: score statistic output contract
    except Exception as exc:
        log_throttle(logger, "_kfold_oof_mi_gpu.except_93", logging.WARNING, "k-fold OOF MI fold on the device failed (%s: %s); host", type(exc).__name__, exc)
        return None
