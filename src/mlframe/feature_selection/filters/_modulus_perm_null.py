"""Permutation null of the modulus search: the best residue MI of every permuted y in one batched pass.

``escalate_modulus`` run once per permuted y re-bins the same residue ``c mod m`` for every permutation and pays a full
single-column MI dispatch per modulus. Here the residue of each candidate modulus is binned once and every permutation's
histogram is accumulated inside one ``prange`` kernel, with the same loop order as ``_plugin_mi_classif_njit`` so the
per-permutation MI values are the ones the scalar path returns.
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

import numpy as np
from numba import njit, prange

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._numba_utils import _quantile_bin_njit
from mlframe.utils.log_throttle import log_throttle

from ._pairwise_modular_fe import _combine, modulus_grid, select_modulus

logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def _perm_mi_from_binned_njit(x_binned: np.ndarray, yp: np.ndarray, n_bins: int, y_min: int, n_classes: int) -> np.ndarray:
    """MI (nats) of one pre-binned x against each row of ``yp`` (permuted encoded targets), same accumulation as ``_plugin_mi_classif_njit``."""
    n_perm = yp.shape[0]
    n = x_binned.shape[0]
    out = np.zeros(n_perm, dtype=np.float64)  # Intentional float64: permutation-null statistic at double, compared against the observed value
    log_n = math.log(n)
    for p in prange(n_perm):
        hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
        hist_x = np.zeros(n_bins, dtype=np.int64)
        hist_y = np.zeros(n_classes, dtype=np.int64)
        for i in range(n):
            b = x_binned[i]
            c = yp[p, i] - y_min
            hist_xy[b, c] += 1
            hist_x[b] += 1
            hist_y[c] += 1
        mi = 0.0
        for b in range(n_bins):
            if hist_x[b] == 0:
                continue
            log_hx = math.log(hist_x[b])
            for c in range(n_classes):
                n_xy = hist_xy[b, c]
                if n_xy == 0 or hist_y[c] == 0:
                    continue
                mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[c]))
        if mi < 0.0:
            mi = 0.0
        out[p] = mi
    return out


def host_residue_path() -> bool:
    """True when the escalation's residue MI runs on the host ``_mi`` (no device-born or resident scoring); only then is the batched null exact."""
    from ._pairwise_modular_fe import _resident_scan_enabled

    if _resident_scan_enabled():
        return False
    try:
        from ._gpu_strict_fe import fe_gpu_device_born_modular_enabled

        return not bool(fe_gpu_device_born_modular_enabled())
    except Exception as exc:
        log_throttle(logger, "modulus_perm_null.host_residue_path_unknown", logging.WARNING, "modulus permutation null: the device-born probe failed, so the batched host null is assumed exact: %s", exc)
        return True


def residue_mi_per_perm(r: np.ndarray, yi_perms: np.ndarray, n_bins: int) -> np.ndarray:
    """``MI(r; yi_perms[p])`` for every permutation ``p``, with ``r`` binned once.

    ``yi_perms`` is an ``(P, n)`` int64 matrix of encoded targets; every row has the same value set, so the class offset and
    count are shared across rows.
    """
    x_binned = _quantile_bin_njit(np.ascontiguousarray(r, dtype=np.float64), int(n_bins))  # Intentional float64: permutation-null statistic at double, compared against the observed value
    y_min = int(yi_perms.min())
    n_classes = int(yi_perms.max()) - y_min + 1
    return cast(np.ndarray, _perm_mi_from_binned_njit(x_binned, yi_perms, int(n_bins), y_min, n_classes))


def perm_best_mis(hit: Any, X: Any, yi_perms: np.ndarray, nbins: int) -> np.ndarray:
    """Best residue MI over the escalation grid for each permuted target, equal to ``escalate_modulus(X, y_perm, hit)[1]`` per row.

    Only valid on the host residue path (no resident residue MI); the caller checks that. The residue of each grid modulus is
    computed once, so the per-permutation work is the histogram accumulation alone.
    """
    c_arr = hit.c_arr
    if c_arr is None:
        arrs = [np.asarray(X[c]) for c in hit.cols]
        c_arr = _combine(arrs, hit.op)
    grid = modulus_grid(hit.modulus)
    n_perm, n_rows = yi_perms.shape
    mis = {m: residue_mi_per_perm(np.mod(c_arr, m).astype(_crit_np_dtype()), yi_perms, max(nbins, m)) for m in grid}
    out = np.empty(n_perm, dtype=np.float64)  # Intentional float64: permutation-null statistic at double, compared against the observed value
    for p in range(n_perm):
        n_classes = int(np.unique(yi_perms[p]).size)
        out[p] = select_modulus(grid, lambda m, p=p: mis[m][p], hit.modulus, n_rows, n_classes)[1]
    return out
