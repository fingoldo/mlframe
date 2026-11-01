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


def route_columns_on_device(X: Any, cols: Sequence[str], y: Any, degrees: Sequence[int], cols_auto: bool, aux_pool: Any) -> "dict[str, str]":
    """``{column: basis}`` of the columns routed on the device; empty when the device path does not apply."""
    try:
        import cupy as cp

        from ._basis_cells_ktc import use_device_cells
        from ._fe_gpu_batch._devices import crit_float_dtype
        from ._gpu_resident_basis import _gpu_route_bases_batched, device_bases
        from ._gpu_resident_fe import fe_gpu_resident_basis_mi_enabled, fe_gpu_routing_enabled
        from ._orthogonal_univariate_fe import _is_int_as_cat_axis, evaluable_bases
        from .hermite_fe._hermite_robust import _robust_axis_enabled
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
            x = np.asarray(X[col].to_numpy(), dtype=np.float64)  # Intentional float64: the generator's own filters read the float64 column
            if (cols_auto and _is_int_as_cat_axis(x)) or not np.isfinite(x).all():
                continue
            eligible.append(col)
            columns.append(x)
        if not eligible:
            return {}
        M = cp.asarray(np.column_stack(columns), dtype=crit_float_dtype())
        chosen = _gpu_route_bases_batched(cp, M, cp.asarray(yv), bases, tuple(int(d) for d in degrees), robust_axis=bool(_robust_axis_enabled()))
    except Exception as exc:
        log_throttle(logger, "univariate_route_device_failed", logging.WARNING, "device routing of the univariate columns failed (%s: %s); the host router decides", type(exc).__name__, exc)
        return {}
    return {c: b for c, b in zip(eligible, chosen) if b is not None}
