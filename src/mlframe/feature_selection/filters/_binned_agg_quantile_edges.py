"""Quantile cut points for the binned numeric aggregation family, with the strict-resident device path for large columns (carved from _binned_numeric_agg_fe; re-exported there)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# Columns at or above this many rows take the device quantile (a host np.quantile there costs 5-10x a device sort plus the upload it overlaps with).
_DEVICE_QUANTILE_MIN_N = 200_000


def quantile_edges(x: np.ndarray, nbins: int) -> np.ndarray:
    """Inner quantile cut points (unique-deduped); code = searchsorted(edges, v, side='right')."""
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    x = np.asarray(x, dtype=np.float64)
    if x.size >= _DEVICE_QUANTILE_MIN_N:
        # Large column under the strict-resident path: one device sort + the order statistics numpy would read, bit-identical to np.quantile (~115 ms
        # per 1M-row column on the host).
        try:
            from ._gpu_strict_fe import fe_gpu_strict_resident_enabled

            if fe_gpu_strict_resident_enabled():
                from ._device_quantile import device_quantile

                dev = device_quantile(x, qs)
                if dev is not None:
                    return np.unique(dev)
        except Exception as e:
            logger.debug("device quantile edges failed, using the host np.quantile: %s", e)
    return np.unique(np.quantile(x, qs))
