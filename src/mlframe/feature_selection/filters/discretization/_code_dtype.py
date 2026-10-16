"""Integer code-dtype widening for the discretisers, kept in a leaf module.

Leaf module: the CUDA discretiser and the package facade both import it, and it imports nothing from the package.
"""

import logging

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.discretization")


def _safe_code_dtype(n_bins: int, dtype: type, reserve_nan_slot: bool = False) -> type:
    """Widen ``dtype`` to one that can hold ordinal codes ``0..n_bins-1`` (``0..n_bins`` when ``reserve_nan_slot``).

    Discretiser codes reach ``n_bins-1``; the default ``int8`` only holds 0..127, so an
    ``astype(int8)`` on ``n_bins>128`` wraps the top bins negative (modulo arithmetic),
    silently mis-binning the high-magnitude region. Auto-widen instead of silently
    corrupting: int8->int16->int32->int64 as the bin count grows. No-op for n_bins<=128.

    The uniform path emits a dedicated NaN sentinel at code ``n_bins`` (one past the top real code
    ``n_bins-1``); pass ``reserve_nan_slot=True`` there so the sentinel never wraps negative at
    ``n_bins == dtype.max+1`` (int8@128, int16@32768). The quantile path routes NaN to the rightmost
    real code ``n_bins-1`` and keeps the tighter default.
    """
    try:
        info: np.iinfo = np.iinfo(np.dtype(dtype))  # type: ignore[arg-type, unused-ignore]  # dtype may be non-integer (e.g. a float dtype); caught below at runtime
    except (ValueError, TypeError) as _exc:
        logger.debug("_safe_code_dtype fell back to its default after %s", type(_exc).__name__)
        return dtype  # non-integer requested dtype: caller owns the contract
    top_code = n_bins if reserve_nan_slot else n_bins - 1
    if top_code <= info.max:
        return dtype
    for cand in (np.int16, np.int32, np.int64):
        if top_code <= np.iinfo(cand).max:
            return cand
    return np.int64
