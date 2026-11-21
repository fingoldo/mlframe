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


_CODE_DTYPES = (np.int8, np.uint8, np.int16, np.int32, np.int64)


def narrowest_code_dtype(lo: int, hi: int) -> "type | None":
    """The narrowest of int8 / uint8 / int16 / int32 / int64 that holds every integer in ``[lo, hi]`` (signed int8 first, so codes up to 127 keep the layout they always had); ``None`` when none does."""
    for cand in _CODE_DTYPES:
        info = np.iinfo(cand)
        if info.min <= lo and hi <= info.max:
            return cand
    return None


def append_code_columns(data: np.ndarray, new: np.ndarray) -> np.ndarray:
    """``data`` with the code columns ``new`` appended, never wrapping a value: ``astype(data.dtype)`` turned a code of 128 into -128 on int8 data and a -1 into 255 on uint8 data, silently.

    When ``new`` has a value outside the range of ``data.dtype`` the result takes the narrowest integer dtype that holds both the whole range of ``data.dtype`` and the values of ``new``; otherwise the result keeps ``data.dtype``.
    A non-integer ``data`` or ``new`` falls back to the plain ``np.append`` cast to ``data.dtype``."""
    new = np.asarray(new)
    if new.ndim == 1:
        new = new[:, None]
    if data.dtype.kind not in "iu" or new.dtype.kind not in "iub":
        return np.append(data, new.astype(data.dtype, copy=False), axis=1)
    target = data.dtype
    if new.size:
        lo, hi = int(new.min()), int(new.max())
        info = np.iinfo(data.dtype)
        if lo < info.min or hi > info.max:
            widened = narrowest_code_dtype(min(lo, int(info.min)), max(hi, int(info.max)))
            target = np.dtype(widened if widened is not None else np.int64)
            logger.warning(
                "append_code_columns: %d new column(s) hold codes in [%d, %d], outside %s; the %d x %d code matrix (%.0f MB) is widened to %s (%.0f MB)",
                int(new.shape[1]), lo, hi, data.dtype.name, int(data.shape[0]), int(data.shape[1]), data.nbytes / 1e6, target.name, data.size * target.itemsize / 1e6,
            )
    return np.concatenate((data, new), axis=1, dtype=target, casting="unsafe")
