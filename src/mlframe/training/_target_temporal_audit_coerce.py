"""Timestamp coercion + lazy ruptures import for temporal target audit.

Carved out of ``target_temporal_audit`` to keep the parent facade under the LOC budget.
The functions here are pure-numpy / pandas timestamp normalisation helpers and have
no upward dependency on the audit dataclasses or aggregation kernels.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _import_ruptures():
    """Lazy import of ``ruptures`` so callers using only the z-score detector
    never pay the import cost (ruptures pulls in scipy.sparse.linalg + numba
    extensions; ~150ms cold). ``ImportError`` propagates to the caller; the
    z-score path is dependency-free and the obvious fallback.
    """
    import ruptures as rpt  # type: ignore[import-not-found]
    return rpt


# Lower / upper datetime sanity bounds for auto-unit detection.
_AUDIT_DATETIME_LOW_NS = np.int64(0)  # 1970-01-01 in nanoseconds-since-epoch
_AUDIT_DATETIME_HIGH_NS = np.int64(7258118400_000_000_000)  # 2200-01-01 in ns
_AUDIT_UNIT_NS_FACTOR: dict[str, int] = {
    "s":  1_000_000_000,
    "ms":     1_000_000,
    "us":         1_000,
    "ns":             1,
}


def coerce_timestamps_for_audit(
    arr,
    *,
    explicit_unit: str | None = None,
) -> np.ndarray:
    """Coerce a numeric timestamp array to numpy datetime64[ns] for audit binning.

    Unit auto-detection (when ``explicit_unit`` is None):
    - Pure datetime64 input or pd.DatetimeIndex / pd.Series of datetimes: returned as-is.
    - Numeric input: try each candidate unit in (s, ms, us, ns) in COARSEST-first
      order. For each, compute resulting timestamp range. Accept the first unit
      whose entire min..max range lands in [1970-01-01, 2200-01-01]; this gives
      the audit the widest meaningful time-span for binning. Coarsest-first matters
      because a value like 1_700_000_000 is legal as ns (=1.7 sec past epoch,
      degenerate single-bin audit) AND as s (=2023-11-14, useful audit); we want
      the latter.
    - If no unit is in-range, log a warning and fall back to ns to preserve
      pre-existing behaviour.

    ``explicit_unit`` (optional): forces the unit interpretation. Must be one of
    {'s', 'ms', 'us', 'ns'}.

    Returns a 1-D numpy datetime64[ns] array. Accepts numpy ndarray, list,
    pd.Series, or any 1-D ``Sequence``-like input.
    """
    arr = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr

    if np.issubdtype(arr.dtype, np.datetime64):
        # Force datetime64[ns] resolution. pandas 2.x / numpy 2.x may surface
        # DatetimeIndex.to_numpy() at us / s units, and downstream audit kernels
        # view-cast bytes to int64 expecting ns. Returning the input unit silently
        # shifts the time-axis by 1000x and lands every row in the 1970 epoch.
        if arr.dtype != np.dtype("datetime64[ns]"):
            arr = arr.astype("datetime64[ns]")
        return arr
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        # Object / string / pd.Timestamp etc. -- let pandas figure it out.
        # If the input contains tz-aware pd.Timestamp entries,
        # ``pd.to_datetime(arr).to_numpy()`` returns an OBJECT dtype (tz info
        # preserved as Timestamp objects), NOT the documented datetime64[ns]
        # invariant. Downstream callers (recommended_filter_mask,
        # compute_ml_perf_by_time) silently get a mixed-dtype array; Grouper /
        # comparisons then behave inconsistently. Force the documented contract:
        # coerce to UTC-aware via ``utc=True``, then strip the tz so the
        # resulting numpy view IS datetime64[ns]. WARN-log when tz info gets
        # dropped so the operator knows the audit assumes UTC.
        _coerced = pd.to_datetime(arr, utc=True, errors="coerce")
        _had_tz = (getattr(_coerced.dtype, "tz", None) is not None)
        if _had_tz:
            _coerced = _coerced.tz_convert("UTC").tz_localize(None)
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "coerce_timestamps_for_audit: input contained tz-aware "
                "Timestamps; converted to UTC then tz-stripped so the "
                "returned dtype matches the documented datetime64[ns] "
                "contract. If your data are not UTC-anchored, localise "
                "BEFORE calling this helper."
            )
        return _coerced.to_numpy().astype("datetime64[ns]")

    # pandas 2.0+ ``DatetimeIndex.to_numpy()`` can return ``datetime64[s]`` /
    # ``[ms]`` / ``[us]`` (preserved resolution) instead of always ``[ns]``.
    # Downstream ``.view("int64")`` callers (``_normalize_timestamps``) then
    # read the raw integer in the original unit, not nanoseconds, and
    # epoch-second values collapse to 1970. Force ``datetime64[ns]`` so the
    # documented contract holds across pandas versions.
    def _to_ns(_dti):
        return _dti.to_numpy().astype("datetime64[ns]")

    if arr.size == 0:
        return _to_ns(pd.to_datetime(arr, unit=explicit_unit or "ns"))

    if explicit_unit is not None:
        if explicit_unit not in _AUDIT_UNIT_NS_FACTOR:
            raise ValueError(
                f"audit timestamp unit={explicit_unit!r} not in "
                f"{sorted(_AUDIT_UNIT_NS_FACTOR)!r}"
            )
        return _to_ns(pd.to_datetime(arr, unit=explicit_unit))

    # Auto-detect: coarsest-first so degenerate ns-as-seconds reads pick "s" not "ns".
    _vmin_f = float(np.nanmin(arr))
    _vmax_f = float(np.nanmax(arr))
    for _unit in ("s", "ms", "us", "ns"):
        _ns_factor = _AUDIT_UNIT_NS_FACTOR[_unit]
        _lo_ns = _vmin_f * _ns_factor
        _hi_ns = _vmax_f * _ns_factor
        if _AUDIT_DATETIME_LOW_NS <= _lo_ns and _hi_ns <= _AUDIT_DATETIME_HIGH_NS:
            return _to_ns(pd.to_datetime(arr, unit=_unit))

    logger.warning(
        "audit timestamp values (min=%g, max=%g) fall outside [1970-01-01, 2200-01-01] "
        "under every candidate unit (s/ms/us/ns). Falling back to ns interpretation; "
        "audit may degenerate to a single bin. Set explicit unit to override.",
        _vmin_f, _vmax_f,
    )
    return _to_ns(pd.to_datetime(arr, unit="ns"))
