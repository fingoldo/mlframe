"""Automatic causal base-column engineering for composite-target discovery.

Composite-target discovery needs *base* columns: predictors a composite is built on
top of (``y ~ f(base) + residual``). For TEMPORAL targets the most valuable bases are
mechanical -- a lag of the target, a trailing rolling statistic of the target, a
first-difference -- yet today the user must hand-supply them. :func:`engineer_temporal_bases`
constructs that family automatically from a time-ordered frame.

CARDINAL RULE -- strict causality (no leakage): every engineered base at row ``i`` uses
ONLY rows whose time is strictly earlier than ``t_i``. The current target value (shift 0)
and any future value never enter a base. Concretely:

* ``lag_k``        : ``y`` shifted forward by ``k >= 1`` (row ``i`` sees ``y[i-k]``).
* ``rolling_mean`` /
  ``rolling_median``: trailing window over ``y[i-w .. i-1]`` -- the window EXCLUDES row
                     ``i`` (it is computed on ``y`` shifted by 1 first, so it can never
                     read the current or any future target).
* ``diff``         : ``y[i-1] - y[i-2]`` (a lag-1 first difference; both operands are
                     strictly past, again never the current row).

The head rows that lack enough history are ``NaN`` (a lag-1 column's first row, a
window-``w`` column's first ``w`` rows). Downstream screening already masks non-finite
rows pairwise, so the NaN head is the correct, honest encoding of "no causal history yet"
-- never back-filled (back-fill would smear a later value into an earlier row = leakage).

Because every base is shift ``>= 1``, none is a same-time re-encoding of ``y``, so
:func:`mlframe.training.composite.discovery._leakage.detect_base_target_leakage` does NOT
flag them: against the current ``y`` a true lag leaves a real residual, and with the time
ordering supplied its lag-probe recognises the shift as a genuine lag rather than leakage.

Memory: the input frame is never copied. Only the target column is pulled once as a single
ndarray (the narrow ``_extract_base``-style pull), and the engineered bases are returned as
a small ``{name: ndarray}`` dict -- a handful of length-``n`` float64 vectors, not a frame.
:func:`add_engineered_bases_to_pool` is a thin helper that merges these into an existing
candidate-base pool (e.g. for :func:`screen_base_pool`) without touching the frame.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _extract_column(df: Any, col: str) -> np.ndarray:
    """Pull a single column as a 1-D float64 ndarray (polars or pandas), no frame copy.

    Mirrors the narrow ``_extract_base`` convention used elsewhere in the composite
    package (classification.py): one column -> one ndarray, never ``to_pandas()`` /
    ``clone()`` on the whole frame.
    """
    getcol = getattr(df, "get_column", None)
    if callable(getcol):  # polars
        arr = df.get_column(col).to_numpy()
    else:  # pandas / mapping-like
        arr = df[col].to_numpy() if hasattr(df[col], "to_numpy") else np.asarray(df[col])
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _time_order(df: Any, time_column: str, n: int) -> np.ndarray:
    """Stable argsort of the time column -- the row order engineered bases are causal in.

    A stable (mergesort) sort keeps ties (equal timestamps) in their original frame order,
    so a deterministic, reproducible causal ordering is used even with duplicate times.
    """
    t = _extract_column(df, time_column)
    if t.shape[0] != n:
        raise ValueError(f"time column '{time_column}' length {t.shape[0]} != target length {n}")
    return np.argsort(t, kind="mergesort")


def _causal_lag(y_sorted: np.ndarray, k: int) -> np.ndarray:
    """``y`` shifted forward by ``k >= 1``: row ``i`` gets ``y[i-k]`` (first ``k`` NaN)."""
    n = y_sorted.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if k < n:
        out[k:] = y_sorted[:-k] if k > 0 else y_sorted
    return out


def _causal_rolling(y_sorted: np.ndarray, window: int, *, median: bool) -> np.ndarray:
    """Trailing rolling mean/median over the ``window`` rows STRICTLY BEFORE each row.

    The statistic at row ``i`` is over ``y[i-window .. i-1]`` (current row excluded). The
    first ``window`` rows are NaN (insufficient strictly-past history). Computed on the
    shift-1 series so the current target can never enter the window -- causal by construction.
    """
    n = y_sorted.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1:
        return out
    for i in range(window, n):
        past = y_sorted[i - window:i]  # strictly earlier than row i
        out[i] = np.median(past) if median else past.mean()
    return out


def _causal_diff(y_sorted: np.ndarray) -> np.ndarray:
    """First difference of the PAST: row ``i`` gets ``y[i-1] - y[i-2]`` (first 2 NaN).

    Both operands are strictly-past targets, so the current row's target never enters --
    this is a lagged difference, distinct from the leaky same-time ``y[i] - y[i-1]``.
    """
    n = y_sorted.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n >= 3:
        out[2:] = y_sorted[1:-1] - y_sorted[:-2]
    return out


_VALID_OPS = ("lag", "rolling_mean", "rolling_median", "diff")


def engineer_temporal_bases(
    df: Any,
    target_col: str,
    time_column: str,
    *,
    lags: Sequence[int] = (1, 2, 3),
    rolling_windows: Sequence[int] = (3, 5),
    ops: Sequence[str] = ("lag", "rolling_mean", "diff"),
) -> dict[str, np.ndarray]:
    """Build a family of STRICTLY CAUSAL temporal base columns from the target series.

    The frame is sorted by ``time_column`` (stable), the engineered bases are computed in
    that order, then mapped BACK to the frame's original row order so each returned ndarray
    aligns positionally with ``df`` (the caller can hand them straight to discovery / leakage
    screening without re-sorting).

    Parameters
    ----------
    df : polars or pandas frame (NOT copied -- only ``target_col`` and ``time_column`` are
        pulled, as single ndarrays).
    target_col : name of the target column ``y`` the bases are derived from.
    time_column : name of the time/order column used to establish causality.
    lags : lag offsets ``k >= 1`` for the ``"lag"`` op. Non-positive lags are rejected
        (lag 0 would be the current target = leakage).
    rolling_windows : trailing window sizes ``w >= 1`` for rolling ops (window excludes the
        current row).
    ops : which families to build; subset of ``{"lag", "rolling_mean", "rolling_median",
        "diff"}``.

    Returns
    -------
    dict ``{engineered_base_name -> ndarray}`` in original-frame row order. Names are
    ``"<target>_lag<k>"``, ``"<target>_rollmean<w>"``, ``"<target>_rollmedian<w>"``,
    ``"<target>_diff1"``. Head rows lacking causal history are ``NaN``.

    Raises
    ------
    ValueError on an unknown op, a non-positive lag, or a non-positive window.
    """
    bad_ops = [o for o in ops if o not in _VALID_OPS]
    if bad_ops:
        raise ValueError(f"unknown op(s) {bad_ops}; valid: {_VALID_OPS}")
    if any(int(k) < 1 for k in lags):
        raise ValueError("lags must be >= 1 (lag 0 is the current target = leakage)")
    if any(int(w) < 1 for w in rolling_windows):
        raise ValueError("rolling_windows must be >= 1")

    y = _extract_column(df, target_col)
    n = y.shape[0]
    order = _time_order(df, time_column, n)
    y_sorted = y[order]
    # Inverse permutation: maps a sorted-order vector back to original frame row order.
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)

    out: dict[str, np.ndarray] = {}

    def _emit(name: str, sorted_vec: np.ndarray) -> None:
        out[name] = sorted_vec[inv]  # back to original row order

    if "lag" in ops:
        for k in lags:
            _emit(f"{target_col}_lag{int(k)}", _causal_lag(y_sorted, int(k)))
    if "rolling_mean" in ops:
        for w in rolling_windows:
            _emit(f"{target_col}_rollmean{int(w)}", _causal_rolling(y_sorted, int(w), median=False))
    if "rolling_median" in ops:
        for w in rolling_windows:
            _emit(f"{target_col}_rollmedian{int(w)}", _causal_rolling(y_sorted, int(w), median=True))
    if "diff" in ops:
        _emit(f"{target_col}_diff1", _causal_diff(y_sorted))

    return out


def add_engineered_bases_to_pool(
    pool: Mapping[str, np.ndarray] | None,
    df: Any,
    target_col: str,
    time_column: str,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Merge auto-engineered temporal bases into a candidate-base pool (no frame copy).

    Thin convenience wrapper: builds the engineered bases via
    :func:`engineer_temporal_bases` and returns a NEW dict combining ``pool`` (if any) with
    them. User-supplied entries in ``pool`` win on a name clash (never silently overwritten).
    The result is ready to pass to ``screen_base_pool`` / discovery as candidate bases.

    ``**kwargs`` are forwarded to :func:`engineer_temporal_bases` (``lags``,
    ``rolling_windows``, ``ops``).
    """
    engineered = engineer_temporal_bases(df, target_col, time_column, **kwargs)
    merged: dict[str, np.ndarray] = dict(engineered)
    if pool:
        merged.update(pool)  # caller-supplied bases take precedence on name clash
    return merged
