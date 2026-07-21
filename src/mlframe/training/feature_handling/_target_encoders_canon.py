"""
Category-token canonicalisation + prior/coercion helpers for ``target_encoders.py``.

Carved out of ``target_encoders.py`` (which was nearing the repo's ~800-900 LOC
carve guidance) so ``LeakageSafeEncoder`` itself stays the only thing left in the
parent module. Re-exported from ``target_encoders`` for backward compatibility --
callers should keep importing from there.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Literal,
    Sequence,
)

import numpy as np

if TYPE_CHECKING:
    # Import for the ``pd.Series`` annotation below. Guarded under TYPE_CHECKING
    # so the runtime import cost stays zero while typecheckers / IDE tooling see
    # the real symbol. polars is never used as a bare annotation here (every
    # runtime use below has its own local ``import polars as pl``), so it is
    # not imported at this scope.
    import pandas as pd

logger = logging.getLogger(__name__)


_NULL_SENTINEL = "__NULL__"


def _canonical_cat_token(value) -> str:
    """Per-value category key robust to int<->float dtype drift.

    A bare ``str`` makes the integer ``1`` (``'1'``) and the float ``1.0``
    (``'1.0'``) DIFFERENT keys, so fitting the encoder on an integer-coded
    categorical then transforming the SAME column arriving as float (a routine
    polars int->float promotion / pandas join upcast) misses every per-category
    entry and returns the prior for every row -- a silently wrong encoding.
    Collapse integral-valued numerics to int form so ``1`` and ``1.0`` share a
    key; non-integral floats keep their repr; other labels pass through ``str``.
    Mirrors ``transforms._canonical_group_key`` / ``_internals``'."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
        return repr(f)
    if isinstance(value, np.datetime64):
        return "dt:" + str(int(value.astype("datetime64[ns]").astype("int64")))
    if isinstance(value, np.timedelta64):
        return "dt:" + str(int(value.astype("timedelta64[ns]").astype("int64")))
    # Python datetime/date: emit flavour-neutral epoch-ns token matching the typed-array branches.
    # Naive datetimes are treated as wall-clock (UTC-naive epoch), identical to pandas/polars/numpy
    # so a list-built category and a typed-Series category for the same instant share a key.
    import datetime as _dt
    if isinstance(value, _dt.datetime):
        return "dt:" + str(int(np.datetime64(value, "ns").astype("int64")))
    if isinstance(value, _dt.date):
        return "dt:" + str(int(np.datetime64(value, "ns").astype("int64")))
    return str(value)


def _temporal_to_epoch_ns_tokens(arr_int_ns: np.ndarray, null_mask: np.ndarray) -> np.ndarray:
    """Map an int64 nanosecond-epoch array to canonical ``"dt:<ns>"`` string tokens.

    Datetime category keys diverge by flavour when string-cast: pandas drops the sub-second part
    (``"2020-01-01 13:30:00"``), polars keeps microseconds (``"...13:30:00.000000"``), numpy uses a
    ``T`` separator + nanoseconds (``"2020-01-01T13:30:00.000000000"``). A datetime categorical fit
    as one flavour then transformed as another then misses every key and returns the prior for every
    row -- the same drift the bool/float canonicalisation guards. Nanosecond epoch is flavour-neutral
    and lossless, so all three paths route here for a single canonical token."""
    toks = np.array(["dt:" + str(int(v)) for v in arr_int_ns], dtype=object)
    if null_mask.any():
        toks[null_mask] = _NULL_SENTINEL
    return toks


def _float_canonical_tokens(arr: np.ndarray) -> np.ndarray:
    """Map a float array to canonical per-value tokens (object dtype), collapsing integral floats to int form.

    Hash-based ``pd.factorize(sort=False)`` replaces the prior sort-based ``np.unique(return_inverse=True)``:
    the token is computed per UNIQUE value then gathered back, so the unique ORDER is irrelevant to the output
    -- the result is bit-identical regardless of sort. ~4-9x faster at 10M (argsort O(n log n) -> hash O(n)).
    NaN cells get factorize code -1; callers overwrite those with the null sentinel via their own NaN mask, so
    the (negative-index) gather result for NaN rows is irrelevant. Falls back to ``np.unique`` when pandas is
    unavailable.
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None
    if pd is not None and arr.size:
        codes, uniq = pd.factorize(arr, sort=False)
    else:
        uniq, codes = np.unique(arr, return_inverse=True)
        codes = np.asarray(codes).reshape(-1)
    toks = np.array([_canonical_cat_token(u) for u in uniq], dtype=object)
    if toks.size == 0:
        # All-NaN input under factorize yields empty uniques + all -1 codes; gather would IndexError.
        # Caller's NaN mask overwrites every cell with the sentinel, so the placeholder value is irrelevant.
        return np.full(arr.shape[0], _NULL_SENTINEL, dtype=object)
    return np.asarray(toks[codes])


def _categorical_to_string_array(values: Sequence | np.ndarray | pd.Series) -> np.ndarray:
    """Coerce a sequence of category values to a numpy string array of object dtype. Handles None / NaN
    by mapping them to a sentinel ``"__NULL__"`` so they form their own category rather than being
    silently dropped.

    Vectorised path: numpy / pandas / polars Series take ``.astype(str)`` -- ~50x faster than per-row
    Python loop on 1M+ rows. Plain Python sequences (list / tuple) fall back to the legacy loop.
    """
    # Fast path: pandas Series with vectorised str cast.
    try:
        import pandas as pd
        if isinstance(values, pd.Series):
            mask_null = values.isna()
            if values.dtype.kind == "f":
                # Float column: canonicalise integral values to int form so a
                # fit-int / transform-float drift (or vice versa) on the SAME
                # integer-coded categorical still hits the per-category entry
                # instead of returning the prior for every row.
                arr = values.to_numpy()
                out = _float_canonical_tokens(arr)
            elif values.dtype.kind in ("M", "m"):
                # Datetime / timedelta: route through flavour-neutral epoch-ns tokens.
                nm = mask_null.to_numpy()
                ns = values.to_numpy(dtype="datetime64[ns]" if values.dtype.kind == "M" else "timedelta64[ns]").astype("int64")
                return _temporal_to_epoch_ns_tokens(ns, nm)
            elif values.dtype.kind == "O":
                # Object dtype can embed python ints/floats/dates/datetimes; canonicalise per value so
                # an int 1 / float 1.0 collapse and python date/datetime objects emit the same epoch-ns
                # token the typed Datetime branches do (parity with the numpy object-array branch).
                arr = values.to_numpy()
                out = np.array([_canonical_cat_token(v) for v in arr], dtype=object)
            else:
                out = values.astype(str).to_numpy(dtype=object)
            if mask_null.any():
                out[mask_null.to_numpy()] = _NULL_SENTINEL
            return out
    except ImportError:
        pass
    # Fast path: polars Series.
    try:
        import polars as pl
        if isinstance(values, pl.Series):
            mask_null = np.asarray(values.is_null().to_numpy())
            if values.dtype in (pl.Float32, pl.Float64):
                # Float column: canonicalise integral values to int form (see
                # the pandas-float branch) so int<->float dtype drift on the
                # same integer-coded categorical does not miss every key.
                arr = values.to_numpy()
                out = _float_canonical_tokens(arr)
                # polars ``is_null()`` does NOT flag NaN, and the float token for
                # NaN is ``repr(nan) == "nan"`` (not "NaN"), so the string rebrand
                # below misses it. Mask NaN directly off the numeric array for
                # parity with the pandas/numpy branches (NaN -> sentinel).
                mask_null = mask_null | np.isnan(arr)
            elif values.dtype == pl.Boolean:
                # polars casts bool to lowercase ``"true"``/``"false"``, but the numpy/pandas/list paths
                # emit canonical ``"True"``/``"False"`` via ``_canonical_cat_token``. Without this branch a
                # bool categorical fit as polars then transformed as pandas (or vice versa) misses every key
                # and returns the prior for every row -- the same drift the float canonicalisation guards.
                arr = values.to_numpy()
                out = np.where(mask_null, _NULL_SENTINEL, np.where(arr, "True", "False")).astype(object)
                return out
            elif values.dtype.is_temporal():
                # Datetime / Date / Time / Duration: route through flavour-neutral epoch-ns tokens.
                # polars Utf8 cast keeps microseconds while pandas drops sub-second + numpy uses
                # ``T``/nanoseconds, so a temporal categorical fit one flavour then transformed another
                # misses every key. Cast to ns-resolution physical int for a single canonical token.
                if values.dtype == pl.Date:
                    phys_s = values.cast(pl.Datetime("ns")).to_physical()
                elif values.dtype == pl.Time:
                    phys_s = values.to_physical()
                elif values.dtype == pl.Duration:
                    phys_s = values.cast(pl.Duration("ns")).to_physical()
                else:
                    phys_s = values.cast(pl.Datetime("ns")).to_physical()
                # fill_null(0) before numpy avoids the null->NaN->int64 invalid-cast warning; the
                # null_mask overwrites those slots with the sentinel so the filler value is irrelevant.
                phys = phys_s.fill_null(0).to_numpy().astype("int64")
                return _temporal_to_epoch_ns_tokens(phys, mask_null)
            else:
                # cast to Utf8 then numpy; ``__NULL__`` overwrites nulls (polars cast yields ``None``).
                out = values.cast(pl.Utf8).to_numpy().astype(object)
            if mask_null.any():
                out[mask_null] = _NULL_SENTINEL
            # Float NaN cells survive cast as ``"NaN"``; rebrand to sentinel for parity with pandas path.
            nan_mask = out == "NaN"
            if nan_mask.any():
                out[nan_mask] = _NULL_SENTINEL
            return out
    except ImportError:
        pass
    # Fast path: numpy array.
    if isinstance(values, np.ndarray):
        if values.dtype.kind == "f":
            mask = np.isnan(values)
            # Canonicalise integral float values to int form (int<->float drift).
            out = _float_canonical_tokens(values)
            if mask.any():
                out[mask] = _NULL_SENTINEL
            return out
        if values.dtype.kind in ("O", "U", "S"):
            # Object/string arrays can embed python ints/floats; canonicalise
            # per value so an int 1 and a float 1.0 in the SAME logical category
            # collapse to one key.
            null_mask = _objectwise_isnull(values)
            toks = np.array([_canonical_cat_token(v) for v in values], dtype=object)
            out = np.where(null_mask, _NULL_SENTINEL, toks).astype(object)
            return out
        if values.dtype.kind in ("M", "m"):
            # datetime64 / timedelta64: route through flavour-neutral epoch-ns tokens (see pandas branch).
            null_mask = np.isnat(values)
            ns = values.astype("datetime64[ns]" if values.dtype.kind == "M" else "timedelta64[ns]").astype("int64")
            return _temporal_to_epoch_ns_tokens(ns, null_mask)
        return values.astype(str).astype(object)
    # Generic iterable / list / tuple: per-row loop is fine (length typically <= a few thousand for tests).
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        if v is None:
            out[i] = _NULL_SENTINEL
            continue
        if isinstance(v, float) and np.isnan(v):
            out[i] = _NULL_SENTINEL
            continue
        out[i] = _canonical_cat_token(v)
    return out


def _objectwise_isnull(arr: np.ndarray) -> np.ndarray:
    """Vectorised None / NaN mask for object-dtype arrays.

    Pre-fix this returned an all-False mask on per-row introspection failure,
    so target-encoder fit silently treated None/NaN rows as valid encoded
    values (their non-numeric content getting averaged into the per-category
    mean / WoE). The "fall back per-row when ufuncs fail" docstring was
    aspirational; the fallback was actually "pretend nothing is null."

    Post-fix: try pandas.isna (the proper vectorised fallback for object-dtype
    arrays). If THAT also fails, raise: silent target-encoder corruption is
    strictly worse than a loud failure that surfaces the bad input early.
    """
    try:
        is_none = np.array([x is None for x in arr], dtype=bool)
    except Exception as _e_none:
        # Per-row `is None` should never raise on a finite-iterable object
        # array; if it does, the input is malformed enough that pandas.isna
        # is the right tool. Don't silently zero - that's the bug we're
        # fixing.
        import pandas as _pd
        try:
            is_none = _pd.isna(arr).astype(bool)
            # pandas.isna already covers NaN for floats; return early.
            return np.asarray(is_none)
        except Exception as _e_pd:
            raise ValueError(
                f"_objectwise_isnull: both per-row `is None` ({_e_none}) and "
                f"pandas.isna ({_e_pd}) failed on object array. Refusing to "
                f"return an all-False mask, which would silently corrupt "
                f"target-encoder fit by treating null rows as valid."
            ) from _e_pd
    try:
        is_nan = np.array(
            [isinstance(x, float) and np.isnan(x) for x in arr],
            dtype=bool,
        )
    except Exception as _e_nan:
        import pandas as _pd
        try:
            is_nan = _pd.isna(arr).astype(bool)
        except Exception as _e_pd:
            raise ValueError(f"_objectwise_isnull: per-row NaN check failed ({_e_nan}) and " f"pandas.isna fallback also failed ({_e_pd}).") from _e_pd
    return np.asarray(is_none | is_nan)


def _coerce_y_to_float64(y) -> np.ndarray:
    """Backend-agnostic float64 coercion. Avoids ``np.asarray(list(y))`` which materialises a Python list
    first (doubles memory + drops dtype on pandas/polars Series). Native ``.to_numpy()`` is zero-copy
    when dtype already matches.

    Return-value contract: read-only. When ``y`` is already a float64 ndarray (or a float64
    pandas/polars Series), the returned array shares storage with the caller's input; in-place
    mutation would corrupt the caller's target. All current callers (``_compute_prior``,
    ``_compute_per_category``, ``_compute_woe_per_category``) treat the result as read-only.
    Take ``.copy()`` at the call site if you need to mutate."""
    if isinstance(y, np.ndarray):
        return y.astype(np.float64, copy=False)
    try:
        import pandas as pd
        if isinstance(y, (pd.Series, pd.DataFrame)):
            return np.asarray(y.to_numpy(dtype=np.float64, copy=False)).ravel()
    except ImportError:
        pass
    try:
        import polars as pl
        if isinstance(y, pl.Series):
            return y.cast(pl.Float64).to_numpy().ravel()
    except ImportError:
        pass
    return np.asarray(y, dtype=np.float64).ravel()


def _compute_prior(y: np.ndarray, prior_kind: Literal["mean", "median"], sample_weight: np.ndarray | None = None) -> float:
    """Global fallback statistic (mean or median of ``y``) used to smooth/backfill rare or unseen categories; supports sample weights (weighted median via cumulative-weight cutoff), and returns 0.0 for empty ``y``."""
    # np.mean/np.median([]) returns NaN with RuntimeWarning, and the weighted-median branch's
    # y[order[idx]] raises IndexError on empty y. _oof_encode is safe (KFold guarantees nonempty
    # train_idx) but the direct fit/_loo_encode paths receive caller-supplied y. Treat empty as no-evidence.
    if len(y) == 0:
        logger.warning("_compute_prior: empty y; returning 0.0 as no-evidence prior.")
        return 0.0
    if sample_weight is None:
        if prior_kind == "median":
            return float(np.median(y))
        return float(np.mean(y))
    # Weighted prior. np.median has no native sample_weight; emulate via sorted cumulative weights for the median
    # branch so weighted_prior(uniform_w) == np.median(y) (up to ties).
    sw = np.asarray(sample_weight, dtype=np.float64)
    total = float(sw.sum())
    if total <= 0:
        if prior_kind == "median":
            return float(np.median(y))
        return float(np.mean(y))
    if prior_kind == "median":
        order = np.argsort(y)
        cw = np.cumsum(sw[order])
        cutoff = 0.5 * total
        # First index with cumulative weight >= cutoff is the weighted median.
        idx = int(np.searchsorted(cw, cutoff))
        idx = min(idx, len(y) - 1)
        return float(y[order[idx]])
    return float(np.dot(sw, y) / total)


__all__ = [
    "_canonical_cat_token",
    "_temporal_to_epoch_ns_tokens",
    "_float_canonical_tokens",
    "_categorical_to_string_array",
    "_objectwise_isnull",
    "_coerce_y_to_float64",
    "_compute_prior",
]
