"""Basic feature engineering for ML."""

from __future__ import annotations


__all__ = [
    "create_date_features",
    "add_cyclical_date_features",
    "run_pysr_fe",
]

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

import os

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - numba is a hard dep in practice
    prange = range

    def njit(*args, **kwargs):  # no-op fallback so the module imports
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco


# Below this row count the prange thread-launch floor (~17 ms on the dev box) dwarfs the per-element sin/cos work, so the serial kernel wins; above it the
# embarrassingly-parallel split scales near-linearly (12.3x @ 10M on a 6-col date frame). Env-overridable for hosts with a cheaper/dearer thread pool.
_CYCLICAL_PAR_THRESHOLD = int(os.environ.get("MLFRAME_CYCLICAL_PAR_THRESHOLD", "1000000"))


@njit(cache=True)
def _cyclical_sincos_serial(base: np.ndarray, scale: float):
    n = base.size
    s = np.empty(n, dtype=np.float32)
    c = np.empty(n, dtype=np.float32)
    for i in range(n):
        a = base[i] * scale
        s[i] = np.float32(math.sin(a))
        c[i] = np.float32(math.cos(a))
    return s, c


@njit(parallel=True, cache=True)
def _cyclical_sincos_parallel(base: np.ndarray, scale: float):
    n = base.size
    s = np.empty(n, dtype=np.float32)
    c = np.empty(n, dtype=np.float32)
    for i in prange(n):
        a = base[i] * scale
        s[i] = np.float32(math.sin(a))
        c[i] = np.float32(math.cos(a))
    return s, c


def _cyclical_sincos_njit(base: np.ndarray, scale: float):
    """Fused single-pass sin/cos of ``base*scale``, both emitted directly as
    float32. Replaces the numpy 2-pass (np.sin(f64) + np.cos(f64)) + double
    .astype(float32) which walked the angle array 4x and allocated two f64
    temporaries. ~1.8x; results match the numpy form to <1e-6 (well within the
    float32 output precision the function already casts to + the sin^2+cos^2==1
    invariant the regression test checks).

    Size-dispatched: each output element is independent (no reduction), so the parallel prange twin is BIT-IDENTICAL to the serial loop and wins by ~12x at
    large n; below ``_CYCLICAL_PAR_THRESHOLD`` the serial kernel avoids the prange thread-launch floor."""
    if base.size >= _CYCLICAL_PAR_THRESHOLD:
        return _cyclical_sincos_parallel(base, scale)
    return _cyclical_sincos_serial(base, scale)

from pyutilz.system import clean_ram, get_own_memory_usage

logger = logging.getLogger(__name__)

# Map of numpy integer dtypes to polars equivalents. Defined at module scope so it's not
# rebuilt per call. Add new entries here if a user passes a wider integer dtype.
_NP_TO_PL_DTYPE: Dict[type, pl.DataType] = {
    np.int8: pl.Int8,
    np.int16: pl.Int16,
    np.int32: pl.Int32,
    np.int64: pl.Int64,
    np.uint8: pl.UInt8,
    np.uint16: pl.UInt16,
    np.uint32: pl.UInt32,
    np.uint64: pl.UInt64,
    np.bool_: pl.Boolean,
    bool: pl.Boolean,
}

# Method-name -> (pandas accessor, polars accessor) alias map. Lets callers use the same
# stable name ("week_of_year", "day_of_year") across backends; we route to whichever native
# accessor each library exposes. ``None`` on the polars side means "needs a custom expression"
# and is handled in the polars branch below.
_DATE_METHOD_ALIASES: Dict[str, Tuple[str, Optional[str]]] = {
    "year": ("year", "year"),
    "month": ("month", "month"),
    "day": ("day", "day"),
    "weekday": ("weekday", "weekday"),
    "hour": ("hour", "hour"),
    "minute": ("minute", "minute"),
    "second": ("second", "second"),
    "quarter": ("quarter", "quarter"),
    "week_of_year": ("isocalendar", "week"),
    "day_of_year": ("dayofyear", "ordinal_day"),
    "is_weekend": (None, None),
}

# Kaggle-style extended default: year (int32 to avoid 2128 overflow), the legacy day/weekday/month
# trio (int8), plus quarter / week_of_year / is_weekend / day_of_year for multi-scale seasonality.
# See ``docs/date_features_kaggle_research.md`` for the rationale.
_DEFAULT_DATE_METHODS: Dict[str, type] = {
    "year": np.int32,
    "quarter": np.int8,
    "month": np.int8,
    "week_of_year": np.int8,
    "day": np.int8,
    "day_of_year": np.int16,
    "weekday": np.int8,
    "is_weekend": np.bool_,
}

# Default periods for ``add_cyclical_date_features``. ``day`` here is day-of-month (period 31);
# ``day_of_year`` is the finer-grained annual cycle (period 365.25).
_DEFAULT_CYCLICAL_PERIODS: Tuple[Tuple[str, float], ...] = (
    ("hour", 24.0),
    ("day", 31.0),
    ("weekday", 7.0),
    ("month", 12.0),
    ("day_of_year", 365.25),
)


def _collect_pandas_tz(series: pd.Series) -> str:
    """Return a tz tag for a pandas datetime series: ``str(tz)`` or ``"naive"``.

    Used to detect mixed-tz columns at the FE boundary. Non-datetime columns return ``"naive"``
    so the caller's downstream type-check (the existing ``.dt`` access) still surfaces the
    real type error.
    """
    tz = getattr(series.dtype, "tz", None)
    if tz is None:
        return "naive"
    return str(tz)


def _collect_polars_tz(df: pl.DataFrame, col: str) -> str:
    """Return a tz tag for a polars datetime column: ``str(tz)`` or ``"naive"``."""
    dtype = df.schema.get(col)
    tz = getattr(dtype, "time_zone", None)
    if tz is None:
        return "naive"
    return str(tz)


def _warn_on_mixed_tz(df: Union[pd.DataFrame, pl.DataFrame], cols: List[str]) -> None:
    """Warn (do NOT raise / convert) when ``cols`` span multiple timezones.

    Multi-tz columns silently produce nonsense ``hour`` / ``weekday`` features because the same
    instant maps to different local-times across rows. We surface the concrete tz list so the
    caller can normalise upstream; we never auto-convert (caller may legitimately want both).
    """
    if len(cols) < 2:
        return
    is_pandas = isinstance(df, pd.DataFrame)
    tzs = []
    for c in cols:
        if c not in df.columns:
            continue
        if is_pandas:
            tzs.append((c, _collect_pandas_tz(df[c])))
        else:
            tzs.append((c, _collect_polars_tz(df, c)))
    unique_tzs = sorted({t for _, t in tzs})
    if len(unique_tzs) > 1:
        col_tz_pairs = ", ".join(f"{c}={t}" for c, t in tzs)
        logger.warning(
            "create_date_features: columns span multiple timezones (%s); extracted hour/weekday/day fields will be inconsistent across rows. "
            "Observed tz list: %s. Normalise to a single tz upstream (e.g. convert all to UTC) before calling.",
            col_tz_pairs,
            unique_tzs,
        )


_MISSING_DT_FIELD = object()


def _resolve_pandas_method(series_dt, method: str, dtype) -> pd.Series:
    """Resolve a logical method name against the pandas .dt accessor and cast.

    Handles the alias map (``week_of_year`` -> ``isocalendar().week``, ``day_of_year`` ->
    ``dayofyear``, ``is_weekend`` -> ``weekday >= 5``) and falls back to ``getattr`` for any
    accessor name we haven't aliased.

    Resolution uses ``getattr(..., sentinel)`` rather than ``hasattr`` then
    ``getattr``: a pandas ``.dt`` field accessor (``.hour`` / ``.month`` /
    ``.dayofyear`` ...) is a COMPUTED property, so ``hasattr`` decodes the whole
    field from the int64-ns array just to test existence, then ``getattr``
    decodes it a SECOND time. The sentinel form extracts each field once -- 1.91x
    on a 5-field / 200k-row extract -- and is bit-identical (same field, same
    cast). Also speeds ``create_date_features``, which shares this helper.
    """
    if method == "is_weekend":
        return (series_dt.weekday >= 5).astype(dtype)
    if method == "week_of_year":
        return series_dt.isocalendar().week.astype(dtype)
    pd_name = _DATE_METHOD_ALIASES.get(method, (method, None))[0]
    field = _MISSING_DT_FIELD
    if pd_name is not None:
        field = getattr(series_dt, pd_name, _MISSING_DT_FIELD)
    if field is _MISSING_DT_FIELD:
        field = getattr(series_dt, method, _MISSING_DT_FIELD)
        if field is _MISSING_DT_FIELD:
            raise ValueError(f"Unknown pandas .dt accessor: {method!r}")
    return field.astype(dtype)


def _resolve_polars_expr(col: str, method: str, pl_dtype: pl.DataType) -> pl.Expr:
    """Resolve a logical method name into a polars expression and cast.

    Mirrors ``_resolve_pandas_method`` for the polars branch; subtracts 1 from ``weekday`` so
    both backends agree on Mon=0..Sun=6.
    """
    col_dt = pl.col(col).dt
    if method == "is_weekend":
        return ((col_dt.weekday() - 1) >= 5).cast(pl_dtype).alias(col + "_" + method)
    if method == "weekday":
        return (col_dt.weekday() - 1).cast(pl_dtype).alias(col + "_" + method)
    pl_name = _DATE_METHOD_ALIASES.get(method, (None, method))[1]
    if pl_name is None or not hasattr(col_dt, pl_name):
        if not hasattr(col_dt, method):
            raise ValueError(f"Unknown polars .dt accessor: {method!r}")
        pl_name = method
    return getattr(col_dt, pl_name)().cast(pl_dtype).alias(col + "_" + method)


def create_date_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    cols: List[str],
    delete_original_cols: bool = True,
    methods: Optional[Dict[str, type]] = None,
    add_cyclical: bool = True,
    cyclical_periods: Optional[Sequence[Tuple[str, float]]] = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Decompose datetime columns into integer date parts (year, day, weekday, month, ...).

    Parameters
    ----------
    df
        pandas or polars frame to augment.
    cols
        Datetime column names to decompose. Empty list returns ``df`` unchanged.
    delete_original_cols
        Drop the source datetime columns after decomposition.
    methods
        Mapping of logical accessor name to target numpy dtype. Defaults to a Kaggle-style
        extended set: ``year`` (int32), ``quarter`` (int8), ``month`` (int8), ``week_of_year``
        (int8), ``day`` (int8), ``day_of_year`` (int16), ``weekday`` (int8), ``is_weekend``
        (bool). Weekday is normalised to pandas convention (Mon=0 .. Sun=6) in both backends.
        Custom callers may pass any subset of the recognised aliases (see
        ``_DATE_METHOD_ALIASES``) or any native pandas / polars ``.dt`` accessor name.
    add_cyclical
        If True (default), also emit sin/cos pairs for each ``(period_name, period)`` in
        ``cyclical_periods`` (defaults to hour/day/weekday/month/day_of_year). Each pair is
        float32, normalised to [-1, 1]. Trees ignore the redundancy; linear / kNN / NN
        learners need the cyclical encoding to model the Dec->Jan adjacency. See
        ``add_cyclical_date_features`` for the same functionality as a standalone helper.
    cyclical_periods
        Iterable of ``(period_name, period_value)`` tuples consumed only when
        ``add_cyclical=True``. ``period_name`` must be a recognised method alias.

    Returns
    -------
    A new frame (same backend as input). The input is never mutated.

    Notes
    -----
    Multi-timezone columns trigger a single WARNING listing every observed tz (including the
    tz-naive bucket). The function never auto-converts -- the caller must normalise upstream
    (e.g. tz-convert all to UTC at ingest) if the intended semantics is "same instant".
    """
    if methods is None:
        methods = dict(_DEFAULT_DATE_METHODS)
    if len(cols) == 0:
        return df

    logger.debug("In create_date_features. RAM usage: %.1fGB.", get_own_memory_usage())

    is_pandas = isinstance(df, pd.DataFrame)
    is_polars = isinstance(df, pl.DataFrame)
    if not (is_pandas or is_polars):
        raise ValueError("df must be pandas or polars DataFrame")

    _warn_on_mixed_tz(df, cols)

    existing_cols = set(df.columns)
    derived_names = [f"{col}_{method}" for col in cols for method in methods.keys()]
    clashes = [n for n in derived_names if n in existing_cols]
    if clashes:
        logger.warning(
            "create_date_features: %d derived column name(s) already exist in the DataFrame and will be OVERWRITTEN: %s. "
            "If any of these are user-engineered features, rename either them or the ``methods`` dict keys to disambiguate.",
            len(clashes),
            clashes,
        )

    precomputed_bases: Dict[Tuple[str, str], np.ndarray] = {}
    if is_pandas:
        df = df.copy(deep=False)
        # Accumulate every derived field and insert them in ONE pd.concat instead of column-at-a-time
        # assignment (which fragments the block manager -> PerformanceWarning + ~O(n_cols^2) copies).
        new_cols: Dict[str, pd.Series] = {}
        for col in cols:
            obj = df[col].dt
            for method, dtype in methods.items():
                field = _resolve_pandas_method(obj, method, dtype)
                new_cols[col + "_" + method] = field
                # Cache the just-extracted integer field so the cyclical pass below reuses it instead of re-decoding .dt.
                if add_cyclical and method != "is_weekend":
                    precomputed_bases[(col, method)] = field.to_numpy()
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1, copy=False)
    else:
        all_exprs = []
        for col in cols:
            for method, np_dtype in methods.items():
                pl_dtype = _NP_TO_PL_DTYPE.get(np_dtype)
                if pl_dtype is None:
                    raise ValueError(
                        f"Unsupported dtype {np_dtype} for polars; supported: {list(_NP_TO_PL_DTYPE.keys())}"
                    )
                all_exprs.append(_resolve_polars_expr(col, method, pl_dtype))
        df = df.with_columns(all_exprs)

    # ``add_cyclical`` MUST run before ``delete_original_cols`` drops the source datetimes -- the cyclical helper reads ``df[col].dt`` directly off the source column rather than recomputing from the just-emitted integer fields.
    if add_cyclical:
        df = add_cyclical_date_features(
            df, cols=cols, periods=cyclical_periods,
            delete_original_cols=False,
            _precomputed_bases=precomputed_bases if is_pandas else None,
        )

    if delete_original_cols:
        if is_pandas:
            df = df.drop(columns=cols)
        else:
            df = df.drop(cols)

    return df


def add_cyclical_date_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    cols: List[str],
    periods: Optional[Sequence[Tuple[str, float]]] = None,
    delete_original_cols: bool = False,
    _precomputed_bases: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Append Kaggle-style sin/cos cyclical encodings of date components.

    For each ``(period_name, period)`` pair and each source column ``c``, emits
    ``<c>_<period_name>_sin`` and ``<c>_<period_name>_cos`` as float32. The pair captures
    the wrap-around adjacency that integer encoding loses (e.g. ``month=12`` is adjacent to
    ``month=1``); essential for linear models and NN blenders, harmless for trees.

    Parameters
    ----------
    df
        pandas or polars frame containing the datetime columns.
    cols
        Datetime column names to derive cyclical features from. The source columns are read
        directly (the function does NOT depend on ``create_date_features`` having been called
        first).
    periods
        Iterable of ``(period_name, period_value)`` tuples. Defaults to
        ``(("hour", 24), ("day", 31), ("weekday", 7), ("month", 12), ("day_of_year", 365.25))``.
        ``period_name`` must be a recognised method alias (see ``_DATE_METHOD_ALIASES``).
    delete_original_cols
        Drop the source datetime columns after extraction. Defaults to ``False`` because
        callers typically chain this after ``create_date_features`` and want to preserve the
        integer fields too.
    _precomputed_bases
        Internal pandas-only fast path: a ``{(col, period_name): int_field_array}`` map of date fields already extracted by
        ``create_date_features``. When a ``(col, period_name)`` is present its integer field is reused as the sin/cos base
        instead of re-decoding the int64-ns array via ``.dt`` (the integer field cast to float64 equals the direct float
        extraction bit-for-bit for day/weekday/month/day_of_year). Saves ~1.27x e2e at 1M rows on the default method+period set.

    Returns
    -------
    A new frame (same backend as input) with the sin/cos pair columns appended.

    Notes
    -----
    * Output dtype is float32 (sufficient precision for [-1, 1] values, halves memory vs
      float64).
    * ``sin^2 + cos^2 == 1`` to within float32 precision; useful invariant for tests.
    * Multi-tz columns trigger the same WARN as ``create_date_features``.
    """
    if len(cols) == 0:
        return df
    if periods is None:
        periods = _DEFAULT_CYCLICAL_PERIODS

    is_pandas = isinstance(df, pd.DataFrame)
    is_polars = isinstance(df, pl.DataFrame)
    if not (is_pandas or is_polars):
        raise ValueError("df must be pandas or polars DataFrame")

    _warn_on_mixed_tz(df, cols)

    for period_name, _ in periods:
        if period_name not in _DATE_METHOD_ALIASES:
            raise ValueError(
                f"Unknown cyclical period name: {period_name!r}. Recognised names: {sorted(_DATE_METHOD_ALIASES.keys())}"
            )

    two_pi = float(2.0 * np.pi)

    if is_pandas:
        df = df.copy(deep=False)
        # Batch all sin/cos columns into one pd.concat (see create_date_features) to avoid
        # block-manager fragmentation from column-at-a-time assignment.
        new_cols: Dict[str, np.ndarray] = {}
        for col in cols:
            obj = df[col].dt
            for period_name, period_value in periods:
                precomputed = None if _precomputed_bases is None else _precomputed_bases.get((col, period_name))
                if precomputed is not None:
                    # The integer date field already extracted by create_date_features equals the float field
                    # bit-for-bit (verified: integer-cast-to-float == direct float extraction for day/weekday/month/day_of_year),
                    # so reuse it instead of decoding the int64-ns array a second time.
                    base = np.ascontiguousarray(precomputed, dtype=np.float64)
                else:
                    base = _resolve_pandas_method(obj, period_name, np.float64).to_numpy()
                    base = np.ascontiguousarray(base, dtype=np.float64)
                s, c = _cyclical_sincos_njit(base, two_pi / float(period_value))
                new_cols[f"{col}_{period_name}_sin"] = s
                new_cols[f"{col}_{period_name}_cos"] = c
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1, copy=False)
        if delete_original_cols:
            df = df.drop(columns=cols)
    else:
        all_exprs = []
        for col in cols:
            col_dt = pl.col(col).dt
            for period_name, period_value in periods:
                if period_name == "is_weekend":
                    raise ValueError(
                        "is_weekend is a binary indicator, not periodic; cyclical encoding is meaningless. Drop it from `periods`."
                    )
                if period_name == "weekday":
                    base_expr = (col_dt.weekday() - 1).cast(pl.Float64)
                else:
                    pl_name = _DATE_METHOD_ALIASES.get(period_name, (None, period_name))[1]
                    if pl_name is None or not hasattr(col_dt, pl_name):
                        raise ValueError(f"Unknown polars .dt accessor for period {period_name!r}")
                    base_expr = getattr(col_dt, pl_name)().cast(pl.Float64)
                angle_expr = (base_expr * two_pi / float(period_value))
                all_exprs.append(angle_expr.sin().cast(pl.Float32).alias(f"{col}_{period_name}_sin"))
                all_exprs.append(angle_expr.cos().cast(pl.Float32).alias(f"{col}_{period_name}_cos"))
        df = df.with_columns(all_exprs)
        if delete_original_cols:
            df = df.drop(cols)

    return df


def run_pysr_fe(
    df: pl.DataFrame,
    nsamples: int = 100_000,
    target_columns_prefix: str = "target_",
    timeout_mins: int = 5,
    fill_nans: bool = True,
):
    """Fit a PySR symbolic regressor on a sampled polars frame.

    DEPRECATED: use :func:`mlframe.feature_engineering.bruteforce.run_pysr_feature_engineering` instead -- that path adds leakage-free OOF support, preset wiring, reserved-name handling, and the global Julia lock. This legacy single-target entry point is retained for back-compat only.

    The frame is split into features (numeric columns NOT prefixed by ``target_columns_prefix``)
    and targets (columns starting with that prefix). Duplicate sanitised names are disambiguated
    with a numeric suffix so PySR sees a unique feature set.

    When exactly one ``target_<...>`` column is detected the body delegates to
    ``bruteforce.run_pysr_feature_engineering`` so callers get the leakage-free OOF + Julia-lock
    plumbing automatically. Multi-target frames (``targets_df.shape[1] > 1``) still take the
    legacy in-place path because the new entry point is single-output by design; the warning
    surfaces this so users migrate explicitly when they need the multi-output behaviour.
    """
    import warnings

    target_cols = [c for c in df.columns if c.startswith(target_columns_prefix)]
    if len(target_cols) == 1:
        warnings.warn(
            "mlframe.feature_engineering.basic.run_pysr_fe is deprecated; delegating to "
            "bruteforce.run_pysr_feature_engineering. Call that function directly to access "
            "preset wiring + random_state + leakage-free OOF knobs.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .bruteforce import run_pysr_feature_engineering

        drop_columns = [c for c in df.columns if c.startswith(target_columns_prefix) and c != target_cols[0]]
        sample_size = nsamples if nsamples and nsamples > 0 else len(df)
        pysr_params_override = {
            "turbo": True,
            "timeout_in_seconds": timeout_mins * 60,
            "maxsize": 10,
            "niterations": 10,
            "binary_operators": ["+", "*"],
            "unary_operators": ["cos", "exp", "log", "sin", "inv(x) = 1/x"],
            "extra_sympy_mappings": {"inv": lambda x: 1 / x},
            "elementwise_loss": "loss(prediction, target) = abs(prediction - target)",
        }
        return run_pysr_feature_engineering(
            df,
            target_col=target_cols[0],
            drop_columns=drop_columns,
            sample_size=sample_size,
            pysr_params_override=pysr_params_override,
            leakage_free=False,
            verbose=0,
        )

    warnings.warn(
        "mlframe.feature_engineering.basic.run_pysr_fe is deprecated and only retains the multi-target "
        "in-place path because bruteforce.run_pysr_feature_engineering is single-output. Migrate to "
        "per-target loops over bruteforce.run_pysr_feature_engineering for leakage-free OOF + Julia-lock support.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pysr import PySRRegressor

    clean_ram()

    model = PySRRegressor(
        turbo=True,
        timeout_in_seconds=timeout_mins * 60,
        maxsize=10,
        niterations=10,
        binary_operators=["+", "*"],
        unary_operators=["cos", "exp", "log", "sin", "inv(x) = 1/x"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
    )

    # Two-pass replace could collapse distinct columns onto the same sanitised name
    # (e.g. "a=b" and "a.b" both become "a_b"). Deduplicate by appending an index suffix on collision.
    rename_map: Dict[str, str] = {}
    used: set = set()
    for col in df.columns:
        candidate = col.replace("=", "_").replace(".", "_")
        base = candidate
        i = 1
        while candidate in used:
            candidate = f"{base}__{i}"
            i += 1
        used.add(candidate)
        rename_map[col] = candidate

    tmp_df = df.head(nsamples) if nsamples and nsamples > 0 else df
    expr = cs.numeric() - cs.starts_with(target_columns_prefix)
    if fill_nans:
        expr = expr.fill_null(0).fill_nan(0)

    features_df = tmp_df.select(expr).rename(rename_map)
    targets_df = tmp_df.select(cs.starts_with(target_columns_prefix))
    model.fit(features_df, targets_df)

    del tmp_df, features_df, targets_df
    clean_ram()

    return model
