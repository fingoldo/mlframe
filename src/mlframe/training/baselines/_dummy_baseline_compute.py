"""Dummy baseline computation functions extracted from ``dummy_baselines.py``.

Per-group prediction, safe metric wrappers, regression/classification/
quantile/multilabel baseline computation.

Wave 92 (2026-05-21): the three large per-target dispatchers
(_compute_regression_baselines, _compute_classification_baselines,
_compute_quantile_baselines) live in sibling files
_dummy_baseline_regression.py / _classification.py / _quantile.py.
They are re-exported below so existing
``from ._dummy_baseline_compute import X`` imports continue to work.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, OrderedDict, Sequence

import numpy as np
import pandas as pd

# Sibling-file re-exports. Imports are at top-level so the symbols resolve at
# module load; each sub-file lazy-imports back from this module inside its
# function body, which is safe because that load happens at call time (post
# module-load). Callers (dummy.py, tests/training/baselines/test_dummy_baseline_compute_split.py)
# still do `from ._dummy_baseline_compute import _compute_regression_baselines` etc.
from ._dummy_baseline_regression import _compute_regression_baselines  # noqa: F401 -- re-exported, see above
from ._dummy_baseline_classification import _compute_classification_baselines  # noqa: F401 -- re-exported, see above
from ._dummy_baseline_quantile import _compute_quantile_baselines  # noqa: F401 -- re-exported, see above

logger = logging.getLogger(__name__)


def _per_target_seed(base_seed: int, target_name: str) -> int:
    """Deterministic per-target seed for stochastic baselines (D13).

    Uses ``hashlib.blake2b`` (NOT Python's builtin ``hash``) so the seed
    is bit-stable across processes / runs / PYTHONHASHSEED values. The
    pre-2026-05-20 form used ``hash(target_name)``, but Python salts the
    builtin string hash per process when PYTHONHASHSEED is unset (the
    default), so the docstring's "same target -> same seed" guarantee
    silently broke across runs: stochastic baselines
    (``random_quantile``, ``_pick_per_group_categorical``, etc. in
    dummy_baselines.py) produced different predictions and metric
    values run-to-run, then those values were persisted into
    ``BaselineReport``s consumed by the eval-report pipeline.

    16 bits of offset are kept (0xFFFF mask) for backward compatibility
    with the prior offset range; the result is folded into a non-
    negative int32 so it round-trips through numpy RNGs that reject
    negative seeds.
    """
    import hashlib as _hashlib
    _digest = _hashlib.blake2b(target_name.encode("utf-8"), digest_size=4).digest()
    _stable_int = int.from_bytes(_digest, "big")
    return (base_seed + (_stable_int & 0xFFFF)) & 0x7FFFFFFF

# Audit D P2-8 (2026-05-18): ``_pick_per_group_categorical`` + ``_per_group_predict`` both call
# ``_to_pandas_for_baseline`` on the same train_X. The bridge is zero-copy on Arrow buffers so
# the second call is cheap, but the log line + the get_pandas_view_of_polars_df overhead repeat.
# A weak-id memo keyed by ``id(X)`` (the caller holds a strong ref across the per-target loop)
# threads the view through without changing the function signature. Bounded to 4 entries because
# at most train/val/test/df are live in one per-target iteration; weakrefs are NOT used (polars
# DataFrames don't support weakref because they are pyo3-backed) so we rely on the explicit cap
# + an id() key that the caller's strong reference keeps alive for the duration of the call.
# Maps id(X) -> (cols, shape, pandas_view). The id() can recycle onto a different polars frame after GC, so
# the hit is CO-VALIDATED against the live X's columns + shape before reuse (a recycled id with a different
# frame fails the check and falls through to a fresh bridge call rather than returning a stale wrong view).
_TO_PANDAS_BASELINE_CACHE: "OrderedDict[int, tuple]" = None  # type: ignore[assignment]


def _to_pandas_for_baseline(X: Any) -> pd.DataFrame | None:
    """Bridge a polars / pandas / unknown ``X`` to a pandas frame view for
    per-group baseline computation.

    Routes polars input through ``get_pandas_view_of_polars_df`` -- the
    Arrow-backed split-blocks bridge -- so numeric columns stay as zero-copy
    views instead of being consolidated (~32x faster than the default
    ``.to_pandas()`` on multi-million-row frames). Returns ``None`` for
    types we cannot coerce so callers can short-circuit.

    Centralising the coercion lets ``_per_group_predict`` convert the three
    train/val/test frames exactly once at its entry, instead of paying the
    bridge cost per column inside the inner ``_col_to_groupkey`` loop.

    Audit D P2-8: id-keyed memoization avoids a second log line + bridge call
    when ``_pick_per_group_categorical`` runs back-to-back with
    ``_per_group_predict`` on the same train_X. The cache is bounded to 4
    entries (train/val/test/scratch) and cleared on every fresh call from a
    different frame.
    """
    if isinstance(X, pd.DataFrame):
        return X
    if hasattr(X, "to_pandas"):
        global _TO_PANDAS_BASELINE_CACHE
        # Lazy init to avoid OrderedDict import at module scope (already imported but cheap).
        if _TO_PANDAS_BASELINE_CACHE is None:
            from collections import OrderedDict
            _TO_PANDAS_BASELINE_CACHE = OrderedDict()
        key = id(X)
        try:
            _cols = tuple(X.columns) if hasattr(X, "columns") else None
        except Exception:
            _cols = None
        _shape = getattr(X, "shape", None)
        cached = _TO_PANDAS_BASELINE_CACHE.get(key)
        if cached is not None:
            _c_cols, _c_shape, _c_view = cached
            if _c_cols == _cols and _c_shape == _shape:
                return _c_view
            # id() recycled onto a different frame -> stale entry, drop and rebuild.
            _TO_PANDAS_BASELINE_CACHE.pop(key, None)
        # Lazy import: utils.py pulls in _nan_processing which has its own
        # transitive cycle with this module's callers; importing at module
        # scope risks circular-load.
        from ..utils import get_pandas_view_of_polars_df
        view = get_pandas_view_of_polars_df(X)
        _TO_PANDAS_BASELINE_CACHE[key] = (_cols, _shape, view)
        # Bound cache to 4 entries (train / val / test / scratch).
        while len(_TO_PANDAS_BASELINE_CACHE) > 4:
            _TO_PANDAS_BASELINE_CACHE.popitem(last=False)
        return view
    return None


def _pick_per_group_categorical(
    train_X: Any,
    cat_features: Sequence[str] | None,
    n_train: int,
    max_cardinality_ratio: float,
) -> str | None:
    """Pick the highest-cardinality categorical that PASSES the cap.

    Returns column name or None if no cat passes the gate.
    """
    if not cat_features:
        return None
    # Coerce to pandas-like view for column access. Use the shared bridge so
    # the polars->pandas materialisation goes through the Arrow split-blocks
    # fast path (~32x faster than default .to_pandas()).
    df = _to_pandas_for_baseline(train_X)
    if df is None:
        return None
    candidates = []
    cap = max_cardinality_ratio * n_train
    for col in cat_features:
        if col not in df.columns:
            continue
        try:
            n_unique = df[col].nunique(dropna=False)
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _dummy_baseline_compute.py:154: %s", e)
            continue
        if 2 <= n_unique <= cap:
            candidates.append((n_unique, col))
    if not candidates:
        return None
    # Highest-cardinality among passing candidates. Tiebreaker: column name
    # (alphabetical) so the strongest-pick is deterministic across runs when
    # two cat features share the same cardinality. Without the tiebreaker, the
    # ordering depended on dict-insertion order of cat_features, which a caller
    # could pass in any order.
    candidates.sort(key=lambda nc: (-nc[0], nc[1]))
    return candidates[0][1]


def _is_polars_frame(X: Any) -> bool:
    """Detect ``pl.DataFrame`` without importing polars unless it's already loaded."""
    try:
        import polars as pl
    except ImportError:
        return False
    return isinstance(X, pl.DataFrame)


def _per_group_predict_polars(
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: np.ndarray,
    cat_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Polars-native per-group baseline. Avoids the polars -> pandas bridge entirely.

    Group keys remain in their native dtype (numeric / categorical / enum / utf8). Per-row predictions are recovered via left-join on the group column, so unseen
    groups inherit the global-mean fallback via ``fill_null``. Numerically bit-equal to the pandas path under standard inputs (verified by sibling regression test).
    """
    import polars as pl

    y_arr = np.asarray(train_y, dtype=np.float64)
    global_mean = float(np.nanmean(y_arr)) if y_arr.size else 0.0

    # Pull JUST the cat column per side. polars select is O(1) metadata, no row copy. The y-vector is materialised once as a polars Series alongside the train cat
    # column so the aggregation runs in a single eager pipeline; intermediate frames here are 2-column views, never the caller's full 100+ GB frame.
    train_pair = pl.DataFrame({cat_col: train_X.get_column(cat_col), "__y__": pl.Series("__y__", y_arr)})
    # Single group_by pass yields both per-group mean AND per-group size; coverage / entity-overlap diagnostics reuse the size column without a second sweep.
    stats_df = train_pair.group_by(cat_col).agg(pl.col("__y__").mean().alias("__mean__"), pl.len().alias("__size__"))
    n_groups = stats_df.height

    # iter386: do ONE left-join per side and reuse its (mean, size, seen)
    # columns for prediction + coverage + high-overlap. The legacy form
    # ran 4 joins (train_pred, val_pred, test_pred, val_sizes_joined) plus
    # 2 separate is_in scans on val_key / test_key; the (__mean__ is null)
    # check on the join result is equivalent to is_in(train_groups) since
    # the left-join NULLs exactly the rows whose group missed stats_df.
    # On c0141 1M-row regression baseline-diagnostics the legacy chain ran
    # 20.96s; reusing the joined frame drops 1 redundant val_X join (~1s)
    # + 2 is_in scans (~0.5s each).
    def _predict_with_join(side_X: Any):
        """Left-join ``side_X``'s category column onto ``stats_df`` to attach per-group mean/size; unmatched groups get null stats (filled by the caller)."""
        return side_X.select(cat_col).join(stats_df, on=cat_col, how="left")

    train_joined = _predict_with_join(train_X)
    val_joined = _predict_with_join(val_X)
    test_joined = _predict_with_join(test_X)
    train_pred = train_joined.get_column("__mean__").fill_null(global_mean).to_numpy()
    val_pred = val_joined.get_column("__mean__").fill_null(global_mean).to_numpy()
    test_pred = test_joined.get_column("__mean__").fill_null(global_mean).to_numpy()

    # Coverage = fraction of val/test rows whose key appears in train_groups,
    # equivalent to non-null __mean__ on the left-join result.
    val_mean_col = val_joined.get_column("__mean__")
    test_mean_col = test_joined.get_column("__mean__")
    val_coverage = float(val_mean_col.is_not_null().mean() or 0.0) * 100.0
    test_coverage = float(test_mean_col.is_not_null().mean() or 0.0) * 100.0

    # Entity-overlap rate: fraction of val rows whose train-group size >= 5.
    # Reuse val_joined's __size__ column (already left-joined above).
    val_high_overlap = float(val_joined.get_column("__size__").fill_null(0).ge(5).mean() or 0.0)

    return train_pred, val_pred, test_pred, {
        "val_coverage_pct": val_coverage,
        "test_coverage_pct": test_coverage,
        "repeat_entity_rate": val_high_overlap,
        "n_groups_train": int(n_groups),
        "global_fallback": global_mean,
    }


def _per_group_predict(
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: np.ndarray,
    cat_col: str,
    target_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Compute per-group baseline predictions on val + test (D1).

    Handles polars Categorical / Enum / pandas categorical / object via uniform string coercion + ``__NULL__`` sentinel for NaN keys.

    When ALL three inputs are ``pl.DataFrame`` the function dispatches to a polars-native group_by + join path that skips the pandas bridge entirely; otherwise
    the legacy pandas implementation is used unchanged. CLAUDE.md forbids in-wrapper format auto-conversion, so we only take the polars-native path when all sides
    are already polars (caller-decided format).

    Returns ``(train_pred, val_pred, test_pred, diagnostics)`` where diagnostics contains coverage_pct + repeat_entity_rate.
    """
    if _is_polars_frame(train_X) and _is_polars_frame(val_X) and _is_polars_frame(test_X):
        return _per_group_predict_polars(train_X, val_X, test_X, train_y, cat_col)

    def _col_to_groupkey(X_pd: pd.DataFrame, col: str) -> pd.Series:
        """Coerce a column to a hashable groupby key.

        Fast path: numeric dtypes (int*, float*, bool) pass through unchanged -- pandas groupby handles NaN with ``dropna=False``. astype(str) is reserved for
        object / categorical / datetime dtypes where the original key is not directly hashable / comparable across pl/pd boundary. ~50% wall-time reduction on
        numeric cat cols at n_train >= 1M (measured: 240ms -> 120ms on n=1M, int32).

        ``X_pd`` is already a pandas frame: ``_per_group_predict`` converts train/val/test exactly once at entry via ``_to_pandas_for_baseline`` so this inner
        helper avoids re-bridging on every cat column.
        """
        s = X_pd[col]
        # Numeric fast path
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s
        if pd.api.types.is_bool_dtype(s):
            return s.astype("int8")  # bool -> int for clean groupby
        # Object / categorical / datetime: stringify with NULL sentinel
        return s.astype(str).fillna("__NULL__")

    train_X_pd = _to_pandas_for_baseline(train_X)
    val_X_pd = _to_pandas_for_baseline(val_X)
    test_X_pd = _to_pandas_for_baseline(test_X)
    # Audit D P1-1 (2026-05-18): the previous ``pd.DataFrame(train_X)`` fallback was a TRAP.
    # ``_to_pandas_for_baseline`` only returns ``None`` for objects without ``.to_pandas`` --
    # i.e. NOT pandas, NOT polars. ``pd.DataFrame(<polars_df>)`` does NOT auto-detect polars
    # and instead wraps the polars frame in a single object-dtype column, silently breaking
    # downstream ``_col_to_groupkey``. Raise loudly instead of producing a dead frame.
    if train_X_pd is None or val_X_pd is None or test_X_pd is None:
        missing = []
        if train_X_pd is None:
            missing.append(f"train_X={type(train_X).__name__}")
        if val_X_pd is None:
            missing.append(f"val_X={type(val_X).__name__}")
        if test_X_pd is None:
            missing.append(f"test_X={type(test_X).__name__}")
        raise TypeError(
            "_per_group_predict: inputs must be pandas or polars DataFrames "
            f"(have to_pandas()); got: {', '.join(missing)}. The polars-only fast path "
            "is dispatched separately; an unsupported type here usually means a caller "
            "passed a numpy array or sparse matrix where a DataFrame was expected."
        )

    cat_train = _col_to_groupkey(train_X_pd, cat_col)
    cat_val = _col_to_groupkey(val_X_pd, cat_col)
    cat_test = _col_to_groupkey(test_X_pd, cat_col)

    # Group-mean (regression) or group-positive-rate (binary).
    if target_type == "binary_classification":
        # For DummyClassifier-style pred, output is class label; for our purposes we predict probability = group positive rate.
        y_series = pd.Series(train_y).astype(float)
    else:
        y_series = pd.Series(train_y).astype(float)
    group_means = y_series.groupby(cat_train, dropna=False).mean()
    global_mean = float(y_series.mean())

    train_pred = cat_train.map(group_means).fillna(global_mean).to_numpy()
    val_pred = cat_val.map(group_means).fillna(global_mean).to_numpy()
    test_pred = cat_test.map(group_means).fillna(global_mean).to_numpy()

    # Coverage diagnostics
    train_groups = set(cat_train.unique())
    val_coverage = (cat_val.isin(train_groups)).mean() * 100.0
    test_coverage = (cat_test.isin(train_groups)).mean() * 100.0

    # Entity-overlap rate: fraction of val rows whose group has >=5 train labels
    group_sizes = cat_train.value_counts()
    val_high_overlap = cat_val.map(group_sizes).fillna(0).ge(5).mean()

    return train_pred, val_pred, test_pred, {
        "val_coverage_pct": float(val_coverage),
        "test_coverage_pct": float(test_coverage),
        "repeat_entity_rate": float(val_high_overlap),
        "n_groups_train": len(train_groups),
        "global_fallback": global_mean,
    }


# ---------------------------------------------------------------------
# Per-cell metric computation with isolated try/except
# ---------------------------------------------------------------------


def _safe_metric(metric_fn: Callable, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
    """Compute metric in isolated try/except -> NaN on failure (D1).

    Failure logged ONCE per (metric_fn.__name__, error type) -- not
    silently swallowed.
    """
    try:
        return float(metric_fn(y_true, y_pred, **kwargs))
    except (ValueError, ZeroDivisionError, FloatingPointError, TypeError) as e:
        # Demote to debug to avoid log noise per cell; the WARN happens
        # at strongest-pick / partial-failure level.
        logger.debug(
            "[dummy-baselines] %s failed (%s: %s) -- recording NaN",
            getattr(metric_fn, "__name__", "metric"), type(e).__name__, e,
        )
        return float("nan")


# ---------------------------------------------------------------------
# Per-target dispatchers
# ---------------------------------------------------------------------
# Wave 92 (2026-05-21): _compute_regression_baselines and
# _compute_classification_baselines moved to sibling files
# (_dummy_baseline_regression.py, _dummy_baseline_classification.py)
# and re-exported from the module-top imports above.


# ---------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------


def compute_dummy_baselines(*args, **kwargs):
    """Delegates to the canonical implementation in ``dummy.py``.

    Wave 92 (2026-05-21) split this facade out from the monolith; this module's own docstring claimed the
    split moved the function OUT of here (leaving only a re-export), but a complete second copy of the
    function body remained -- already drifted from the canonical ``dummy.py`` version (missing the
    ``config.overlay_plot`` feature entirely) and importable/callable with zero production call sites.
    Lazy import avoids a circular load: ``dummy.py`` imports FROM this module at its own top level.
    """
    from .dummy import compute_dummy_baselines as _canonical_compute_dummy_baselines

    return _canonical_compute_dummy_baselines(*args, **kwargs)


# ---------------------------------------------------------------------
# Multilabel + LTR dispatchers + metrics + plot + helpers
# ---------------------------------------------------------------------
# Wave 92 (2026-05-21): _compute_quantile_baselines moved to sibling file
# _dummy_baseline_quantile.py and re-exported from the module-top imports.
