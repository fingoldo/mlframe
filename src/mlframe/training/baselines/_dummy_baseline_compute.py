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
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Sequence, TYPE_CHECKING, Tuple, Union

import numpy as np
import pandas as pd

# Sibling-file re-exports (Wave 92). Imports are at top-level so the symbols
# resolve at module load; each sub-file lazy-imports back from this module
# inside its function body, which is safe because that load happens at call
# time (post module-load).
from ._dummy_baseline_regression import _compute_regression_baselines
from ._dummy_baseline_classification import _compute_classification_baselines
from ._dummy_baseline_quantile import _compute_quantile_baselines

if TYPE_CHECKING:
    # Forward annotation only -- runtime import lives inside compute_dummy_baselines() to break the circular load with dummy_baselines.py.
    from .dummy import BaselineReport

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
            from collections import OrderedDict as _OD
            _TO_PANDAS_BASELINE_CACHE = _OD()
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
        "n_groups_train": int(len(train_groups)),
        "global_fallback": global_mean,
    }


# ---------------------------------------------------------------------
# Per-cell metric computation with isolated try/except
# ---------------------------------------------------------------------


def _safe_metric(
    metric_fn: callable, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any
) -> float:
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


def compute_dummy_baselines(
    target_type: str,
    target_name: str,
    *,
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: Any,
    val_y: Any,
    test_y: Any,
    timestamps_train: Any = None,
    timestamps_val: Any = None,
    timestamps_test: Any = None,
    group_ids_train: Any = None,
    group_ids_val: Any = None,
    group_ids_test: Any = None,
    doc_ids_train: Any = None,
    doc_ids_val: Any = None,
    doc_ids_test: Any = None,
    cat_features: Sequence[str] | None = None,
    target_label_encoder: Any = None,
    quantile_alphas: Sequence[float] | None = None,
    config: Any = None,
    plot_file_prefix: str = "",
) -> BaselineReport:
    """Compute dummy baselines for one (target_type, target_name).

    Public entry point. Routes to per-target dispatcher, computes
    per-cell metrics in isolated try/except, picks strongest with
    non-degeneracy + paired-bootstrap robustness gates, optionally
    saves overlay plot for strongest baseline.
    """
    import time as _time
    # Lazy local imports for circular-load helpers (dummy_baselines.py <- this module).
    from .dummy import (
        BaselineReport,
        _bootstrap_ci_for_strongest,
        _coerce_y,
        _compute_ltr_baselines,
        _compute_metrics_table,
        _compute_multi_output_regression,
        _compute_multilabel_baselines,
        _empty_report,
        _is_finite_mask,
        _normalize_timestamps,
        _paired_bootstrap_vs_runner_up,
        _pick_strongest,
    )
    t0 = _time.time()

    if config is None:
        from ..configs import DummyBaselinesConfig
        config = DummyBaselinesConfig()

    # Coerce y to 1D / 2D numpy as appropriate (object-dtype gate).
    train_y_arr = _coerce_y(train_y, target_type, target_name)
    val_y_arr = _coerce_y(val_y, target_type, target_name) if val_y is not None else None
    test_y_arr = _coerce_y(test_y, target_type, target_name) if test_y is not None else None

    if train_y_arr is None:
        return _empty_report(target_type, target_name, t0, reason="object-dtype-target")

    n_train = len(train_y_arr)
    n_val = 0 if val_y_arr is None else len(val_y_arr)
    n_test = 0 if test_y_arr is None else len(test_y_arr)
    n_train_finite = int(_is_finite_mask(train_y_arr).sum()) if train_y_arr.ndim == 1 else n_train
    n_val_finite = int(_is_finite_mask(val_y_arr).sum()) if val_y_arr is not None and val_y_arr.ndim == 1 else n_val
    n_test_finite = int(_is_finite_mask(test_y_arr).sum()) if test_y_arr is not None and test_y_arr.ndim == 1 else n_test

    # Skip block if both val and test are uninformative
    if n_val_finite < 2 and n_test_finite < 2:
        logger.warning(
            "[DUMMY_BASELINES] FAILED target='%s' - both val (%d/%d finite) and "
            "test (%d/%d finite) targets have <2 finite values",
            target_name, n_val_finite, n_val, n_test_finite, n_test,
        )
        return _empty_report(target_type, target_name, t0, reason="both-splits-uninformative")

    # Multi-output regression. For 2D y in regression / quantile_regression,
    # run the dispatcher per output and aggregate per-output strongest +
    # cross-output normalized strongest. Headline emission stays one verdict
    # block per target (not K verdicts).
    if (
        target_type in ("regression", "quantile_regression")
        and train_y_arr.ndim == 2
        and train_y_arr.shape[1] > 1
    ):
        return _compute_multi_output_regression(
            target_type=target_type,
            target_name=target_name,
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y_arr=train_y_arr, val_y_arr=val_y_arr, test_y_arr=test_y_arr,
            timestamps_train=timestamps_train, timestamps_val=timestamps_val,
            timestamps_test=timestamps_test,
            cat_features=cat_features,
            config=config,
            plot_file_prefix=plot_file_prefix,
            t0=t0,
            n_train=n_train, n_val=n_val, n_test=n_test,
            n_train_finite=n_train_finite, n_val_finite=n_val_finite,
            n_test_finite=n_test_finite,
        )

    # Normalize timestamps once (mixed-tz handling).
    ts_train = _normalize_timestamps(timestamps_train)
    ts_val = _normalize_timestamps(timestamps_val)
    ts_test = _normalize_timestamps(timestamps_test)

    # Dispatch by target_type
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}

    if target_type == "quantile_regression" and quantile_alphas is not None:
        # Per-alpha empirical-quantile baselines + pinball-loss metric.
        # Falls back to regression path when quantile_alphas not provided.
        val_preds, test_preds, extras = _compute_quantile_baselines(
            target_name, train_y_arr, val_y_arr, test_y_arr,
            list(quantile_alphas), config,
        )
        extras["quantile_alphas"] = list(quantile_alphas)
    elif target_type in ("regression", "quantile_regression"):
        val_preds, test_preds, extras = _compute_regression_baselines(
            target_name, train_X, val_X, test_X,
            train_y_arr, val_y_arr, test_y_arr,
            ts_train, ts_val, ts_test,
            cat_features, config, target_type=target_type,
        )
    elif target_type in ("binary_classification", "multiclass_classification"):
        # Determine n_classes from train + val + test union
        all_y = np.concatenate([
            train_y_arr,
            val_y_arr if val_y_arr is not None else np.array([], dtype=train_y_arr.dtype),
            test_y_arr if test_y_arr is not None else np.array([], dtype=train_y_arr.dtype),
        ])
        unique_classes = np.unique(all_y[~pd.isna(all_y)] if all_y.dtype.kind in "fc" else all_y)
        n_classes = max(2, len(unique_classes))
        # Label-encode to positions 0..K-1 against the sorted class union. The classification
        # baselines assume positional labels: ``np.bincount(train_y, minlength=K)`` (returns
        # max(label)+1 wide for non-0-indexed labels -> phantom class-0 column -> wrong-width
        # (N, K) prob matrices) and ``log_loss(y, p, labels=np.arange(K))`` in the metrics table
        # (raises when a raw label like 3 is not in {0,1,2} -> every classification metric NaN).
        # Integer multiclass targets are NOT label-encoded upstream (only string/object are), so
        # {1,2,3} / {10,20,30} reach here raw. searchsorted is identity for already-0..K-1 labels,
        # so the common path is bit-identical; only non-contiguous / non-0-based labels are remapped.
        _cls_sorted = unique_classes
        if len(_cls_sorted) and not np.array_equal(_cls_sorted, np.arange(len(_cls_sorted))):
            def _enc(_y):
                if _y is None:
                    return None
                return np.searchsorted(_cls_sorted, np.asarray(_y)).astype(np.int64)
            train_y_arr = _enc(train_y_arr)
            val_y_arr = _enc(val_y_arr)
            test_y_arr = _enc(test_y_arr)
        val_preds, test_preds, extras = _compute_classification_baselines(
            target_name, train_X, val_X, test_X,
            train_y_arr, val_y_arr, test_y_arr,
            ts_train, cat_features, config,
            target_type=target_type, n_classes=n_classes,
        )
        extras["n_classes"] = n_classes
        extras["class_labels"] = list(_cls_sorted)
    elif target_type == "multilabel_classification":
        val_preds, test_preds, extras = _compute_multilabel_baselines(
            target_name, train_y_arr, val_y_arr, test_y_arr, config,
        )
    elif target_type == "learning_to_rank":
        val_preds, test_preds, extras = _compute_ltr_baselines(
            target_name,
            train_y_arr, val_y_arr, test_y_arr,
            group_ids_train, group_ids_val, group_ids_test,
            ts_train, ts_val, ts_test,
            config,
            doc_ids_train=doc_ids_train,
            doc_ids_val=doc_ids_val,
            doc_ids_test=doc_ids_test,
        )
    else:
        return _empty_report(
            target_type, target_name, t0,
            reason=f"unsupported target_type={target_type}",
        )

    # Compute metrics table
    table, primary_metric = _compute_metrics_table(
        target_type, val_preds, test_preds, val_y_arr, test_y_arr,
        group_ids_val=group_ids_val, group_ids_test=group_ids_test,
        extras=extras,
    )

    # Strongest-pick: non-degeneracy gate + paired-bootstrap
    strongest, ts_period_used = _pick_strongest(
        target_type, table, val_y_arr, test_y_arr, primary_metric, extras, config,
    )

    # Paired-bootstrap robustness: compute delta vs runner-up + 95% CI +
    # P(strongest beats runner-up). Below `strongest_min_beat_runner_up_prob`
    # the strongest is annotated as TIE and the overlay plot is skipped.
    # Gated on the same n-threshold as bootstrap CI -- at large n the
    # point-estimate signal-to-noise is high enough that paired bootstrap
    # is just expensive ceremony (~3-4s on n=10^5).
    n_ref_for_paired = min(
        n_val_finite if n_val_finite > 0 else 10_000_000,
        n_test_finite if n_test_finite > 0 else 10_000_000,
    )
    if (
        strongest is not None
        and primary_metric is not None
        and n_ref_for_paired < config.bootstrap_ci_threshold
    ):
        try:
            paired = _paired_bootstrap_vs_runner_up(
                target_type, strongest, primary_metric, table,
                val_preds, test_preds, val_y_arr, test_y_arr,
                n_resamples=config.paired_bootstrap_n_resamples,
                seed=_per_target_seed(config.random_state, target_name) + 1,
            )
            if paired is not None:
                extras["paired_bootstrap"] = paired
                if paired.get("p_strongest_beats") is not None and (
                    paired["p_strongest_beats"] < config.strongest_min_beat_runner_up_prob
                ):
                    extras["tie"] = True
        except Exception as e:
            logger.debug(
                "[dummy-baselines] target='%s' paired-bootstrap failed (%s); skipping",
                target_name, e,
            )

    # Bootstrap CI for strongest baseline when min(n_val, n_test) < 2000.
    # Below that threshold the noise floor on RMSE / log_loss / NDCG is non-
    # trivial (>1%), so a CI grounds the verdict line. Above 2000, point
    # estimate is accurate to <1% and CI is suppressed to keep output compact.
    n_ref_for_ci = min(
        n_val_finite if n_val_finite > 0 else 10_000_000,
        n_test_finite if n_test_finite > 0 else 10_000_000,
    )
    if (
        strongest is not None
        and primary_metric is not None
        and n_ref_for_ci < config.bootstrap_ci_threshold
        and n_ref_for_ci >= 10
    ):
        try:
            ci = _bootstrap_ci_for_strongest(
                target_type, strongest, primary_metric,
                val_preds, test_preds, val_y_arr, test_y_arr,
                n_resamples=config.bootstrap_ci_n_resamples,
                seed=_per_target_seed(config.random_state, target_name),
            )
            if ci is not None:
                extras["bootstrap_ci"] = ci
        except Exception as e:
            logger.debug(
                "[dummy-baselines] target='%s' bootstrap CI failed (%s); skipping",
                target_name, e,
            )

    # Dummy-baselines overlay plot REMOVED.
    # The standard ``report_regression_model_perf`` / ``report_probabilistic_model_perf``
    # already produce per-model scatter + residual + calibration charts
    # with full title-metric headers. Re-rendering a separate
    # baseline-overlay PNG was redundant noise on disk and operators
    # asked to "see my standard charts and reports, not a new chart
    # type". The dummy-baselines TABLE (val/test metric grid + strongest
    # verdict line) remains the actionable artifact.
    plot_path = None

    # Expose strongest-baseline val/test predictions via
    # ``extras`` so a downstream consumer (core.py, between
    # dummy-baselines computation and the per-target model-training
    # loop) can render the "best-baseline-overlay" pre-training chart
    # the user repeatedly asked for. We keep the prediction arrays
    # OUT of ``BaselineReport``'s top-level fields (they'd bloat
    # JSON serialization of metadata.pkl) and store them in extras
    # under explicit keys that the renderer reads by name. Memory
    # cost: 2 x n_split float arrays per target, freed once the
    # renderer consumes them.
    if strongest is not None:
        sv = val_preds.get(strongest)
        st = test_preds.get(strongest)
        if sv is not None:
            extras["strongest_val_preds"] = np.asarray(sv)
        if st is not None:
            extras["strongest_test_preds"] = np.asarray(st)

    elapsed_s = _time.time() - t0
    return BaselineReport(
        target_type=target_type,
        target_name=target_name,
        table=table,
        strongest=strongest,
        primary_metric=primary_metric,
        ts_period_used=ts_period_used,
        plot_path=plot_path,
        elapsed_s=elapsed_s,
        n_train=n_train, n_val=n_val, n_test=n_test,
        n_train_finite=n_train_finite, n_val_finite=n_val_finite, n_test_finite=n_test_finite,
        extras=extras,
    )


# ---------------------------------------------------------------------
# Multilabel + LTR dispatchers + metrics + plot + helpers
# ---------------------------------------------------------------------
# Wave 92 (2026-05-21): _compute_quantile_baselines moved to sibling file
# _dummy_baseline_quantile.py and re-exported from the module-top imports.


