"""Dummy baseline computation functions extracted from ``dummy_baselines.py``.

Per-group prediction, safe metric wrappers, regression/classification/
quantile/multilabel baseline computation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Forward annotation only -- runtime import lives inside compute_dummy_baselines() to break the circular load with dummy_baselines.py.
    from .dummy_baselines import BaselineReport

logger = logging.getLogger(__name__)


def _per_target_seed(base_seed: int, target_name: str) -> int:
    """Deterministic per-target seed for stochastic baselines (D13).

    ``base_seed + (hash(target_name) & 0xFFFF)`` keeps reproducibility
    across runs (same target -> same seed) while ensuring independence
    across targets in the same suite (different target -> different
    seed). 0xFFFF mask keeps the offset bounded.
    """
    return (base_seed + (hash(target_name) & 0xFFFF)) & 0x7FFFFFFF

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
    """
    if isinstance(X, pd.DataFrame):
        return X
    if hasattr(X, "to_pandas"):
        # Lazy import: utils.py pulls in _nan_processing which has its own
        # transitive cycle with this module's callers; importing at module
        # scope risks circular-load.
        from .utils import get_pandas_view_of_polars_df
        return get_pandas_view_of_polars_df(X)
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
        except Exception:
            continue
        if 2 <= n_unique <= cap:
            candidates.append((n_unique, col))
    if not candidates:
        return None
    # Highest-cardinality among passing candidates.
    candidates.sort(reverse=True)
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

    n_train = train_y.shape[0]
    y_arr = np.asarray(train_y, dtype=np.float64)
    global_mean = float(np.nanmean(y_arr)) if y_arr.size else 0.0

    # Pull JUST the cat column per side. polars select is O(1) metadata, no row copy. The y-vector is materialised once as a polars Series alongside the train cat
    # column so the aggregation runs in a single eager pipeline; intermediate frames here are 2-column views, never the caller's full 100+ GB frame.
    train_pair = pl.DataFrame({cat_col: train_X.get_column(cat_col), "__y__": pl.Series("__y__", y_arr)})
    # Single group_by pass yields both per-group mean AND per-group size; coverage / entity-overlap diagnostics reuse the size column without a second sweep.
    stats_df = train_pair.group_by(cat_col).agg(pl.col("__y__").mean().alias("__mean__"), pl.len().alias("__size__"))
    n_groups = stats_df.height
    train_groups_series = stats_df.get_column(cat_col)

    # Left-join each split's cat column against the per-group stats; nulls (unseen group) -> global_mean.
    def _predict(side_X: Any) -> np.ndarray:
        joined = side_X.select(cat_col).join(stats_df, on=cat_col, how="left")
        return joined.get_column("__mean__").fill_null(global_mean).to_numpy()

    train_pred = _predict(train_X)
    val_pred = _predict(val_X)
    test_pred = _predict(test_X)

    # Coverage = fraction of val/test rows whose key appears in train_groups. is_in() handles numeric/string/cat uniformly; null on the val side is "unseen" unless
    # train has its own null group (semantically correct for both pandas and polars paths).
    val_key = val_X.get_column(cat_col)
    test_key = test_X.get_column(cat_col)
    val_coverage = float(val_key.is_in(train_groups_series).mean() or 0.0) * 100.0
    test_coverage = float(test_key.is_in(train_groups_series).mean() or 0.0) * 100.0

    # Entity-overlap rate: fraction of val rows whose train-group size >= 5. Reuses stats_df from the single group_by above; no extra sweep.
    val_sizes_joined = val_X.select(cat_col).join(stats_df.select(cat_col, "__size__"), on=cat_col, how="left")
    val_high_overlap = float(val_sizes_joined.get_column("__size__").fill_null(0).ge(5).mean() or 0.0)

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

    Handles polars Categorical / Enum / pandas categorical / object via uniform string coercion + ``__NULL__`` sentinel for NaN keys (round-3 A#13).

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
    if train_X_pd is None:
        train_X_pd = pd.DataFrame(train_X)
    if val_X_pd is None:
        val_X_pd = pd.DataFrame(val_X)
    if test_X_pd is None:
        test_X_pd = pd.DataFrame(test_X)

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
# Per-cell metric computation with isolated try/except (D1)
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


def _compute_regression_baselines(
    target_name: str,
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: np.ndarray,
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    timestamps_train: np.ndarray | None,
    timestamps_val: np.ndarray | None,
    timestamps_test: np.ndarray | None,
    cat_features: Sequence[str] | None,
    config: Any,
    target_type: str = "regression",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build {baseline_name: val_pred} + {baseline_name: test_pred} dicts.

    Returns ``(val_preds, test_preds, extras)``.
    """
    # Lazy local imports: helpers live in dummy_baselines.py which imports our compute funcs (circular load).
    from .dummy_baselines import _is_temporally_monotonic, _normalize_timestamps, _resolve_ts_periods

    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}

    # --- Constant baselines (mean / median / quantile) ---
    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    train_mean = float(np.mean(train_y))
    train_median = float(np.median(train_y))

    val_preds["mean"] = np.full(n_val, train_mean)
    test_preds["mean"] = np.full(n_test, train_mean)

    val_preds["median"] = np.full(n_val, train_median)
    test_preds["median"] = np.full(n_test, train_median)

    for q_label, q_alpha in [("quantile_p25", 0.25), ("quantile_p75", 0.75)]:
        c = float(np.quantile(train_y, q_alpha, method="linear"))
        val_preds[q_label] = np.full(n_val, c)
        test_preds[q_label] = np.full(n_test, c)

    # --- per_group_mean (D1) ---
    cat_col = _pick_per_group_categorical(
        train_X, cat_features, len(train_y), config.per_group_max_cardinality_ratio,
    )
    if cat_col is not None:
        try:
            _, val_pg, test_pg, pg_diag = _per_group_predict(
                train_X, val_X, test_X, train_y, cat_col, target_type,
            )
            # Use TS-aware row label when monotonic split present
            ts_active = (
                timestamps_train is not None
                and timestamps_val is not None
                and timestamps_test is not None
                and _is_temporally_monotonic(timestamps_train, timestamps_val, timestamps_test)
            )
            label = "per_group_historical_mean (ts)" if ts_active else "per_group_mean"
            # Annotate row label with high-overlap warning (D1)
            if pg_diag["repeat_entity_rate"] >= config.per_group_high_overlap_threshold:
                label = f"{label} (high_entity_overlap={pg_diag['repeat_entity_rate']:.2f})"
            val_preds[label] = val_pg
            test_preds[label] = test_pg
            extras["per_group"] = {"cat_col": cat_col, **pg_diag}
            # Coverage gate: exclude from strongest-pick if low
            if (
                pg_diag["val_coverage_pct"] < config.per_group_min_val_coverage_pct
                or pg_diag["test_coverage_pct"] < config.per_group_min_val_coverage_pct
            ):
                extras.setdefault("strongest_pick_excluded", []).append(label)
                logger.info(
                    "[dummy-baselines] target='%s' per_group_mean coverage low "
                    "(val=%.1f%%, test=%.1f%%) -- excluded from strongest-pick",
                    target_name, pg_diag["val_coverage_pct"], pg_diag["test_coverage_pct"],
                )
        except Exception as e:
            logger.info(
                "[dummy-baselines] target='%s' per_group_mean failed (%s); skipping",
                target_name, e,
            )
    else:
        logger.debug(
            "[dummy-baselines] target='%s' per_group_mean: no eligible categorical "
            "(cat_features=%s, n_train=%d, max_cardinality_ratio=%.2f)",
            target_name, cat_features, len(train_y), config.per_group_max_cardinality_ratio,
        )

    # --- TS baselines (D17 + round-3 A#2 prediction rules) ---
    if (
        timestamps_train is not None
        and timestamps_val is not None
        and timestamps_test is not None
    ):
        ts_train = _normalize_timestamps(timestamps_train)
        ts_val = _normalize_timestamps(timestamps_val)
        ts_test = _normalize_timestamps(timestamps_test)
        if (
            ts_train is not None
            and ts_val is not None
            and ts_test is not None
            and _is_temporally_monotonic(ts_train, ts_val, ts_test)
        ):
            periods, ts_diag = _resolve_ts_periods(
                train_y, ts_train, config.ts_extra_periods,
            )
            extras["ts_diagnostics"] = ts_diag
            logger.debug(
                "[dummy-baselines] target='%s' ts_periods: step=%s defaults=%s acf_peaks=%s using=%s",
                target_name,
                ts_diag.get("step_label"),
                ts_diag.get("step_periods"),
                ts_diag.get("acf_peaks"),
                ts_diag.get("using"),
            )

            # naive_last (round-3 A#2: suppress when n_val > inferred_period to avoid mean-rebrand)
            min_period = min(periods) if periods else 0
            if n_val > 0 and (min_period == 0 or n_val <= min_period):
                # Single-constant prediction = last train value
                last_val = float(train_y[-1])
                val_preds["naive_last (ts)"] = np.full(n_val, last_val)
                test_preds["naive_last (ts)"] = np.full(n_test, last_val)
            else:
                logger.debug(
                    "[dummy-baselines] target='%s' naive_last: suppressed "
                    "(n_val=%d > inferred_period=%d; would degenerate to constant -- "
                    "use seasonal_naive_pP instead)",
                    target_name, n_val, min_period,
                )

            # naive_lagP / seasonal_naive_pP for each period
            for P in periods:
                if P < 2 or len(train_y) < P:
                    continue
                # seasonal_naive: predict y_train[-P + (k mod P)] for val row k
                val_sn = np.array([train_y[-P + (k % P)] for k in range(n_val)])
                test_sn = np.array([train_y[-P + (k % P)] for k in range(n_test)])
                label = f"seasonal_naive_p{P} (ts)"
                if P in (ts_diag.get("acf_peaks") or []):
                    label = f"seasonal_naive_p{P} (ts, ACF-detected)"
                val_preds[label] = val_sn
                test_preds[label] = test_sn

            # rolling_mean: include only when ACF detected a peak >= W
            acf_peaks = ts_diag.get("acf_peaks") or []
            for W in (7, 30):
                if W < len(train_y) and any(p >= W for p in acf_peaks):
                    c = float(np.mean(train_y[-W:]))
                    val_preds[f"rolling_mean_w{W} (ts)"] = np.full(n_val, c)
                    test_preds[f"rolling_mean_w{W} (ts)"] = np.full(n_test, c)

            # linear_extrap: OLS y ~ ts on train tail
            try:
                tail_n = min(len(train_y), 10_000)
                ts_tail = ts_train[-tail_n:].astype(np.float64)
                y_tail = np.asarray(train_y[-tail_n:], dtype=np.float64)
                # Center timestamps to avoid float overflow on large epoch ints
                ts_offset = ts_tail[0]
                ts_centered = ts_tail - ts_offset
                slope, intercept = np.polyfit(ts_centered, y_tail, 1)
                val_lin = slope * (ts_val.astype(np.float64) - ts_offset) + intercept
                test_lin = slope * (ts_test.astype(np.float64) - ts_offset) + intercept
                val_preds["linear_extrap (ts)"] = val_lin
                test_preds["linear_extrap (ts)"] = test_lin
            except Exception as e:
                logger.debug(
                    "[dummy-baselines] target='%s' linear_extrap failed (%s); skipping",
                    target_name, e,
                )
        else:
            extras["ts_skip_reason"] = (
                "interleaved split -- TS baselines skipped; for TS-naive use val_placement='forward'"
            )
            logger.info(
                "[dummy-baselines] target='%s' timestamps present but split is interleaved "
                "(monotonic check failed) -- TS baselines skipped",
                target_name,
            )

    return val_preds, test_preds, extras


def _compute_classification_baselines(
    target_name: str,
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: np.ndarray,
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    timestamps_train: np.ndarray | None,
    cat_features: Sequence[str] | None,
    config: Any,
    target_type: str,
    n_classes: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build {baseline: probs} dicts for binary / multiclass.

    Returns ``(val_probs, test_probs, extras)`` where probs are
    ``(N, K)`` matrices.
    """
    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}

    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    seed = _per_target_seed(config.random_state, target_name)

    # Compute train priors
    classes = np.arange(n_classes)
    train_y_int = train_y.astype(np.int64)
    bincounts = np.bincount(train_y_int, minlength=n_classes).astype(np.float64)
    train_prior = bincounts / bincounts.sum() if bincounts.sum() > 0 else np.full(n_classes, 1.0 / n_classes)

    # prior baseline: constant per-class prob = train prior
    prior_probs = np.tile(train_prior, (max(n_val, 1), 1)) if n_val > 0 else np.empty((0, n_classes))
    if n_val > 0:
        val_probs["prior"] = prior_probs
        test_probs["prior"] = np.tile(train_prior, (n_test, 1))

    # most_frequent: predict argmax of prior with one-hot probs
    most_freq_class = int(np.argmax(train_prior))
    mf_probs_row = np.zeros(n_classes)
    mf_probs_row[most_freq_class] = 1.0
    val_probs["most_frequent"] = np.tile(mf_probs_row, (n_val, 1))
    test_probs["most_frequent"] = np.tile(mf_probs_row, (n_test, 1))

    # uniform: 1/K per row
    uniform_probs_row = np.full(n_classes, 1.0 / n_classes)
    val_probs["uniform"] = np.tile(uniform_probs_row, (n_val, 1))
    test_probs["uniform"] = np.tile(uniform_probs_row, (n_test, 1))

    # all_zeros / all_ones (binary only)
    if target_type == "binary_classification" and n_classes == 2:
        # all-class-0: probs = [1, 0]
        z_row = np.array([1.0, 0.0])
        val_probs["all_zeros"] = np.tile(z_row, (n_val, 1))
        test_probs["all_zeros"] = np.tile(z_row, (n_test, 1))
        # all-class-1: probs = [0, 1]
        o_row = np.array([0.0, 1.0])
        val_probs["all_ones"] = np.tile(o_row, (n_val, 1))
        test_probs["all_ones"] = np.tile(o_row, (n_test, 1))

    # stratified: n_repeats over different seeds (D-inline / round-3 C#2)
    # Predicted class sampled from prior; probs = one-hot of sampled class.
    n_repeats = config.stratified_n_repeats
    val_strat_runs: list[np.ndarray] = []
    test_strat_runs: list[np.ndarray] = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        if n_val > 0:
            val_classes = rng.choice(classes, size=n_val, p=train_prior)
            val_strat = np.zeros((n_val, n_classes))
            val_strat[np.arange(n_val), val_classes] = 1.0
            val_strat_runs.append(val_strat)
        if n_test > 0:
            test_classes = rng.choice(classes, size=n_test, p=train_prior)
            test_strat = np.zeros((n_test, n_classes))
            test_strat[np.arange(n_test), test_classes] = 1.0
            test_strat_runs.append(test_strat)
    # Mean over repeats -- gives smoothed probs ~ train_prior on average,
    # but with the realized variance preserved for log_loss / AUC scoring.
    if val_strat_runs:
        val_probs["stratified"] = np.mean(val_strat_runs, axis=0)
    if test_strat_runs:
        test_probs["stratified"] = np.mean(test_strat_runs, axis=0)
    extras["stratified_n_repeats"] = n_repeats

    # per_group_prior (binary only for now)
    if target_type == "binary_classification":
        cat_col = _pick_per_group_categorical(
            train_X, cat_features, len(train_y), config.per_group_max_cardinality_ratio,
        )
        if cat_col is not None:
            try:
                _, val_pg, test_pg, pg_diag = _per_group_predict(
                    train_X, val_X, test_X, train_y.astype(np.float64), cat_col, target_type,
                )
                # Convert to (N, 2) probs: [1-p, p]
                val_pg_2d = np.column_stack([1 - val_pg, val_pg])
                test_pg_2d = np.column_stack([1 - test_pg, test_pg])
                label = "per_group_prior"
                if pg_diag["repeat_entity_rate"] >= config.per_group_high_overlap_threshold:
                    label = f"per_group_prior (high_entity_overlap={pg_diag['repeat_entity_rate']:.2f})"
                val_probs[label] = val_pg_2d
                test_probs[label] = test_pg_2d
                extras["per_group"] = {"cat_col": cat_col, **pg_diag}
                if (
                    pg_diag["val_coverage_pct"] < config.per_group_min_val_coverage_pct
                    or pg_diag["test_coverage_pct"] < config.per_group_min_val_coverage_pct
                ):
                    extras.setdefault("strongest_pick_excluded", []).append(label)
            except Exception as e:
                logger.info(
                    "[dummy-baselines] target='%s' per_group_prior failed (%s); skipping",
                    target_name, e,
                )

    return val_probs, test_probs, extras


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
    from .dummy_baselines import (
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
        from .configs import DummyBaselinesConfig
        config = DummyBaselinesConfig()

    # Coerce y to 1D / 2D numpy as appropriate (D8 object-dtype gate).
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

    # D9: skip block if both val and test are uninformative
    if n_val_finite < 2 and n_test_finite < 2:
        logger.warning(
            "[DUMMY_BASELINES] FAILED target='%s' - both val (%d/%d finite) and "
            "test (%d/%d finite) targets have <2 finite values",
            target_name, n_val_finite, n_val, n_test_finite, n_test,
        )
        return _empty_report(target_type, target_name, t0, reason="both-splits-uninformative")

    # D4: multi-output regression. For 2D y in regression / quantile_regression,
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

    # Normalize timestamps once (round-3 A#4 mixed-tz handling).
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
        val_preds, test_preds, extras = _compute_classification_baselines(
            target_name, train_X, val_X, test_X,
            train_y_arr, val_y_arr, test_y_arr,
            ts_train, cat_features, config,
            target_type=target_type, n_classes=n_classes,
        )
        extras["n_classes"] = n_classes
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

    # Strongest-pick (D2): non-degeneracy gate + paired-bootstrap
    strongest, ts_period_used = _pick_strongest(
        target_type, table, val_y_arr, test_y_arr, primary_metric, extras, config,
    )

    # D2 (paired-bootstrap robustness): compute delta vs runner-up + 95% CI +
    # P(strongest beats runner-up). Below `strongest_min_beat_runner_up_prob`
    # the strongest is annotated as TIE and the overlay plot is skipped.
    # Gated on the same n-threshold as bootstrap CI (D16) -- at large n the
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

    # D16: bootstrap CI for strongest baseline when min(n_val, n_test) < 2000.
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

    # 2026-05-10: dummy-baselines overlay plot REMOVED per user feedback.
    # The standard ``report_regression_model_perf`` / ``report_probabilistic_model_perf``
    # already produce per-model scatter + residual + calibration charts
    # with full title-metric headers. Re-rendering a separate
    # baseline-overlay PNG was redundant noise on disk and operators
    # asked to "see my standard charts and reports, not a new chart
    # type". The dummy-baselines TABLE (val/test metric grid + strongest
    # verdict line) remains the actionable artifact.
    plot_path = None

    # 2026-05-11: expose strongest-baseline val/test predictions via
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


def _compute_quantile_baselines(
    target_name: str,
    train_y: np.ndarray,
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    alphas: Sequence[float],
    config: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Per-alpha empirical-quantile baselines for QUANTILE_REGRESSION.

    Emits, per requested alpha:
      - ``quantile_alpha_{a:.3f}``: constant prediction = empirical alpha-th
        percentile of train_y (clamped to [1e-3, 1-1e-3] for boundary alpha
        per round-3 A#9); shape ``(N, K)`` where K=len(alphas).
      - ``median_for_all``: single ``np.median(train_y)`` constant
        broadcast across all alpha (D19: identical to alpha=0.5 row by
        construction; documented in row label).

    Predictions are 2D ``(N, K)``. Pinball loss is computed per alpha
    plus a ``mean_pinball`` aggregate over non-boundary alpha (alpha in
    ``[0.05, 0.95]``; round-3 C#7).
    """
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}
    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    K = len(alphas)
    if K == 0:
        return val_preds, test_preds, extras

    train_median = float(np.median(train_y))
    boundary_log: list[tuple[float, float]] = []  # (orig, clamped)
    n_eff_val: dict[float, int] = {}
    n_eff_test: dict[float, int] = {}

    # Per-alpha: emit one baseline whose prediction is a constant column for
    # that alpha only, broadcast across the K-output shape so the metrics
    # table can compute pinball@alpha uniformly.
    consts_per_alpha: list[float] = []
    for a in alphas:
        clamped_a = float(min(max(a, 1e-3), 1 - 1e-3))
        if clamped_a != a:
            boundary_log.append((float(a), clamped_a))
        c = float(np.quantile(train_y, clamped_a, method="linear"))
        consts_per_alpha.append(c)
        if val_y is not None:
            n_eff_val[a] = int(np.sum(val_y < c))
        if test_y is not None:
            n_eff_test[a] = int(np.sum(test_y < c))

    # Build (N, K) predictions per baseline.
    if K > 0:
        # Per-alpha empirical-quantile baselines: each one is a (N, K)
        # constant matrix where every output uses its own alpha-th percentile.
        for j, a in enumerate(alphas):
            row_const = consts_per_alpha[j]
            # The j-th baseline emits the j-th constant for ALL alphas
            # (interpretation: "use this alpha-th percentile to predict every
            # quantile" -- degenerate but informative as a reference).
            label = f"quantile_alpha_{a:.3f}"
            if a == 0.5:
                label = f"quantile_alpha_{a:.3f} (=median by construction)"
            val_preds[label] = np.full((n_val, K), row_const)
            test_preds[label] = np.full((n_test, K), row_const)

        # median_for_all: single np.median(train_y) across all alpha.
        val_preds["median_for_all"] = np.full((n_val, K), train_median)
        test_preds["median_for_all"] = np.full((n_test, K), train_median)

        # multi_quantile_empirical: predicts the j-th alpha-th percentile in
        # the j-th column -- the "right" multi-quantile constant baseline.
        # This is actually what most quantile-loss models should beat.
        consts_arr = np.asarray(consts_per_alpha, dtype=np.float64)
        val_preds["multi_quantile_empirical"] = np.broadcast_to(
            consts_arr, (n_val, K)
        ).copy()
        test_preds["multi_quantile_empirical"] = np.broadcast_to(
            consts_arr, (n_test, K)
        ).copy()

    if boundary_log:
        extras["quantile_boundary_clamped"] = boundary_log
        for orig, clamped in boundary_log:
            logger.info(
                "[dummy-baselines] target='%s' alpha=%g: clamped to %g for empirical "
                "baseline (degenerate at boundary)",
                target_name, orig, clamped,
            )
    if n_eff_val:
        extras["quantile_n_eff_val"] = n_eff_val
    if n_eff_test:
        extras["quantile_n_eff_test"] = n_eff_test

    return val_preds, test_preds, extras


