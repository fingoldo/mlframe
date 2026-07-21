"""Regression-side dummy baseline computation.

Wave 92 (2026-05-21): split out from `_dummy_baseline_compute.py` to keep
that file below the 1k-line threshold. Behaviour preserved bit-for-bit;
the function is re-exported from `_dummy_baseline_compute` so existing
`from ._dummy_baseline_compute import _compute_regression_baselines`
imports continue to work.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _extract_col_1d(df: Any, col: str) -> np.ndarray:
    """Return ``df[col]`` as a flat ndarray, polars/pandas/array agnostic. Polars uses ``get_column`` (no ``.select().reshape`` round-trip)."""
    if df is None:
        return np.array([])
    if hasattr(df, "columns") and not hasattr(df, "schema"):
        return np.asarray(df[col].to_numpy())
    if hasattr(df, "get_column"):
        return np.asarray(df.get_column(col).to_numpy())
    if hasattr(df, "select"):
        return np.asarray(df.select(col).to_numpy().reshape(-1))
    return np.asarray(df[col])


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
    # Lazy local imports break the circular load with dummy_baselines.py and
    # the helpers that live in _dummy_baseline_compute.
    from .dummy import _is_temporally_monotonic, _normalize_timestamps, _resolve_ts_periods
    from ._dummy_baseline_compute import _pick_per_group_categorical, _per_group_predict

    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}

    # --- Constant baselines (mean / median / quantile) ---
    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    # nan* variants: a single NaN in train_y (a realistic partially-labeled-data occurrence) previously
    # made plain np.mean/np.median/np.quantile return NaN, wiping out every constant-prediction baseline
    # and collapsing the whole verdict to strongest=None even when val/test are fully finite.
    train_mean = float(np.nanmean(train_y))
    train_median = float(np.nanmedian(train_y))

    val_preds["mean"] = np.full(n_val, train_mean)
    test_preds["mean"] = np.full(n_test, train_mean)

    val_preds["median"] = np.full(n_val, train_median)
    test_preds["median"] = np.full(n_test, train_median)

    for q_label, q_alpha in [("quantile_p25", 0.25), ("quantile_p75", 0.75)]:
        c = float(np.nanquantile(train_y, q_alpha, method="linear"))
        val_preds[q_label] = np.full(n_val, c)
        test_preds[q_label] = np.full(n_test, c)

    # Lag-predict baseline. For strongly auto-regressive
    # targets (lag1_corr ~ 0.999 within groups),
    # the dumbest possible prediction -- ``y_hat = lag_target_value``
    # in the same row -- can dramatically outperform mean / median
    # baselines AND sometimes the user's trained models. Observed
    # in prod: BaselineDiagnostics measured init_score(lag1)
    # RMSE=8.06 vs Ridge raw RMSE=11.63 -- a 31% improvement available
    # for free, but the framework never reported it as a baseline.
    #
    # Detect via column-name heuristic so the user doesn't have to
    # wire BaselineDiagnostics output into dummy-baselines: look for
    # ``{target_name}_prev``, ``{target_name}_lag_1``, ``{target_name}_lag1``.
    # If found, emit a ``lag_predict`` baseline that returns the column
    # value per row.
    for _lag_suffix in ("_prev", "_lag_1", "_lag1", "_lag"):
        _lag_col = f"{target_name}{_lag_suffix}"
        _col_in_train = (hasattr(train_X, "columns") and _lag_col in train_X.columns) or (hasattr(train_X, "schema") and _lag_col in train_X.schema)
        if not _col_in_train:
            continue
        try:
            _val_lag = _extract_col_1d(val_X, _lag_col) if n_val else np.array([])
            _test_lag = _extract_col_1d(test_X, _lag_col) if n_test else np.array([])
            val_preds["lag_predict"] = np.asarray(_val_lag, dtype=np.float64)
            test_preds["lag_predict"] = np.asarray(_test_lag, dtype=np.float64)
            extras["lag_predict"] = {"feature_used": _lag_col}
            logger.info(
                "[dummy-baselines] target='%s' lag_predict baseline added "
                "(feature=%s) -- y_hat = lag_target_value per row. On "
                "strong-AR targets this often beats mean / Ridge baselines.",
                target_name, _lag_col,
            )
            break
        except Exception as _lag_err:
            logger.debug(
                "[dummy-baselines] target='%s' lag_predict(%s) probe "
                "failed (%s); skipping",
                target_name, _lag_col, _lag_err,
            )

    # --- per_group_mean ---
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
            # Annotate row label with high-overlap warning
            if pg_diag["repeat_entity_rate"] >= config.per_group_high_overlap_threshold:
                label = f"{label} (high_entity_overlap={pg_diag['repeat_entity_rate']:.2f})"
            val_preds[label] = val_pg
            test_preds[label] = test_pg
            extras["per_group"] = {"cat_col": cat_col, **pg_diag}
            # Coverage gate: exclude from strongest-pick if low
            if pg_diag["val_coverage_pct"] < config.per_group_min_val_coverage_pct or pg_diag["test_coverage_pct"] < config.per_group_min_val_coverage_pct:
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

    # --- TS baselines (prediction rules) ---
    if timestamps_train is not None and timestamps_val is not None and timestamps_test is not None:
        ts_train = _normalize_timestamps(timestamps_train)
        ts_val = _normalize_timestamps(timestamps_val)
        ts_test = _normalize_timestamps(timestamps_test)
        if ts_train is not None and ts_val is not None and ts_test is not None and _is_temporally_monotonic(ts_train, ts_val, ts_test):
            periods, ts_diag = _resolve_ts_periods(
                train_y, ts_train, config.ts_extra_periods, random_state=getattr(config, "random_state", 42),
            )
            extras["ts_diagnostics"] = ts_diag
            logger.debug(
                "[dummy-baselines] target='%s' ts_periods: step=%s defaults=%s acf_peaks=%s using=%s",
                target_name,
                ts_diag.get("step_label"),
                ts_diag.get("step_periods"),
                ts_diag.get("acf_periods"),
                ts_diag.get("using"),
            )

            # naive_last (suppress when n_val > inferred_period to avoid mean-rebrand)
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
                if P in (ts_diag.get("acf_periods") or []):
                    label = f"seasonal_naive_p{P} (ts, ACF-detected)"
                val_preds[label] = val_sn
                test_preds[label] = test_sn

            # rolling_mean: include only when ACF detected a peak >= W
            # ts_diag's real key is "acf_periods" (matches _dummy_timeseries.py's own internal name); the
            # stale "acf_peaks" read here always missed, so rolling_mean_w{7,30} (ts) never fired and
            # seasonal_naive_pP never got its ", ACF-detected" suffix.
            acf_peaks = ts_diag.get("acf_periods") or []
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
            extras["ts_skip_reason"] = "interleaved split -- TS baselines skipped; for TS-naive use val_placement='forward'"
            logger.info(
                "[dummy-baselines] target='%s' timestamps present but split is interleaved " "(monotonic check failed) -- TS baselines skipped",
                target_name,
            )

    return val_preds, test_preds, extras
