"""Regression: ``make_train_test_split`` must fall back from
wholeday-splitting to row-based when:

1. The available unique days are too few to honour the requested
   val/test fractions (would otherwise produce empty val/test splits
   that downstream models reject -- iter-48 300k seed=7 surfaced this
   as ``CatBoostError: Input data must have at least one feature``).
2. The timestamps are numeric (int64 epoch-seconds or float seconds);
   ``pd.to_datetime`` without a unit hint collapses these to a single
   1970-01-01 date, triggering the same empty-val/test failure mode.

Pre-fix path:
- ``wholeday_splitting=True`` + only 1 unique day -> ``n_total =
  len(unique_dates) = 1`` -> ``_calculate_split_sizes(1, 0.1, ...)``
  returned (0, 0) for both val and test -> empty index arrays were
  passed downstream.
- The harness in ``_profile_fuzz_1m.py`` saw the WARN lines
  ``val_size=0.1 requested but val split is empty (...)`` and
  ``test_size=0.1 requested but test split is empty (...)``, then
  ``CatBoostError: Input data must have at least one feature`` from
  CatBoost trying to fit on an empty val Pool.

Post-fix: at function entry the wholeday branch now (a) routes
numeric ts dtypes (i/u/f kind) straight to the row-based path with a
WARN, and (b) gates on ``n_unique_days >= 1 + (val_size>0) +
(test_size>0)``; below that floor, fall back to row-based with a
clear WARN.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.training.splitting import make_train_test_split


def _build_df(n: int = 400):
    """Build df."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x0": rng.standard_normal(n).astype(np.float32),
            "y": rng.standard_normal(n).astype(np.float32),
        }
    )


def test_single_unique_day_falls_back_to_row_based(caplog) -> None:
    """All 200 rows share the SAME calendar date -> wholeday-splitting
    would produce empty val/test (1 day, can't split 3-way). Must
    fall back to row-based with non-empty val/test."""
    n = 200
    df = _build_df(n)
    # All rows on the same calendar date.
    ts = pd.Series(pd.to_datetime(["2026-01-01"] * n))
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        result = make_train_test_split(
            df=df,
            test_size=0.1,
            val_size=0.1,
            timestamps=ts,
            wholeday_splitting=True,
        )
    train_idx, val_idx, test_idx = result[0], result[1], result[2]
    assert len(train_idx) > 0
    assert len(val_idx) > 0, "val split must be non-empty after fallback"
    assert len(test_idx) > 0, "test split must be non-empty after fallback"
    # Fallback WARN must mention the unique-day count + the fallback
    # decision so future debuggers know what happened.
    assert any(
        "only 1 unique day" in rec.message and "row-based" in rec.message for rec in caplog.records
    ), f"expected fallback WARN; got: {[r.message for r in caplog.records]}"


def test_numeric_ts_falls_back_to_row_based(caplog) -> None:
    """int64 epoch-seconds ts + wholeday_splitting=True -> fall back to
    row-based; otherwise pd.to_datetime treats the ints as nanoseconds
    and collapses everything to 1970-01-01."""
    n = 200
    df = _build_df(n)
    ts = pd.Series(np.arange(n, dtype=np.int64) + 1_700_000_000)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        result = make_train_test_split(
            df=df,
            test_size=0.1,
            val_size=0.1,
            timestamps=ts,
            wholeday_splitting=True,
        )
    train_idx, val_idx, test_idx = result[0], result[1], result[2]
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0
    assert any(
        "numeric" in rec.message and "row-based" in rec.message for rec in caplog.records
    ), f"expected numeric-ts WARN; got: {[r.message for r in caplog.records]}"


def test_two_unique_days_with_val_and_test_falls_back(caplog) -> None:
    """2 unique days but val_size>0 AND test_size>0 -> need >=3 days;
    must fall back. Locks the ``1 + val + test`` floor."""
    n = 200
    df = _build_df(n)
    ts = pd.Series(
        pd.to_datetime(
            ["2026-01-01"] * (n // 2) + ["2026-01-02"] * (n - n // 2),
        )
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        result = make_train_test_split(
            df=df,
            test_size=0.1,
            val_size=0.1,
            timestamps=ts,
            wholeday_splitting=True,
        )
    train_idx, val_idx, test_idx = result[0], result[1], result[2]
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0
    assert any("only 2 unique day" in rec.message for rec in caplog.records)


def test_enough_unique_days_no_fallback(caplog) -> None:
    """10 unique days + val_size=0.34 + test_size=0.34 ->
    predicted_n_test=3, predicted_n_val=2; both non-zero, no fallback.
    wholeday_splitting path stays active."""
    df = _build_df(400)
    # 10 unique calendar days (40 rows per day).
    day_strs: list[str] = []
    for d in range(10):
        day_strs.extend([f"2026-01-{1 + d:02d}"] * 40)
    ts = pd.Series(pd.to_datetime(day_strs))
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        result = make_train_test_split(
            df=df,
            test_size=0.34,
            val_size=0.34,
            timestamps=ts,
            wholeday_splitting=True,
        )
    train_idx, val_idx, test_idx = result[0], result[1], result[2]
    # All three splits populated by date assignment.
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0
    # No fallback WARN -- both predicted sizes were non-zero.
    assert not any("falling back to row-based" in rec.message for rec in caplog.records), "expected no fallback at n_days=10; got fallback warnings"


def test_single_day_val_size_zero_no_fallback() -> None:
    """1 unique day + val_size=0 -> floor is 1+0+1=2, still too few.
    But the user opted out of val; the test+train split is still
    achievable. Must fall back since 1<2."""
    n = 200
    df = _build_df(n)
    ts = pd.Series(pd.to_datetime(["2026-01-01"] * n))
    result = make_train_test_split(
        df=df,
        test_size=0.1,
        val_size=0.0,
        timestamps=ts,
        wholeday_splitting=True,
    )
    train_idx, val_idx, test_idx = result[0], result[1], result[2]
    assert len(train_idx) > 0
    assert len(test_idx) > 0
    # val_size=0 -> val_idx empty by design (not by bug).
    assert len(val_idx) == 0


def test_single_day_zero_val_and_test_no_fallback_needed() -> None:
    """1 unique day, val_size=0 AND test_size=0 -> floor is 1, met. No
    fallback; train gets all rows."""
    n = 100
    df = _build_df(n)
    ts = pd.Series(pd.to_datetime(["2026-01-01"] * n))
    result = make_train_test_split(
        df=df,
        test_size=0.0,
        val_size=0.0,
        timestamps=ts,
        wholeday_splitting=True,
    )
    train_idx = result[0]
    assert len(train_idx) == n
