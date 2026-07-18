"""Unit + biz_value coverage for ``mlframe.training.pipeline._event_proximity_decay_composite_fe``.

The underlying trick (``event_proximity_decay_features``) already has its own biz_value test at the
function level. This file covers the suite-wiring layer: the caller-supplied literal event-dates
config contract, dates sourced from ``timestamps`` (NOT a train/val/test column -- the suite
decomposes datetime columns into numeric day/month/weekday parts in an earlier phase, before this
step's insertion point would see a raw datetime column), schema alignment, no-op gates, and
predict-time replay (no fit-time state -- the SAME persisted event-date list is reapplied) -- plus
one biz_value test proving the wired module recovers an event-driven spike a raw date feature can't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._event_proximity_decay_composite_fe import (
    apply_event_proximity_decay_composite_fe,
    replay_event_proximity_decay_composite_fe,
)


def _date_frame_and_timestamps(n=200, seed=0):
    """Date frame and timestamps."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({"dummy": np.arange(n)})
    return df, np.asarray(dates)


def test_apply_event_proximity_decay_composite_fe_noop_when_unset():
    """Apply event proximity decay composite fe noop when unset."""
    df, ts = _date_frame_and_timestamps()
    cfg = PreprocessingExtensionsConfig()
    train, _val, _test = apply_event_proximity_decay_composite_fe(
        df.iloc[:150],
        df.iloc[150:],
        None,
        cfg,
        ts,
        np.arange(150),
        np.arange(150, 200),
        None,
        verbose=0,
    )
    assert list(train.columns) == list(df.columns)


def test_apply_event_proximity_decay_composite_fe_noop_without_timestamps():
    """Apply event proximity decay composite fe noop without timestamps."""
    df, _ts = _date_frame_and_timestamps()
    cfg = PreprocessingExtensionsConfig(event_proximity_decay_event_dates=["2024-01-15"])
    train, _, _ = apply_event_proximity_decay_composite_fe(df, None, None, cfg, None, np.arange(len(df)), None, None, verbose=0)
    assert list(train.columns) == list(df.columns)


def test_apply_event_proximity_decay_composite_fe_schema_aligned_across_splits():
    """Apply event proximity decay composite fe schema aligned across splits."""
    df, ts = _date_frame_and_timestamps()
    train_idx, val_idx, test_idx = np.arange(0, 150), np.arange(150, 175), np.arange(175, 200)
    cfg = PreprocessingExtensionsConfig(event_proximity_decay_event_dates=["2024-01-15", "2024-03-01"], event_proximity_decay_cap=10)
    metadata: dict = {}
    train, val, test = apply_event_proximity_decay_composite_fe(
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
        cfg,
        ts,
        train_idx,
        val_idx,
        test_idx,
        metadata=metadata,
        verbose=0,
    )
    assert set(train.columns) == set(val.columns) == set(test.columns)
    assert "event_proximity_total_force" in train.columns
    assert metadata["event_proximity_decay_event_dates"] == ["2024-01-15", "2024-03-01"]


def test_apply_event_proximity_decay_composite_fe_polars_roundtrip():
    """Apply event proximity decay composite fe polars roundtrip."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pl.DataFrame({"dummy": np.arange(n)})
    cfg = PreprocessingExtensionsConfig(event_proximity_decay_event_dates=["2024-02-01"], event_proximity_decay_cap=15)
    train, _, _ = apply_event_proximity_decay_composite_fe(df, None, None, cfg, np.asarray(dates), np.arange(n), None, None, metadata={}, verbose=0)
    assert isinstance(train, pl.DataFrame)
    assert "event_proximity_total_force" in train.columns


def test_replay_event_proximity_decay_composite_fe_matches_fit_time_columns():
    """Replay event proximity decay composite fe matches fit time columns."""
    df, ts = _date_frame_and_timestamps()
    cfg = PreprocessingExtensionsConfig(event_proximity_decay_event_dates=["2024-01-15"])
    metadata: dict = {}
    train, _, _ = apply_event_proximity_decay_composite_fe(df, None, None, cfg, ts, np.arange(len(df)), None, None, metadata=metadata, verbose=0)

    fresh_idx = np.arange(0, 20)
    fresh = df.iloc[fresh_idx][["dummy"]].reset_index(drop=True)
    replayed = replay_event_proximity_decay_composite_fe(fresh, metadata, ts[fresh_idx], verbose=0)
    assert set(replayed.columns) == set(train.columns)


def test_replay_event_proximity_decay_composite_fe_noop_without_persisted_metadata():
    """Replay event proximity decay composite fe noop without persisted metadata."""
    df, ts = _date_frame_and_timestamps(n=20)
    out = replay_event_proximity_decay_composite_fe(df, {}, ts, verbose=0)
    assert list(out.columns) == list(df.columns)


def test_biz_val_event_proximity_decay_composite_wiring_detects_event_spike():
    """The label is a demand SPIKE that occurs only within a short window around known event dates
    (a holiday calendar); a raw ordinal day-index feature carries no notion of which days are near a
    holiday (holidays are irregularly spaced). The wired total_force feature (built from the
    caller-supplied event-date list, sourced from timestamps) directly encodes proximity to the
    nearest known event."""
    n = 365
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    event_dates = ["2024-02-14", "2024-07-04", "2024-11-28", "2024-12-25"]
    day_idx = np.arange(n)

    event_days = pd.to_datetime(event_dates)
    days_to_nearest_event = np.array(
        [min(abs((d - pd.Timestamp("2024-01-01")).days - e) for e in [(ed - pd.Timestamp("2024-01-01")).days for ed in event_days]) for d in dates]
    )
    y = (days_to_nearest_event <= 3).astype(int)

    df = pd.DataFrame({"dummy": day_idx})
    cfg = PreprocessingExtensionsConfig(event_proximity_decay_event_dates=event_dates, event_proximity_decay_cap=10)
    out_df, _, _ = apply_event_proximity_decay_composite_fe(df, None, None, cfg, np.asarray(dates), np.arange(n), None, None, verbose=0)

    def _auc(X):
        """Auc."""
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        return roc_auc_score(y, clf.predict_proba(X)[:, 1])

    auc_raw = _auc(day_idx.reshape(-1, 1).astype(float))
    auc_wired = _auc(out_df[["event_proximity_total_force"]].to_numpy())

    assert auc_wired > auc_raw + 0.3, (
        f"wired event-proximity total_force should detect the event-driven spike far better than a "
        f"raw day-index feature, got auc_wired={auc_wired:.3f} vs auc_raw={auc_raw:.3f}"
    )
