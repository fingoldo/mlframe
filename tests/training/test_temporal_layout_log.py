"""Regression sensor for D1 P2 #10.

``make_train_test_split`` MUST emit a one-line INFO with the implied temporal
layout (train range, val range, test range, val->train gap days,
train->prod estimated gap days) whenever timestamps are supplied AND
``val_placement="forward"`` (the default) is in effect.

Time-series users should see at-a-glance how their default split lays out
without having to thread an extra knob.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd


def test_d1_p2_10_temporal_layout_info_fires_on_default_forward_placement(caplog):
    from mlframe.training.splitting import make_train_test_split

    n = 200
    df = pd.DataFrame({"x": np.arange(n)})
    # 200 daily timestamps -- enough resolution for distinct train/val/test ranges
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))

    with caplog.at_level(logging.INFO, logger="mlframe.training.splitting"):
        make_train_test_split(
            df=df, timestamps=ts, test_size=0.2, val_size=0.2,
            random_seed=42,
        )
    msgs = [r.message for r in caplog.records if r.levelno >= logging.INFO]
    layout_msgs = [m for m in msgs if "Temporal layout" in m and "val->train_gap" in m]
    assert layout_msgs, f"Expected 'Temporal layout' INFO line; got: {msgs}"


def test_d1_p2_10_temporal_layout_not_emitted_when_no_timestamps(caplog):
    from mlframe.training.splitting import make_train_test_split

    n = 200
    df = pd.DataFrame({"x": np.arange(n)})

    with caplog.at_level(logging.INFO, logger="mlframe.training.splitting"):
        make_train_test_split(
            df=df, timestamps=None, test_size=0.2, val_size=0.2,
            random_seed=42,
        )
    msgs = [r.message for r in caplog.records]
    assert not any("Temporal layout" in m for m in msgs), (
        f"Did not expect 'Temporal layout' line without timestamps; got: {msgs}"
    )


def test_d1_p2_10_temporal_layout_not_emitted_on_backward_placement(caplog):
    from mlframe.training.splitting import make_train_test_split

    n = 200
    df = pd.DataFrame({"x": np.arange(n)})
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))

    with caplog.at_level(logging.INFO, logger="mlframe.training.splitting"):
        make_train_test_split(
            df=df, timestamps=ts, test_size=0.2, val_size=0.2,
            val_placement="backward", random_seed=42,
        )
    msgs = [r.message for r in caplog.records]
    assert not any("Temporal layout" in m and "val->train_gap" in m for m in msgs), (
        f"Did not expect default-forward INFO under explicit backward; got: {msgs}"
    )
