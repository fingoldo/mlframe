"""Regression: row-based timestamp split must not leak a timestamp across splits.

Multi-entity panels share one timestamp across many rows (one row per entity per
tick). Positional slicing of the time-sorted index can cut THROUGH such a tied
block, leaving the same timestamp value in BOTH train and val (or val and test) --
a time leak: the model is validated/tested on a timestamp it also trained on.

CLAUDE.md invariant: a TIME split must guarantee train timestamps strictly precede
val/test (no future->past leakage; here, no SAME-tick co-membership either).

Pre-fix these tests fail (shared timestamp across adjacent time-ordered splits);
post-fix boundary-tied rows snap to the later split so precedence is strict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.splitting import make_train_test_split


def _panel_ts(n_ticks: int, per_tick: int) -> pd.Series:
    vals = np.repeat(np.arange(n_ticks), per_tick)
    return pd.Series(pd.to_datetime("2020-01-01") + pd.to_timedelta(vals, "D"))


def test_no_timestamp_shared_across_train_val_test_forward():
    # 7 rows per tick, 33 ticks -> n=231; default cut int(231*0.2)=46 lands
    # mid-way through a tied block pre-fix.
    ts = _panel_ts(33, 7)
    df = pd.DataFrame({"x": np.arange(len(ts))})
    tr, va, te, *_ = make_train_test_split(
        df, test_size=0.2, val_size=0.2, timestamps=ts, wholeday_splitting=False,
    )
    s_tr, s_va, s_te = set(ts[tr]), set(ts[va]), set(ts[te])
    assert not (s_tr & s_va), "timestamp shared between train and val (leak)"
    assert not (s_va & s_te), "timestamp shared between val and test (leak)"
    assert not (s_tr & s_te), "timestamp shared between train and test (leak)"
    # Strict precedence.
    assert ts[tr].max() < ts[va].min()
    assert ts[va].max() < ts[te].min()
    # No row lost or duplicated.
    assert len(tr) + len(va) + len(te) == len(ts)
    assert len(set(tr) | set(va) | set(te)) == len(ts)


def test_no_timestamp_shared_backward_placement():
    ts = _panel_ts(33, 7)
    df = pd.DataFrame({"x": np.arange(len(ts))})
    tr, va, te, *_ = make_train_test_split(
        df, test_size=0.2, val_size=0.2, timestamps=ts,
        wholeday_splitting=False, val_placement="backward",
    )
    s_tr, s_va, s_te = set(ts[tr]), set(ts[va]), set(ts[te])
    # Backward layout: val(oldest) < train < test, all disjoint in time.
    assert not (s_va & s_tr)
    assert not (s_tr & s_te)
    assert not (s_va & s_te)
    assert ts[va].max() < ts[tr].min()
    assert ts[tr].max() < ts[te].min()
    assert len(set(tr) | set(va) | set(te)) == len(ts)


def test_distinct_timestamps_unaffected():
    # Sanity: when every row has a distinct timestamp, the de-leak is a no-op
    # and sizes match the plain sequential split.
    n = 200
    ts = pd.Series(pd.date_range("2020-01-01", periods=n, freq="h"))
    df = pd.DataFrame({"x": np.arange(n)})
    tr, va, te, *_ = make_train_test_split(
        df, test_size=0.2, val_size=0.2, timestamps=ts, wholeday_splitting=False,
    )
    assert len(te) == int(n * 0.2)
    assert ts[tr].max() < ts[va].min() < ts[va].max() < ts[te].min()
