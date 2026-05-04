"""Tests for the per-split target-rate summary in select_target +
_append_split_rate_suffix.

Pre-Session-7 the model_name carried a single ``BT=74%`` summary
computed on the FULL target (train + val + test combined). That
masked split-specific drift in forward-mode runs — operators could
not tell from a chart header whether train/val/test rates diverged.

Session 7 changes:
- ``select_target`` stamps ONLY the train rate as
  ``BTTR=`` / ``MTTR=`` / ``MLTR=`` (TR for "train").
- ``_append_split_rate_suffix`` (in trainer.py) appends the matching
  ``/BTV=`` / ``/BTTS=`` (V for "val", TS for "test") inside
  ``_compute_split_metrics`` so chart titles read e.g.
  ``BTTR=74%/BTV=86%`` (val) and ``BTTR/BTTS=74%/83%`` (test).

The user's specific format request: BTTR=X1%/BTV=X2% (val side),
BTTR=X1%/BTTS=X3% (test side).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.trainer import _append_split_rate_suffix


# -----------------------------------------------------------------------------
# Binary classification — the user's main use case
# -----------------------------------------------------------------------------


def test_binary_val_appends_BTV():
    """VAL side: model_name BTTR=74% + val target → ...BTTR/BTV=74%/80%"""
    val_target = np.array([1, 1, 1, 1, 0])  # 80%
    out = _append_split_rate_suffix(
        "cb_run BTTR=74%", split_name="val", target=val_target,
    )
    assert out == "cb_run BTTR/BTV=74%/80%"


def test_binary_test_appends_BTTS():
    """TEST side: model_name BTTR=74% + test target → ...BTTR/BTTS=74%/83%"""
    # 5/6 = 0.833... rounds to 83
    test_target = np.array([1, 1, 1, 1, 1, 0])
    out = _append_split_rate_suffix(
        "cb_run BTTR=74%", split_name="test", target=test_target,
    )
    assert out == "cb_run BTTR/BTTS=74%/83%"


def test_binary_user_production_pattern():
    """Mirrors the production log: train 74%, val 86%, test 83%.
    Chart title for val → BTTR=74%/BTV=86%. For test → BTTR/BTTS=74%/83%.
    """
    rng = np.random.default_rng(0)
    val_y = (rng.uniform(size=10_000) < 0.86).astype(np.int8)
    test_y = (rng.uniform(size=10_000) < 0.83).astype(np.int8)

    val_out = _append_split_rate_suffix(
        "cl_act_total_hired_above_1 BTTR=74%", split_name="val", target=val_y,
    )
    test_out = _append_split_rate_suffix(
        "cl_act_total_hired_above_1 BTTR=74%", split_name="test", target=test_y,
    )
    # New format: BTTR/BTV=74%/<val_rate>% (spliced inline, not appended).
    assert "BTTR/BTV=74%/" in val_out
    assert val_out.endswith("%")
    val_rate = int(val_out.split("BTTR/BTV=74%/")[1].rstrip("%"))
    assert 85 <= val_rate <= 87
    assert "BTTR/BTTS=74%/" in test_out
    test_rate = int(test_out.split("BTTR/BTTS=74%/")[1].rstrip("%"))
    assert 82 <= test_rate <= 84


def test_binary_train_split_no_append():
    """Train split's metrics call passes split_name='train' — no append
    (BTTR= is already on the model_name from select_target)."""
    target = np.array([0, 1, 1, 0])
    out = _append_split_rate_suffix(
        "cb_run BTTR=74%", split_name="train", target=target,
    )
    assert out == "cb_run BTTR=74%"


# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------


def test_regression_val_appends_MTV():
    val_target = np.array([1.0, 2.0, 3.0])  # mean 2.0
    out = _append_split_rate_suffix(
        "cb_run MTTR=1.5000", split_name="val", target=val_target,
    )
    assert out == "cb_run MTTR/MTV=1.5000/2.0000"


def test_regression_test_appends_MTTS():
    test_target = np.array([10.0, 20.0])  # mean 15.0
    out = _append_split_rate_suffix(
        "cb_run MTTR=12.5000", split_name="test", target=test_target,
    )
    assert out == "cb_run MTTR/MTTS=12.5000/15.0000"


# -----------------------------------------------------------------------------
# Multilabel
# -----------------------------------------------------------------------------


def test_multilabel_val_appends_MLV():
    """VAL multilabel: per-label rate joined with commas, e.g. MLV=50,50,100%"""
    val_target = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
    ])  # per-label means: 0.5, 0.5, 1.0
    out = _append_split_rate_suffix(
        "cb_run MLTR=40,52,31%", split_name="val", target=val_target,
    )
    assert out == "cb_run MLTR/MLV=40,52,31%/50,50,100%"


def test_multilabel_test_appends_MLTS():
    test_target = np.array([
        [1, 1, 0],
        [1, 0, 1],
    ])  # per-label means: 1.0, 0.5, 0.5
    out = _append_split_rate_suffix(
        "cb_run MLTR=40,52,31%", split_name="test", target=test_target,
    )
    assert out == "cb_run MLTR/MLTS=40,52,31%/100,50,50%"


# -----------------------------------------------------------------------------
# Edge / fallback cases
# -----------------------------------------------------------------------------


def test_legacy_BT_tag_passthrough():
    """When model_name carries the legacy BT= tag (no train indices were
    available at select_target time), the appender doesn't try to add a
    split-specific suffix — the legacy callers are non-suite paths."""
    target = np.array([0, 1, 1])
    out = _append_split_rate_suffix(
        "cb_run BT=74%", split_name="val", target=target,
    )
    assert out == "cb_run BT=74%"


def test_no_recognized_tag_passthrough():
    """If model_name doesn't carry any *TR= token, leave it alone."""
    target = np.array([0, 1, 1])
    out = _append_split_rate_suffix(
        "cb_run something_else", split_name="val", target=target,
    )
    assert out == "cb_run something_else"


def test_target_none_returns_unchanged():
    out = _append_split_rate_suffix(
        "cb_run BTTR=74%", split_name="val", target=None,
    )
    assert out == "cb_run BTTR=74%"


def test_empty_target_returns_unchanged():
    out = _append_split_rate_suffix(
        "cb_run BTTR=74%", split_name="val", target=np.array([], dtype=np.int8),
    )
    assert out == "cb_run BTTR=74%"


def test_pandas_series_input():
    val_target = pd.Series([1, 1, 1, 0])  # 75%
    out = _append_split_rate_suffix(
        "cb_run BTTR=70%", split_name="val", target=val_target,
    )
    assert out == "cb_run BTTR/BTV=70%/75%"


def test_polars_series_input():
    pl = pytest.importorskip("polars")
    val_target = pl.Series([1, 1, 1, 0])
    out = _append_split_rate_suffix(
        "cb_run BTTR=70%", split_name="val", target=val_target,
    )
    assert out == "cb_run BTTR/BTV=70%/75%"


def test_invalid_split_name_returns_unchanged():
    out = _append_split_rate_suffix(
        "cb_run BTTR=74%", split_name="oof", target=np.array([0, 1]),
    )
    assert out == "cb_run BTTR=74%"


def test_multilabel_1d_input_returns_unchanged():
    """Multilabel tag with 1-D target is malformed; pass through."""
    target = np.array([0, 1, 1])
    out = _append_split_rate_suffix(
        "cb_run MLTR=40,52,31%", split_name="val", target=target,
    )
    assert out == "cb_run MLTR=40,52,31%"
