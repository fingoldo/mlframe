"""Predict must blend the members training blended, not every model it can load for the target.

Training's quality / catastrophic / diversity gates drop members, but nothing persisted which ones survived: at predict
time the dropped member's .dump still loaded and was averaged back in, so the deployed blend was one whose metrics
were never measured (the 2026-05-21 shape: four members, one MLP at R2=-4.75, reported as a clean 3-member blend).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.core.predict import _select_trained_members


def _probs(v: float) -> np.ndarray:
    return np.full((4, 2), v)


def test_a_dropped_member_is_left_out():
    probs = [_probs(0.1), _probs(0.2), _probs(0.9)]
    names = ["cb", "lgb", "mlp"]
    out, flags, _ = _select_trained_members(probs, names, [False] * 3, {"members": ["cb", "lgb"]})
    assert [p[0, 0] for p in out] == [0.1, 0.2]
    assert flags == [False, False]


def test_members_are_reordered_to_training_order_so_weights_line_up():
    probs = [_probs(0.1), _probs(0.2)]
    out, _, weights = _select_trained_members(probs, ["lgb", "cb"], None, {"members": ["cb", "lgb"], "blend_weights": [0.8, 0.2]})
    assert [p[0, 0] for p in out] == [0.2, 0.1], "the cb array must come first, like its 0.8 weight"
    assert weights == [0.8, 0.2]


def test_a_missing_member_falls_back_to_the_loaded_set_without_weights(caplog):
    probs = [_probs(0.1), _probs(0.2)]
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        out, _, weights = _select_trained_members(probs, ["cb", "lgb"], None, {"members": ["cb", "xgb"], "blend_weights": [0.5, 0.5]})
    assert len(out) == 2
    assert weights is None, "weights fitted for another member set cannot be applied"
    assert "did not load" in caplog.text


def test_legacy_metadata_keeps_every_loaded_member():
    probs = [_probs(0.1), _probs(0.2)]
    out, _, weights = _select_trained_members(probs, ["cb", "lgb"], None, {"rrf_k": 60})
    assert len(out) == 2 and weights is None
