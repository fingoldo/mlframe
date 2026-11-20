"""Regression tests for the MI-greedy constructor: array-like ``cols`` and the logged gate fallbacks."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _mi_greedy_device as dev
from mlframe.feature_selection.filters import _mi_greedy_fe as fe
from mlframe.feature_selection.filters import _mi_prange_kernel as pk
from mlframe.utils.log_throttle import reset_throttle_counts


def _frame():
    """A small frame with two signal columns."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n), "c": rng.standard_normal(n)})
    y = ((X["a"] * X["b"]) > 0).astype(np.int64).to_numpy()
    return X, y


def test_greedy_mi_fe_construct_accepts_ndarray_cols():
    """``cols`` given as an ndarray / Index must not hit the ambiguous truth value of ``cols or X.columns``."""
    X, y = _frame()
    out, _scores = fe.greedy_mi_fe_construct(X, y, cols=np.array(["a", "b"]), top_k=2, include_trig_on_bounded=False)
    assert "c" in out.columns
    out2, _ = fe.greedy_mi_fe_construct(X, y, cols=X.columns[:2], top_k=2, include_trig_on_bounded=False)
    assert out2.shape[0] == len(X)


def test_materialise_failure_is_logged_at_warning(monkeypatch, caplog):
    """A transform that raises is skipped with a throttled WARNING."""
    X, _y = _frame()

    def boom(_x):
        """Raise like a broken transform."""
        raise ValueError("bad transform")

    monkeypatch.setitem(fe.UNARY_TRANSFORMS, "abs", boom)
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=fe.logger.name):
        assert fe._materialise_candidate(X, ("a",), "abs") is None
        for _ in range(20):
            fe._materialise_candidate(X, ("a",), "abs")
    msgs = [r for r in caplog.records if "materialising" in r.getMessage() and r.levelno == logging.WARNING]
    assert 1 <= len(msgs) <= 5


def test_reject_sink_failure_is_logged(caplog):
    """A raising ``reject_sink`` no longer vanishes into a debug line."""
    X, y = _frame()
    calls = []

    def sink(**kw):
        """Fail on every call."""
        calls.append(kw)
        raise RuntimeError("sink broke")

    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=fe.logger.name):
        fe._greedy_score_and_select(
            X[["a"]], pd.DataFrame({"abs(a)": np.abs(X["a"])}), [(("a",), "abs")], y,
            top_k=1, min_uplift=0.0, min_abs_mi_frac=10.0, nbins=10, reject_sink=sink,
        )
    assert calls
    assert any("reject_sink raised" in r.getMessage() for r in caplog.records)


class _FailingSpec:
    """A kernel-tuning spec whose choose() fails."""

    def choose(self, **_k):
        """Raise like a broken tuning cache."""
        raise RuntimeError("ktc broke")


def test_device_choice_failure_is_logged(monkeypatch, caplog):
    """A failing kernel-tuning ``choose`` falls back to the row-count rule and logs it."""
    monkeypatch.setattr(dev, "_SPEC", _FailingSpec())
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: False)
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=dev.logger.name):
        assert dev.use_device(10) is False
    assert any("choose() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_prange_dispatch_choose_failure_is_throttled(monkeypatch, caplog):
    """The permutation dispatcher runs per feature; its tuner failure must go through the throttle, not one warning per call."""
    monkeypatch.setattr(pk, "_MI_PRANGE_PARALLELISM_SPEC", _FailingSpec())
    rng = np.random.default_rng(0)
    cx = rng.integers(0, 4, 50).astype(np.int32)
    cy = rng.integers(0, 2, 50).astype(np.int32)
    fx = np.bincount(cx, minlength=4).astype(np.int32)
    fy = np.bincount(cy, minlength=2).astype(np.int32)
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=pk.logger.name):
        for _ in range(12):
            pk._mi_prange_dispatch(cx, fx, cy, fy, 3, 0.5, np.uint64(7))
    msgs = [r for r in caplog.records if "choose() failed" in r.getMessage()]
    assert 1 <= len(msgs) <= 5
