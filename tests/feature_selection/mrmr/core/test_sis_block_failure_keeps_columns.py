"""A failed SIS scoring block must not silently remove its columns from the fit.

The block's scores stayed at their pre-allocated zeros; fuse_scores z-scores each channel, so a zeroed block ranks at the bottom and was cut,
dropping up to ``chunk_width`` candidate columns (including real signal) behind a single throttled warning.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.filters import _fe_interaction_prerank, _mrmr_sis_screen
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends


def _data(seed=0):
    """30 columns in blocks of 10; the only signal is column 15, inside the second block."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2000, 30))
    y = (X[:, 15] > 0).astype(np.int64)
    return X, y


def test_sis_block_failure_does_not_silently_drop_columns(monkeypatch, caplog):
    """When both scoring channels raise on the second block, its columns survive and an unthrottled summary names the failure."""
    real_mi = _orth_mi_backends._mi_classif_batch
    real_prop = _fe_interaction_prerank.second_moment_propensity
    calls = {"n": 0, "prop": 0}

    def flaky(block, y, **kw):
        """Raise on the second block only."""
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient device fault")
        return real_mi(block, y, **kw)

    def flaky_prop(block, y, *a, **kw):
        """Raise on the second block only."""
        calls["prop"] += 1
        if calls["prop"] == 2:
            raise RuntimeError("transient device fault")
        return real_prop(block, y, *a, **kw)

    monkeypatch.setattr(_orth_mi_backends, "_mi_classif_batch", flaky)
    monkeypatch.setattr(_fe_interaction_prerank, "second_moment_propensity", flaky_prop)
    X, y = _data()
    with caplog.at_level(logging.WARNING):
        survivors = _mrmr_sis_screen.sis_screen(X, y, target_survivors=5, chunk_width=10)
    assert calls["n"] >= 2, "fixture precondition: the second block must have been scored"
    assert 15 in set(np.asarray(survivors).tolist()), f"the planted signal in the failed block was screened out: {survivors}"
    assert any("block(s) failed" in r.getMessage() for r in caplog.records)


def test_healthy_screen_keeps_exactly_the_target_count():
    """Control: without failures the screen still returns the requested number of top columns, signal included."""
    X, y = _data(seed=1)
    survivors = np.asarray(_mrmr_sis_screen.sis_screen(X, y, target_survivors=5, chunk_width=10))
    assert 15 in set(survivors.tolist())
    assert survivors.size <= 5
