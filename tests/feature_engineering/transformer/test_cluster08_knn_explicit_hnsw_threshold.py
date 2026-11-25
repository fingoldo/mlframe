"""Cluster 08 regression: an explicit prefer_hnsw_at_n of 50_000 is honoured, not treated as the tuned-default sentinel (audit 2026-11-20, finding 08-F20)."""

import numpy as np

from mlframe.feature_engineering.transformer import _knn_helper as kh


def test_explicit_50000_skips_the_tuned_choice(monkeypatch):
    """With an explicit threshold the per-host tuned choice is not consulted (08-F20)."""
    calls = {"n": 0}

    def _spy(**kwargs):
        """Record that the tuned choice was consulted."""
        calls["n"] += 1
        return "exact"

    monkeypatch.setattr(kh._KNN_SPEC, "choose", _spy)
    rng = np.random.default_rng(0)
    X_sub = rng.normal(size=(30, 3))
    X_q = rng.normal(size=(4, 3))
    kh.knn_search(X_sub, X_q, 2, prefer_hnsw_at_n=50_000)
    assert calls["n"] == 0


def test_default_still_consults_the_tuned_choice(monkeypatch):
    """Control: the default (None) defers to the tuned choice (08-F20)."""
    calls = {"n": 0}

    def _spy(**kwargs):
        """Record that the tuned choice was consulted."""
        calls["n"] += 1
        return "exact"

    monkeypatch.setattr(kh._KNN_SPEC, "choose", _spy)
    monkeypatch.setattr(kh, "_check_hnsw_available", lambda: True)
    rng = np.random.default_rng(1)
    kh.knn_search(rng.normal(size=(30, 3)), rng.normal(size=(4, 3)), 2)
    assert calls["n"] == 1
