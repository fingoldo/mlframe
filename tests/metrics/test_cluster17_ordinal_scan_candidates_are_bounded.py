"""Cluster 17 regression: the SciPy-free ordinal cutpoint scan evaluates a bounded number of candidates (audit 2026-11-20, finding 17-F17)."""

import numpy as np

from mlframe.metrics.classification import _ordinal_cutpoints as oc


def _data(n, seed=0):
    """Ordinal grades 0..3 and a noisy continuous prediction with almost all-unique values."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    return y, y + rng.normal(0, 0.6, n)


def test_scan_scores_at_most_a_bounded_number_of_candidates(monkeypatch):
    """With thousands of unique predictions the number of _score calls stays within 4 passes x thresholds x the candidate cap."""
    y, p = _data(4000)
    calls = []
    real = oc._score
    monkeypatch.setattr(oc, "_score", lambda *a, **k: calls.append(1) or real(*a, **k))
    thr, best = oc._coordinate_scan(y, p, np.array([0.5, 1.5, 2.5]), 4, "qwk")
    assert len(calls) <= 1 + 4 * 3 * oc._MAX_SCAN_CANDIDATES
    assert best > 0.5 and len(thr) == 3


def test_small_inputs_still_scan_every_midpoint(monkeypatch):
    """With fewer unique predictions than the cap, the candidate list is not thinned: each pass scores every midpoint."""
    y, p = _data(120)
    calls = []
    real = oc._score
    monkeypatch.setattr(oc, "_score", lambda *a, **k: calls.append(1) or real(*a, **k))
    oc._coordinate_scan(y, p, np.array([0.5, 1.5, 2.5]), 4, "qwk")
    n_mids = np.unique(p).shape[0] - 1
    assert len(calls) >= 1 + 3 * n_mids
