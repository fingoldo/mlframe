"""Audit 2026-11-20 finding 23-F2: an explicit leak tolerance decides the verdict, not a fixed default."""

import numpy as np

from mlframe.evaluation.expanding_window_leakage import _leak_verdict


def test_explicit_tolerance_is_the_one_used_and_decides_detection():
    """inflation 0.3 is below an explicit tolerance of 0.5 (no leak) and above 0.1 (leak); the reported tolerance is the explicit one."""
    leaky = np.array([0.8, 0.8, 0.8])
    honest = np.array([0.5, 0.5, 0.5])
    tol, _p, detected_loose = _leak_verdict(leaky, honest, honest_mean=0.5, inflation=0.3, leak_tolerance=0.5)
    assert tol == 0.5 and detected_loose is False
    _tol, _p, detected_tight = _leak_verdict(leaky, honest, honest_mean=0.5, inflation=0.3, leak_tolerance=0.1)
    assert detected_tight is True
