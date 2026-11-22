"""Cluster 22 regression: an absent calibration library skips only its own candidates (audit 2026-11-20, finding 22-F3)."""

import sys

import numpy as np

from mlframe.calibration import post


def test_missing_pycalib_does_not_abort_the_zoo(monkeypatch):
    """With pycalib absent, get_postcalibrators still returns the other candidates and none from pycalib (22-F3)."""
    monkeypatch.setitem(sys.modules, "pycalib", None)  # import raises ImportError
    monkeypatch.setitem(sys.modules, "pycalib.models", None)
    calib_target = np.array([0, 1] * 50)
    calibrators = post.get_postcalibrators(calib_target, num_bins=10)
    libs = {c.lib for c in calibrators} if hasattr(calibrators[0], "lib") else set()
    assert calibrators, "the calibrator zoo came back empty"
    assert "pycalib" not in libs
    assert any(getattr(c, "lib", None) == "sklearn" for c in calibrators)
