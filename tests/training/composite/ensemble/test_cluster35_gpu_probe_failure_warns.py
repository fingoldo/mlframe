"""Audit 2026-11-20 finding 35-F18: a failed xgboost GPU probe is logged at WARNING, not swallowed at DEBUG.

The probe returned "no GPU" silently, so a CUDA host whose probe raised got no GPU warning at all.
"""

import logging
import sys
import types

from mlframe.training.composite.ensemble import detect_gpu_in_use


def test_failed_xgboost_probe_is_logged_at_warning(monkeypatch, caplog):
    """With build_info raising, the probe reports no GPU and a WARNING names the failure."""
    fake = types.ModuleType("xgboost")

    def _boom():
        """Stand in for a build_info call that fails."""
        raise RuntimeError("driver missing")

    fake.build_info = _boom
    monkeypatch.setitem(sys.modules, "xgboost", fake)
    with caplog.at_level(logging.WARNING):
        out = detect_gpu_in_use(["xgb"])
    assert out == []
    assert any(r.levelno == logging.WARNING and "xgboost GPU probe failed" in r.getMessage() for r in caplog.records)
