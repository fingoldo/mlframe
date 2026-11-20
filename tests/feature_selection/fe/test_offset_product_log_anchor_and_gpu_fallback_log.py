"""Offset-product: the frozen log anchor ignores infinities, and a failed device scan is logged when the CPU kernel takes over."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import _offset_product_fe as opf


def test_log_anchor_ignores_infinities_in_the_fit_column():
    """A -inf entry made the anchor inf, so the whole log column turned non-finite and was replaced by a constant."""
    assert opf._log_anchor(np.array([-np.inf, -1.0, 2.0, np.inf])) == pytest.approx(1e-5 + 1.0)
    assert opf._log_anchor(np.array([np.inf, 0.5, 2.0])) == 0.0


def test_scan_all_logs_a_warning_when_the_device_scan_fails(monkeypatch, caplog):
    """The device scan failure used to fall back to the CPU kernel with a debug line only."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch
    from mlframe.feature_selection.filters import _offset_product_gpu as gpu
    from mlframe.utils.log_throttle import reset_throttle_counts

    def _boom(*_a, **_k):
        """Device scan stand-in that fails."""
        raise RuntimeError("device fault")

    monkeypatch.setattr(dispatch, "lookup_offset_scan_backend", lambda *a, **k: "gpu")
    monkeypatch.setattr(gpu, "scan_offset_products_device", _boom)
    reset_throttle_counts("offset_product.gpu_scan_failed")
    rng = np.random.default_rng(0)
    n = 200
    xs = [rng.standard_normal(n), rng.standard_normal(n)]
    funcs = opf._unary_funcs("minimal")
    rows = np.arange(n)
    tasks = np.array([[0, 1, 0, 1]], dtype=np.int64)
    yr = rng.random(n)
    codes = (rng.random(n) > 0.5).astype(np.int64)
    with caplog.at_level(logging.WARNING, logger=opf.logger.name):
        out_mi, clips = opf._scan_all(xs, funcs, ("identity", "abs"), rows, tasks, yr, codes, 2, [0.0, 0.0])
    assert out_mi.shape[0] == 1 and clips.shape[0] == 2
    assert any("offset-product GPU scan failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
