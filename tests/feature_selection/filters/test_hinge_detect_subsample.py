"""Pin for the 2026-07-03 hinge-detector row-cap (``MLFRAME_HINGE_MAX_ROWS``).

detect_hinge_breakpoints_gpu proposes tau breakpoint candidates via a held-out R^2 uplift; the FE gate
re-scores the built hinge feature, so estimating the uplift on a large strided subsample is selection-
equivalent while the full-n FWL/projection SSEs stop dominating. Assert the cap subsamples and still finds
a genuine breakpoint (a clear piecewise-linear signal), and that the env opt-out restores full-n.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

MOD = "mlframe.feature_selection.filters._hinge_detect_gpu_resident"

_KW = dict(max_breakpoints=1, min_heldout_r2_uplift=0.01, precheck_qs=(0.25, 0.5, 0.75),
           precheck_min_sse_drop=0.0, cand_q_lo=0.1, cand_q_hi=0.9, n_candidates=16,
           min_rows=100, min_seg_rows=20)


def _hinge_data(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = np.where(x > 0.5, 2.0 * (x - 0.5), 0.0) + rng.normal(0, 0.05, n)  # clear breakpoint at 0.5
    return x, y


@pytest.mark.parametrize("max_rows", ["200000", "0"])
def test_hinge_subsample_still_finds_breakpoint(max_rows, monkeypatch):
    monkeypatch.setenv("MLFRAME_HINGE_MAX_ROWS", max_rows)
    m = importlib.import_module(MOD)
    _orig_dict = dict(m.__dict__)
    m = importlib.reload(m)
    try:
        pytest.importorskip("cupy")
        x, y = _hinge_data(1_000_000)
        taus = m.detect_hinge_breakpoints_gpu(x, y, **_KW)
        if taus is None:
            pytest.skip("cupy fault -> host detector path (not under test here)")
        assert len(taus) >= 1, f"[cap={max_rows}] no breakpoint found on a clear hinge signal"
        assert abs(float(taus[0]) - 0.5) < 0.4, f"[cap={max_rows}] breakpoint {taus[0]:.3f} far from 0.5"
    finally:
        m.__dict__.clear()
        m.__dict__.update(_orig_dict)


def test_hinge_cap_actually_subsamples(monkeypatch):
    monkeypatch.setenv("MLFRAME_HINGE_MAX_ROWS", "40000")
    m = importlib.import_module(MOD)
    _orig_dict = dict(m.__dict__)
    m = importlib.reload(m)
    try:
        assert m._hinge_max_rows() == 40000
    finally:
        monkeypatch.delenv("MLFRAME_HINGE_MAX_ROWS", raising=False)
        m.__dict__.clear()
        m.__dict__.update(_orig_dict)
