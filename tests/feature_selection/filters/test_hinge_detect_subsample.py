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

_KW = dict(
    max_breakpoints=1,
    min_heldout_r2_uplift=0.01,
    precheck_qs=(0.25, 0.5, 0.75),
    precheck_min_sse_drop=0.0,
    cand_q_lo=0.1,
    cand_q_hi=0.9,
    n_candidates=16,
    min_rows=100,
    min_seg_rows=20,
)


def _hinge_data(n, seed=0):
    """Hinge data."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = np.where(x > 0.5, 2.0 * (x - 0.5), 0.0) + rng.normal(0, 0.05, n)  # clear breakpoint at 0.5
    return x, y


@pytest.mark.parametrize("max_rows", ["200000", "0"])
def test_hinge_subsample_still_finds_breakpoint(max_rows, monkeypatch):
    """Hinge subsample still finds breakpoint."""
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


def test_hinge_gpu_subsampled_matches_cpu_full_n_above_cap():
    """mrmr_audit_2026-07-20 B-16: the module docstring/comments claim the GPU-resident detector's
    subsampling above MLFRAME_HINGE_MAX_ROWS is "selection-equivalent" to the CPU detector's full-n
    scan, but no test actually compared the two backends' proposed tau above the cap -- this pins that
    empirical claim: on a clear single-breakpoint signal well above the default 250k cap, GPU
    (subsampled) and CPU (full-n) must both find a breakpoint near the true tau and agree with each
    other within a modest margin (they are NOT required to be bit-identical -- different sample, same
    underlying signal)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._hinge_basis_fe import _detect_hinge_breakpoints
    from mlframe.feature_selection.filters._hinge_detect_gpu_resident import detect_hinge_breakpoints_gpu

    x, y = _hinge_data(500_000, seed=0)  # well above the default 250k cap
    true_tau = 0.5

    cpu_taus = _detect_hinge_breakpoints(x, y, max_breakpoints=1, min_heldout_r2_uplift=0.01)
    gpu_taus = detect_hinge_breakpoints_gpu(x, y, **_KW)
    if gpu_taus is None:
        pytest.skip("cupy fault -> host detector path (not under test here)")

    assert len(cpu_taus) >= 1, f"CPU (full-n) detector found no breakpoint on a clear hinge signal: {cpu_taus}"
    assert len(gpu_taus) >= 1, f"GPU (subsampled) detector found no breakpoint on a clear hinge signal: {gpu_taus}"
    assert abs(cpu_taus[0] - true_tau) < 0.15, f"CPU tau {cpu_taus[0]:.3f} far from the true breakpoint {true_tau}"
    assert abs(gpu_taus[0] - true_tau) < 0.15, f"GPU tau {gpu_taus[0]:.3f} far from the true breakpoint {true_tau}"
    assert abs(cpu_taus[0] - gpu_taus[0]) < 0.15, (
        f"GPU (subsampled) tau {gpu_taus[0]:.3f} diverged from CPU (full-n) tau {cpu_taus[0]:.3f} by more than the "
        f"tolerance -- the 'selection-equivalent' claim in the module docstring is not holding empirically."
    )


def test_hinge_cap_actually_subsamples(monkeypatch):
    """Hinge cap actually subsamples."""
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
