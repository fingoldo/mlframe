"""Regression: the exhaustive-synergy decision must NOT be gated on GPU PRESENCE.

Found by the CPU/GPU-equivalency agent (P0-1, 2026-06-22): decide_exhaustive_sweep hard-declined when no
CUDA device was present, so a balanced (L=0) interaction feature -- which only the exhaustive C(p,2) sweep
recovers -- existed on a CUDA host and silently vanished on a CPU host. The fix predicts on the available
backend (CUDA where present, else the CPU njit-prange kernel) and runs the sweep there, so the decision is
hardware-INDEPENDENT (affordable-or-not), not device-gated.
"""

from __future__ import annotations

import types

import pytest

from mlframe.feature_selection.filters import _fe_synergy_exhaustive as ex


def _stub(mode, *, max_seconds=None, max_mins=None):
    s = types.SimpleNamespace()
    s.fe_synergy_exhaustive = mode
    s.fe_synergy_exhaustive_max_seconds = max_seconds
    s.max_runtime_mins = max_mins
    return s


@pytest.fixture
def no_gpu(monkeypatch):
    import mlframe.feature_selection.filters.batch_pair_mi_gpu as bg

    monkeypatch.setattr(bg, "_CUDA_AVAIL", False, raising=False)
    return bg


def test_force_runs_exhaustive_on_cpu_without_gpu(no_gpu):
    use, reason = ex.decide_exhaustive_sweep(_stub("force"), n_samples=2000, n_raw=10, verbose=0)
    assert use is True, f"force must run exhaustive even without a GPU (on CPU): {reason}"
    assert "CPU" in reason


def test_auto_runs_exhaustive_on_cpu_when_affordable(no_gpu):
    # Unlimited budget (no max_runtime_mins / override) -> auto must sweep on CPU, not decline.
    use, reason = ex.decide_exhaustive_sweep(_stub("auto"), n_samples=2000, n_raw=8, verbose=0)
    assert use is True, f"auto + unlimited budget must run exhaustive on CPU: {reason}"
    assert "CPU" in reason


def test_auto_declines_on_cost_not_on_gpu_absence(no_gpu):
    # A tiny budget makes the large CPU sweep unaffordable -> decline, but the REASON must be cost/budget,
    # NOT "no CUDA GPU available" (the old device-gate wording).
    use, reason = ex.decide_exhaustive_sweep(
        _stub("auto", max_seconds=0.001),
        n_samples=1_000_000,
        n_raw=2000,
        verbose=0,
    )
    assert use is False
    assert "budget" in reason.lower()
    assert "no cuda gpu available" not in reason.lower(), "decline must be cost-based, not device-gated (the P0-1 regression)"


def test_never_mode_still_declines(no_gpu):
    use, _ = ex.decide_exhaustive_sweep(_stub("never"), n_samples=2000, n_raw=10, verbose=0)
    assert use is False
