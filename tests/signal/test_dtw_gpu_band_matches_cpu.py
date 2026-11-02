"""GPU DTW uses the same Sakoe-Chiba band as dtaidistance and unequal lengths stay on the CPU (tracker 2026-11-02 wave 13, PAR13B-1, PAR13B-2)."""

import numpy as np
import pytest

from mlframe.signal import dtw as dtwm


def test_unequal_lengths_are_dispatched_to_the_cpu(monkeypatch):
    """dtaidistance widens its band by |n - m| while the kernels do not; with the GPU forced available the auto path must still pick the CPU."""
    called = []
    monkeypatch.setattr(dtwm, "dtw_cpu", lambda x, y, **k: called.append("cpu") or (0.0, []))
    monkeypatch.setattr(dtwm, "dtw_cupy", lambda *a, **k: called.append("cupy") or (0.0, []))
    monkeypatch.setattr(dtwm, "_HAS_CUPY", True)
    monkeypatch.delenv("MLFRAME_DTW_BACKEND", raising=False)
    dtwm.dtw_dispatch(np.zeros(10), np.zeros(400), window=200)
    assert called == ["cpu"]


@pytest.mark.skipif(not dtwm._HAS_CUPY, reason="needs cupy")
def test_cupy_banded_window_one_allows_only_the_diagonal():
    """x=[0,1,0,0], y=[0,0,1,0], window=1: dtaidistance gives sqrt(2); the GPU band was one cell wider and gave 0."""
    pytest.importorskip("dtaidistance")
    x = np.array([0.0, 1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 0.0])
    cpu, _ = dtwm.dtw_cpu(x, y, window=1)
    gpu, _ = dtwm.dtw_cupy(x, y, window=1)
    assert gpu == pytest.approx(cpu, abs=1e-5)
    assert cpu == pytest.approx(np.sqrt(2.0), abs=1e-5)
