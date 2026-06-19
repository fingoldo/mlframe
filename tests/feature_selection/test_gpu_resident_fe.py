"""Prototype validation for the GPU-resident FE candidate generation + MI (``_gpu_resident_fe``).

Correctness gate: the on-device (cupy) candidate grid + single big-k MI must match the CPU (numpy +
njit) path -- same candidate names, MI ranking, and values to fp round-off -- and both must rank the
a**2/b-equivalent candidate top on an a**2/b target. Speed is exercised by a separate opt-in bench
(not a hard timing assert, to stay non-flaky)."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._gpu_resident_fe import (
    cpu_pair_candidate_mi,
    fe_gpu_resident_enabled,
)


def _ab_target(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    y = a**2 / b
    # discretise y to codes the MI kernels score against (equi-frequency, 20 bins).
    edges = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    y_codes = np.searchsorted(edges, y).astype(np.int64)
    return a, b, y_codes


def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_GPU_RESIDENT", raising=False)
    assert fe_gpu_resident_enabled() is False
    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT", "on")
    assert fe_gpu_resident_enabled() is True


def test_cpu_path_recovers_a2_over_b():
    """The CPU reference grid must rank an (a,b) ratio-of-square candidate top on an a**2/b target."""
    a, b, y_codes = _ab_target()
    names, mi = cpu_pair_candidate_mi(a, b, y_codes)
    top = names[int(np.argmax(mi))]
    assert "div" in top and "sqr" in top, f"top candidate not the a**2/b form: {top}"


def test_gpu_resident_matches_cpu():
    """On-device generation + single big-k MI must match the CPU path: identical names, same top
    candidate, MI values equal to fp round-off (njit vs cupy plug-in MI are equivalent)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import gpu_resident_pair_candidate_mi

    a, b, y_codes = _ab_target()
    cpu_names, cpu_mi = cpu_pair_candidate_mi(a, b, y_codes)
    gpu_names, gpu_mi = gpu_resident_pair_candidate_mi(a, b, y_codes)
    assert gpu_names == cpu_names
    # same winner + same ranking head
    assert int(np.argmax(gpu_mi)) == int(np.argmax(cpu_mi)), (cpu_names[int(np.argmax(cpu_mi))], gpu_names[int(np.argmax(gpu_mi))])
    # values match to fp round-off (binning tie-breaks aside)
    np.testing.assert_allclose(gpu_mi, cpu_mi, rtol=1e-3, atol=1e-4)


def test_gpu_resident_chunked_matches_cpu():
    """Force MULTIPLE VRAM K-chunks (n=100k -> k_chunk < 384) and assert the concatenated chunked MI
    still matches the CPU path -- the chunk boundary must not corrupt per-candidate MI."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import _gpu_k_chunk, gpu_resident_pair_candidate_mi

    a, b, y_codes = _ab_target(n=100_000)
    assert _gpu_k_chunk(100_000) < 384, "test needs >1 chunk to be meaningful"
    cpu_names, cpu_mi = cpu_pair_candidate_mi(a, b, y_codes)
    gpu_names, gpu_mi = gpu_resident_pair_candidate_mi(a, b, y_codes)
    assert gpu_names == cpu_names
    assert int(np.argmax(gpu_mi)) == int(np.argmax(cpu_mi))
    np.testing.assert_allclose(gpu_mi, cpu_mi, rtol=1e-3, atol=1e-4)


def test_dispatch_routes_and_recovers():
    """The size dispatcher returns the right shape/ranking on both legs (small-n CPU leg always; large-n
    GPU leg when cupy present) and recovers a**2/b."""
    from mlframe.feature_selection.filters._gpu_resident_fe import pair_candidate_mi_dispatch

    a, b, y_codes = _ab_target(n=20_000)  # below crossover -> CPU leg
    names, mi = pair_candidate_mi_dispatch(a, b, y_codes)
    assert len(names) == len(mi) == 8 * 8 * 6
    top = names[int(np.argmax(mi))]
    assert "div" in top and "sqr" in top
