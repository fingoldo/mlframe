"""Under STRICT the resident usability greedy runs only when the call carries enough work; a tiny pool on few rows stays on the host path with the same selection."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _usability_aware_selection as ua
from mlframe.feature_selection.filters import _usability_greedy_gpu_resident as resident


def _pool(n, p, seed=0):
    """A pool of p usable candidates over n rows and a target they linearly explain."""
    rng = np.random.default_rng(seed)
    cols = [rng.normal(size=n) for _ in range(p)]
    y = cols[0] - 0.5 * cols[1] + 0.1 * rng.normal(size=n)
    pool = [ua.UsableCandidate(name=f"c{i}", values=c, mi=float(abs(np.corrcoef(c, y)[0, 1]))) for i, c in enumerate(cols)]
    return pool, y


@pytest.fixture
def calls(monkeypatch):
    """Record whether the resident twin was entered, delegating to the real one."""
    seen = []
    real = resident.usability_greedy_gpu_resident
    monkeypatch.setattr(resident, "usability_greedy_gpu_resident", lambda *a, **k: seen.append(1) or real(*a, **k))
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    return seen


def test_a_tiny_call_stays_on_the_host(calls):
    """9 candidates on 3000 rows (27k cells) is far below the 1e6-cell floor."""
    pool, y = _pool(3_000, 9)
    ua.usability_greedy(pool, y)
    assert calls == []


def test_a_call_with_enough_work_uses_the_resident_twin(calls):
    """8 candidates on 200k rows clears the floor."""
    pool, y = _pool(200_000, 8)
    ua.usability_greedy(pool, y)
    assert calls == [1]


def test_host_and_resident_select_the_same_candidates(monkeypatch):
    """The size gate only changes where the greedy runs, not what it selects."""
    pool, y = _pool(4_000, 7, seed=3)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    host = [c.name for c in ua.usability_greedy(pool, y)]
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_strict.fe_gpu_strict_enabled", lambda **k: True)
    dev = [c.name for c in ua.usability_greedy(pool, y)]
    assert host == dev
