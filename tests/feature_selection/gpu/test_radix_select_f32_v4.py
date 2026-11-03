"""The compaction v4 float32 radix select gives the same quantile edges as v3, including columns whose candidates overflow the scratch."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_select import _radix_select_interior_edges


def _frame(kind: str, n: int, k: int) -> np.ndarray:
    """Continuous, heavily tied (overflows the compaction scratch), constant, tiny-magnitude and mixed-sign columns."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((n, k)).astype(np.float32)
    if kind == "ties":
        x = np.round(x * 2).astype(np.float32)
    elif kind == "const":
        x[:] = 1.5
    elif kind == "tiny":
        x = (rng.exponential(1.0, (n, k)) * 1e-3).astype(np.float32)
    elif kind == "mixed":
        x[:, ::2] = 0.0
        x[::5] = -x[::5]
    return x


@pytest.mark.parametrize("kind", ["normal", "ties", "const", "tiny", "mixed"])
@pytest.mark.parametrize("n,k,nbins", [(30000, 50, 10), (1000, 7, 4), (5000, 40, 20), (20011, 13, 10)])
def test_v4_edges_equal_v3(monkeypatch, kind, n, k, nbins):
    """Same interior edges with ``MLFRAME_RADIX_F32_V4`` on and off."""
    cand = cp.asarray(_frame(kind, n, k))
    monkeypatch.setenv("MLFRAME_RADIX_F32_V4", "0")
    want = _radix_select_interior_edges(cand, nbins)
    monkeypatch.setenv("MLFRAME_RADIX_F32_V4", "1")
    got = _radix_select_interior_edges(cand, nbins)
    assert want is not None and got is not None
    assert cp.array_equal(got, want)
