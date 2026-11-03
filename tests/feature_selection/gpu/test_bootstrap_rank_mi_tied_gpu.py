"""The device counting sort of the bootstrap rank MI gives the host's MI for columns with repeated values."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._bootstrap_rank_mi_njit import bootstrap_rank_mi


def _frame(n: int, rng) -> np.ndarray:
    """Columns with few levels, a dominant level spanning several bins, many levels above the device cap, distinct values, and one non-finite column."""
    base = rng.normal(size=n) * 30
    cols = [
        np.round(base),
        np.floor(np.abs(base)) % 10,
        np.where(rng.random(n) < 0.6, 0.0, np.round(base)),  # one level holds 60% of the rows: it straddles several bin ends
        np.round(base * 100) / 100,  # more levels than the shared table holds: stays on the host
        base,  # distinct: host walk
        (rng.random(n) < 0.3).astype(float),
    ]
    bad = np.round(base * 10) / 10
    bad[5] = np.nan
    cols.append(bad)
    return np.ascontiguousarray(np.stack(cols, axis=1).astype(np.float32))


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("n_bins", [5, 10])
def test_device_mi_equals_host(monkeypatch, n_classes, n_bins):
    """Same finite flags, MI equal to rounding (the device sums the same table in the same order, ``log`` may differ by an ulp)."""
    rng = np.random.default_rng(n_classes * 10 + n_bins)
    n = 20000
    arr = _frame(n, rng)
    y = rng.integers(0, n_classes, n)
    reps = [rng.integers(0, n, int(0.8 * n)) for _ in range(6)]
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_DEVICE", "0")
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE", "0")
    want, ok_want = bootstrap_rank_mi(arr, y, reps, n_bins)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE", "1")
    got, ok_got = bootstrap_rank_mi(arr, y, reps, n_bins)
    assert np.array_equal(ok_got, ok_want)
    assert not ok_got[-1]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_streams_many_column_blocks(monkeypatch):
    """More columns than one device block: every block is scored and placed at its own columns."""
    rng = np.random.default_rng(1)
    n = 8000
    arr = np.ascontiguousarray((rng.integers(0, 7, size=(n, 300)) * rng.random(300)).astype(np.float32))
    y = rng.integers(0, 2, n)
    reps = [rng.integers(0, n, int(0.8 * n)) for _ in range(4)]
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_DEVICE", "0")
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE", "0")
    want, _ = bootstrap_rank_mi(arr, y, reps, 10)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE", "1")
    got, _ = bootstrap_rank_mi(arr, y, reps, 10)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
