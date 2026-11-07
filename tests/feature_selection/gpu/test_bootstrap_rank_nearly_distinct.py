"""The device walk of nearly distinct columns (opt-in) gives the host walk's MI: columns with a bin end inside a run of equal values are left to the host, the others match to rounding."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _bootstrap_rank_mi_gpu as G
from mlframe.feature_selection.filters import _bootstrap_rank_mi_njit as H
from mlframe.feature_selection.filters._orthogonal_bootstrap_mi_fe import seeded_bootstrap_rows


def _device_ok() -> bool:
    """Whether a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no driver: skipped
        return False


pytestmark = pytest.mark.skipif(not _device_ok(), reason="no CUDA device")


def _inputs(n=60000, k=12, seed=3):
    """Continuous float32 columns, some with rounding that makes many equal neighbours and some with a few injected duplicates."""
    rng = np.random.default_rng(seed)
    cols = []
    for j in range(k):
        c = rng.standard_normal(n).astype(np.float32)
        if j % 4 == 1:
            c = np.round(c, 4)
        if j % 4 == 2:
            c[rng.integers(0, n, 200)] = c[rng.integers(0, n, 200)]
        cols.append(c)
    arr = np.stack(cols, axis=1)
    y = (rng.random(n) < 0.4).astype(np.int64)
    return arr, y, seeded_bootstrap_rows(0, y, n, int(0.8 * n), 10)


def test_nearly_distinct_columns_match_the_host_walk(monkeypatch):
    """With the opt-in on, the finished columns equal the host's MI and the rest are marked not done."""
    arr, y, reps = _inputs()
    n = arr.shape[0]
    prep = H._resample_tables(reps, y, n, 10, True)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_TIES_DEVICE", "1")
    mi_dev, done = G.distinct_walk_gpu(arr, prep, 10)
    host_mi, _host_ok = H.bootstrap_rank_mi(arr, y, reps, 10, try_tied_device=False)
    assert done.any()
    np.testing.assert_allclose(mi_dev[:, done], host_mi[:, done], rtol=0, atol=1e-12)


def test_default_leaves_tied_columns_to_the_host(monkeypatch):
    """Without the opt-in only fully distinct columns are finished on the device."""
    arr, y, reps = _inputs()
    prep = H._resample_tables(reps, y, arr.shape[0], 10, True)
    monkeypatch.delenv("MLFRAME_BOOTSTRAP_RANK_TIES_DEVICE", raising=False)
    _mi, done = G.distinct_walk_gpu(arr, prep, 10)
    has_ties = np.array([len(np.unique(arr[:, j])) < arr.shape[0] for j in range(arr.shape[1])])
    assert not (done & has_ties).any()


def test_full_call_is_unchanged_by_the_opt_in(monkeypatch):
    """`bootstrap_rank_mi` returns the same MI with the opt-in on and off."""
    arr, y, reps = _inputs()
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_DEVICE", "1")
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_TIES_DEVICE", "0")
    off, ok_off = H.bootstrap_rank_mi(arr, y, reps, 10, try_tied_device=False)
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_TIES_DEVICE", "1")
    on, ok_on = H.bootstrap_rank_mi(arr, y, reps, 10, try_tied_device=False)
    np.testing.assert_allclose(on, off, rtol=0, atol=1e-12)
    assert (ok_on == ok_off).all()
