"""The FE step scores its CMI-gate candidates and escalation pool with one batched device workload, equal to the per-candidate path.

Each candidate used to be binned and scored separately (one upload, one binning, one MI kernel launch per candidate) although batched
twins of both stages already existed. The batched helper must return the same marginal MI as the host per-candidate computation, split
its upload into chunks without changing a value, and hand back ``None`` (so the caller keeps the per-candidate loop) whenever a column is
not all-finite or the device path fails.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin
from mlframe.feature_selection.filters._mrmr_fe_step import _step_batched_marginals as sbm

NBINS = 10


def _gpu_available() -> bool:
    """True when a CUDA device can run a trivial kernel."""
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0 and int(cp.asnumpy(cp.arange(2).sum())) == 1
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="needs a CUDA device")


def _candidates(n: int, k: int, seed: int):
    """Correlated candidate columns, a quarter rounded (ties), some shifted by a class step, plus 3-class codes."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    block = rng.normal(size=(n, 1)) * rng.uniform(0.1, 2.0, size=(1, k)) + rng.normal(size=(n, k))
    block[:, ::4] = np.round(block[:, ::4], 1)
    block[:, 1::7] = np.exp(block[:, 1::7] * 0.1) + y[:, None] * 0.3
    return [np.ascontiguousarray(block[:, j]) for j in range(k)], y


def _host_marginals(cols, y):
    """The per-candidate host computation the batched helper replaces."""
    return [float(_cmi_from_binned(np.asarray(_quantile_bin(c, NBINS)), y, None, kx=NBINS)) for c in cols]


@pytest.mark.parametrize("n,k", [(20_000, 8), (50_000, 32), (50_000, 128)])
def test_batched_marginals_match_per_candidate_host(n, k):
    """Every candidate's batched marginal MI equals the host per-candidate value."""
    cols, y = _candidates(n, k, seed=0)
    got = sbm.batched_device_marginals(cols, y, NBINS)
    assert got is not None and len(got) == k
    np.testing.assert_allclose(got, _host_marginals(cols, y), rtol=0, atol=1e-12)


def test_chunked_upload_gives_identical_values(monkeypatch):
    """Forcing many small chunks returns exactly the single-chunk values."""
    cols, y = _candidates(20_000, 40, seed=3)
    whole = sbm.batched_device_marginals(cols, y, NBINS)
    monkeypatch.setattr(sbm, "_MAX_BLOCK_ELEMENTS", 20_000 * 7)  # 7 columns per chunk -> 6 chunks
    chunked = sbm.batched_device_marginals(cols, y, NBINS)
    assert whole is not None and chunked is not None
    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(whole))


def test_non_finite_column_returns_none():
    """A column with a NaN makes the helper decline, so the caller's per-candidate path (which bins non-finite values on the host) runs."""
    cols, y = _candidates(5_000, 4, seed=1)
    cols[2] = cols[2].copy()
    cols[2][10] = np.nan
    assert sbm.batched_device_marginals(cols, y, NBINS) is None


def test_device_failure_returns_none_with_warning(monkeypatch, caplog):
    """A device error is reported and leaves the per-candidate path to score the pool."""
    import mlframe.feature_selection.filters._fe_batched_mi as fbm

    def _boom(*args, **kwargs):
        """Simulate a device fault."""
        raise RuntimeError("simulated device fault")

    monkeypatch.setattr(fbm, "batched_cmi_gpu", _boom)
    cols, y = _candidates(5_000, 4, seed=2)
    with caplog.at_level(logging.WARNING):
        assert sbm.batched_device_marginals(cols, y, NBINS) is None
    assert any("batched device marginal MI failed" in r.getMessage() for r in caplog.records)


def test_empty_pool_returns_empty_list():
    """No candidates: nothing to score, and no device call."""
    assert sbm.batched_device_marginals([], np.zeros(10, dtype=np.int64), NBINS) == []
