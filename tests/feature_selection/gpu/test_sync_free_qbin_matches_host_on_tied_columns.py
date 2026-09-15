"""The sync-free device quantile binner assigns every row the same bin as the host ``_quantile_bin``, tied columns included.

On columns with repeated values (rounded measurements, step functions) the device binner put a tied value's whole mass in the neighbouring bin
for a few columns in every block: 2 to 5 of every 32 to 128 columns, about a thousand rows each. That binner feeds strict-GPU FE scoring
(the CMI redundancy gate and the escalation pool), so those candidates were scored on a partition the host path never produces.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin, _sync_free_qbin_codes

NBINS = 10


def _gpu_available() -> bool:
    """True when a CUDA device can run a trivial kernel."""
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0 and int(cp.asnumpy(cp.arange(2).sum())) == 1
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="needs a CUDA device")


def _tied_block(n: int, k: int, seed: int) -> np.ndarray:
    """Correlated columns, a quarter of them rounded to one decimal (ties at every value), some shifted by a class step."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    block = rng.normal(size=(n, 1)) * rng.uniform(0.1, 2.0, size=(1, k)) + rng.normal(size=(n, k))
    block[:, ::4] = np.round(block[:, ::4], 1)
    block[:, 1::7] = np.exp(block[:, 1::7] * 0.1) + y[:, None] * 0.3
    return block


@pytest.mark.parametrize("n,k,seed", [(50_000, 32, 0), (50_000, 128, 0), (250_000, 32, 0), (20_000, 64, 7)])
def test_device_codes_equal_host_codes_on_tied_block(n, k, seed):
    """Every column of a tie-heavy block bins identically on device and host."""
    block = _tied_block(n, k, seed)
    mismatched = []
    for j in range(k):
        col = np.ascontiguousarray(block[:, j])
        host = np.asarray(_quantile_bin(col, NBINS))
        dev = cp.asnumpy(_sync_free_qbin_codes(cp, cp.asarray(col), NBINS))
        if not np.array_equal(dev, host):
            mismatched.append((j, int((dev != host).sum())))
    assert not mismatched, f"device codes differ from host on (column, rows): {mismatched[:10]}"


@pytest.mark.parametrize(
    "values",
    [
        np.repeat([0.1, 0.2, 0.3, 0.7], 5000),
        np.round(np.linspace(-3.0, 3.0, 40_000), 1),
        np.where(np.arange(30_000) % 3 == 0, 0.0, np.arange(30_000) * 0.1),
        np.full(10_000, 2.5),
        np.repeat([1.0, 2.0], 7000),
    ],
    ids=["four_levels", "rounded_ramp", "mass_point", "constant", "two_values"],
)
def test_device_codes_equal_host_codes_on_degenerate_shapes(values):
    """Low-cardinality, mass-point, constant and two-value columns keep the host partition and cardinality."""
    col = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    host = np.asarray(_quantile_bin(col, NBINS))
    dev = cp.asnumpy(_sync_free_qbin_codes(cp, cp.asarray(col), NBINS))
    np.testing.assert_array_equal(dev, host)
