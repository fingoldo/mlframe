"""The GPU-resident nearest-neighbour propagation must pick the same neighbour as the host path on every row."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.data_valuation import _propagate_gpu_ktc as m

cp = pytest.importorskip("cupy")


def _has_device() -> bool:
    """True when a CUDA device is usable."""
    try:
        return cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_device(), reason="needs a CUDA device")


@pytest.mark.parametrize("dims", [{"n_full": 5_000, "n_sub": 2_000}, {"n_full": 20_000, "n_sub": 2_000}])
def test_resident_values_equal_host_values_row_by_row(dims):
    """Exact equality: the sweep runs with a zero relative tolerance, so any neighbour flip would be visible here."""
    args = m._make_propagate_inputs(dims)
    host = np.asarray(m._propagate_host(*args), dtype=np.float64)
    resident = np.asarray(m._propagate_resident(*args), dtype=np.float64)
    np.testing.assert_array_equal(resident, host)


def test_a_wrong_neighbour_is_detectable():
    """Negative control: shuffling the subsample values changes the output, so equality above is not vacuous."""
    X_full, X_sub, values = m._make_propagate_inputs({"n_full": 3_000, "n_sub": 2_000})
    good = np.asarray(m._propagate_resident(X_full, X_sub, values))
    bad = np.asarray(m._propagate_resident(X_full, X_sub, values[::-1].copy()))
    assert not np.array_equal(good, bad)
