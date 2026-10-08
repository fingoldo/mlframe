"""The ext-val candidate codes stay on the device when the caller feeds them to the resident noise gate, and the survivor column need not round-trip."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _gpu_resident_fe as gfe
from mlframe.feature_selection.filters._gpu_resident_extval import gpu_materialise_extval_codes_host
from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit

_OPS = np.array([0, 1, 2, 3], dtype=np.int64)


def _inputs(n: int = 4000):
    """A survivor column and three external factors."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(n), [rng.standard_normal(n) for _ in range(3)]


def test_deferred_extval_codes_skip_the_host_copy_and_fill_on_demand_identically():
    """No device->host copy at materialise time; the stashed device codes are the eager codes, and a host read fills the same bytes."""
    a, bs = _inputs()
    eager = gpu_materialise_extval_codes_host(a, bs, _OPS, 10)
    with residency_audit() as rep:
        out = gpu_materialise_extval_codes_host(a, bs, _OPS, 10, defer_host_fill=True)
    assert not rep.d2h
    try:
        dev = gfe.take_resident_codes(out)
        assert dev is not None
        gfe.ensure_host_codes_filled(out)
        np.testing.assert_array_equal(out, eager)
        np.testing.assert_array_equal(cp.asnumpy(dev), eager)
    finally:
        gfe.clear_resident_codes_handoff(out)


def test_device_survivor_column_gives_the_same_codes_as_the_host_column():
    """Passing the survivor column already on the device equals uploading the host copy, without needing the host copy."""
    a, bs = _inputs()
    from_host = gpu_materialise_extval_codes_host(a, bs, _OPS, 10)
    from_dev = gpu_materialise_extval_codes_host(None, bs, _OPS, 10, param_a_dev=cp.asarray(a, dtype=cp.float32).astype(cp.float64))
    np.testing.assert_array_equal(from_dev, from_host)
