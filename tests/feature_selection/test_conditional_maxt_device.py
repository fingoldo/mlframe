"""The device conditional-MI null of the partner gate equals the host kernel's (counts are integers, the sums agree to rounding), and the gate's verdicts do not depend on the path."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._mrmr_fe_step import _conditional_maxt as C
from mlframe.feature_selection.filters._mrmr_fe_step._conditional_maxt_gpu import conditional_mi_targets_device


def _device_ok() -> bool:
    """Whether a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no driver: skipped
        return False


pytestmark = pytest.mark.skipif(not _device_ok(), reason="no CUDA device")


def _inputs(seed=0, n=4000, p=30, ky=3, kz=5):
    """Random ordinal codes of different widths, a conditioning column and several targets (the first is the observed one)."""
    rng = np.random.default_rng(seed)
    widths = rng.integers(2, 9, p).astype(np.int64)
    codes = np.stack([rng.integers(0, int(w), n) for w in widths]).astype(np.int16)
    z = rng.integers(0, kz, n).astype(np.int64)
    targets = rng.integers(0, ky, (6, n)).astype(np.int64)
    return codes, widths, targets, z, ky, kz


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_device_values_equal_the_host_kernel(seed):
    """Every (target, column) value agrees with the host kernel to rounding."""
    codes, widths, targets, z, ky, kz = _inputs(seed)
    got = conditional_mi_targets_device(codes, widths, targets, z, ky, kz)
    assert got is not None and got.shape == (targets.shape[0], codes.shape[0])
    for b in range(targets.shape[0]):
        want = np.empty(codes.shape[0])
        C._conditional_mi_columns(codes, widths, targets[b], z, ky, kz, want)
        np.testing.assert_allclose(got[b], want, rtol=0, atol=1e-12)


def test_oversized_tables_decline_to_the_host():
    """A table beyond the shared-memory cap or too many strata returns None (the host kernel answers)."""
    codes, widths, targets, z, ky, kz = _inputs(3)
    assert conditional_mi_targets_device(codes, np.full_like(widths, 5000), targets, z, ky, kz) is None
    assert conditional_mi_targets_device(codes, widths, targets, z, ky, 65) is None


def test_gate_verdicts_do_not_depend_on_the_path(monkeypatch):
    """The partner gate gives the same observed values and nearly the same floor with the device path forced and with it off."""
    rng = np.random.default_rng(5)
    n, p = 6000, 12
    data = rng.integers(0, 6, (n, p + 1)).astype(np.int8)
    y = (data[:, 0] + data[:, 1] + rng.integers(0, 3, n) > 7).astype(np.int64)
    data = np.ascontiguousarray(data)
    monkeypatch.setattr(C, "DEVICE_MIN_CELLS", 1)
    on = C.ConditionalPartnerGate(data, y, list(range(1, p + 1)), seed=3)._compute(0)
    monkeypatch.setattr(C, "DEVICE_MIN_CELLS", 10**18)
    off = C.ConditionalPartnerGate(data, y, list(range(1, p + 1)), seed=3)._compute(0)
    assert on is not None and off is not None
    np.testing.assert_allclose(on[0], off[0], rtol=0, atol=1e-12)
    assert on[1] == pytest.approx(off[1], rel=0, abs=1e-12)
