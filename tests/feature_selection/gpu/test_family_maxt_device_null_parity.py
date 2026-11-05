"""The family max-T gate returns the same mask whether its permutation null runs on the device or on the host.

The device null has two branches. Up to ``BITSET_MAX_BINS`` bins a column is counted through bit sets on the device
(``null_max_bits_device``); above it the histogram kernel runs there (``family_null_max_device``). Both are only taken when
``_null_device_wanted`` says so, which the tuned backend choice does not do for small problems, so the test forces it on and
counts the device calls to make sure the branch really ran.

A dense family with a strong signal is decided by the analytic triage before any permutation is drawn, so the bit-set case uses
pure noise (the triage stays undecided and the permutation null runs), and the histogram case uses a sparse table, which the
triage never decides.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _family_maxt_cuda as cuda
from mlframe.feature_selection.filters import _family_maxt_cuda_bits as cuda_bits
from mlframe.feature_selection.filters import _family_maxt_gate as gate

if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)


def _data(seed: int, signal: bool):
    """Six standard-normal candidates; with ``signal`` the first one carries the target, otherwise the target is pure noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 6))
    if signal:
        y = (X[:, 0] + 0.3 * rng.normal(size=400) > 0).astype(np.int64)
    else:
        y = rng.integers(0, 2, size=400).astype(np.int64)
    return X, y


@pytest.mark.parametrize(
    "n_bins, signal, device_module, device_name",
    [(8, False, cuda_bits, "null_max_bits_device"), (40, True, cuda, "family_null_max_device")],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_device_null_mask_equals_host_null_mask(monkeypatch, n_bins, signal, device_module, device_name, seed):
    """The gate's mask is identical on the device null and the host null, and the device branch ran at least once."""
    X, y = _data(seed, signal)
    real_device = getattr(device_module, device_name)
    calls = {"n": 0}

    def _counting_device(*args, **kwargs):
        """Count the device calls and forward them to the real kernel."""
        calls["n"] += 1
        return real_device(*args, **kwargs)

    monkeypatch.setattr(device_module, device_name, _counting_device)
    monkeypatch.setattr(gate, "_null_device_wanted", lambda cols, yy, n_perm: True)
    on_device = gate.family_maxt_pass(X, y, n_bins=n_bins, alpha=0.05, n_perm=199, seed=seed)
    assert calls["n"] > 0, "the device null was never taken, so the parity below proves nothing"

    monkeypatch.setattr(gate, "_null_device_wanted", lambda cols, yy, n_perm: False)
    on_host = gate.family_maxt_pass(X, y, n_bins=n_bins, alpha=0.05, n_perm=199, seed=seed)
    assert np.array_equal(on_device, on_host)
    if signal:
        assert on_device[0], "the signal column must pass on both paths"
