"""Cluster 28 regression: the predict batch sizing probes the device it is asked for (audit 2026-11-20, finding 28-F37).

The resolver used to probe device 0 regardless of where the model ran. It now takes a device_id and forwards it to the memory probe. The
default stays 0, so existing callers keep their behaviour.
"""

import pytest

from mlframe.training import mlp_runtime_defaults as mrd


@pytest.fixture
def probe_calls(monkeypatch):
    """Replace the memory probe with a recorder that returns a fixed budget."""
    calls = []

    def _fake(*, cuda_available=None, device_id=0):
        """Record the device id the resolver asked for and return a fixed budget."""
        calls.append(device_id)
        return 8 * 1024**3

    monkeypatch.setattr(mrd, "_probe_available_memory_bytes", _fake)
    return calls


def test_resolver_probes_the_requested_device(probe_calls):
    """A device_id of 1 is forwarded to the memory probe, not replaced by 0 (28-F37)."""
    mrd.resolve_mlp_predict_batch_size(n_features=64, device_id=1)
    assert probe_calls == [1]


def test_default_device_is_zero(probe_calls):
    """Control: without an explicit device the probe still targets device 0 (28-F37)."""
    mrd.resolve_mlp_predict_batch_size(n_features=64)
    assert probe_calls == [0]
