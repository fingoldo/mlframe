"""Audit 2026-11-20 finding 28-F36: a host-RAM fallback from a failed GPU memory probe is not cached under the GPU key.

The fallback value used to land under cuda:N, so later batch sizes on the GPU came from host RAM. The test makes the GPU probe fail with
CUDA reported available, and checks the GPU cache slot stays empty.
"""

from mlframe.training import mlp_runtime_defaults as mrd


def test_host_fallback_is_not_cached_under_the_gpu_key(monkeypatch):
    """With the GPU probe failing, the value returned is the host figure and the cuda:0 slot is not filled."""
    monkeypatch.setattr(mrd, "_PROBE_MEM_CACHE", {})
    monkeypatch.setattr(mrd, "_measure_available_bytes", lambda cuda_ok, device_id: (12345, False))
    value = mrd._probe_available_memory_bytes(cuda_available=True, device_id=0)
    assert value == 12345
    assert "cuda:0" not in mrd._PROBE_MEM_CACHE
