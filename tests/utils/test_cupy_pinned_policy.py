"""Uploads under the pinned policy: above-cap and failed pinned allocations take cupy's synchronous copy and the data arrive intact."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _device_ok():
    """Device ok."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _device_ok(), reason="no CUDA device")


def test_uploads_intact_below_and_above_cap():
    """Uploads intact below and above cap."""
    from mlframe.utils.cupy_pinned import install_pinned_policy

    install_pinned_policy()
    rng = np.random.default_rng(0)
    for n in (1000, (64 << 20) // 8):
        a = rng.standard_normal(n)
        np.testing.assert_array_equal(cp.asnumpy(cp.asarray(a)), a)


def test_failed_pinned_alloc_falls_back(monkeypatch):
    """Failed pinned alloc falls back."""
    from cupy_backends.cuda.api import runtime

    import mlframe.utils.cupy_pinned as mod

    class _Boom:
        """Boom."""

        def malloc(self, nbytes):
            """Malloc."""
            raise runtime.CUDARuntimeError(208)  # cudaErrorAlreadyMapped, as the Windows driver reports a failed cudaHostAlloc

    monkeypatch.setattr(mod, "_installed", False)
    monkeypatch.setattr(mod, "_make_pool", lambda _pm: _Boom())
    monkeypatch.setenv("MLFRAME_PINNED_STAGING_MAX_MB", "64")
    mod.install_pinned_policy()
    a = np.arange(1000, dtype=np.float64)
    np.testing.assert_array_equal(cp.asnumpy(cp.asarray(a)), a)
