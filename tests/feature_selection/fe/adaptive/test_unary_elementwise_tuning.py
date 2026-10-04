"""Residency-aware backend selection for the elementwise unary FE path."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _unary_elementwise_tuning as u


def test_fallback_is_residency_aware(monkeypatch):
    # DRAM-resident: cupy only above the breakeven; VRAM-resident: cupy always (if available).
    """Fallback is residency aware."""
    with monkeypatch.context() as m:
        m.setattr(u, "_HAS_CUPY", True)
        assert u._unary_fallback_choice(u._UNARY_DEFAULT_MIN_CELLS + 1, "host") == "cupy"
        assert u._unary_fallback_choice(u._UNARY_DEFAULT_MIN_CELLS - 1, "host") == "numpy"
        assert u._unary_fallback_choice(1000, "host") == "numpy"
        assert u._unary_fallback_choice(1000, "device") == "cupy"  # no transfer to pay
    with monkeypatch.context() as m:
        m.setattr(u, "_HAS_CUPY", False)
        assert u._unary_fallback_choice(10_000_000, "device") == "numpy"  # no GPU -> numpy
        assert u._unary_fallback_choice(10_000_000, "host") == "numpy"


def test_public_choice_returns_valid_backend():
    """Public choice returns valid backend."""
    u._UNARY_SPEC._choice_cache.clear()  # the dispatch now memoizes via the spec
    for loc in ("host", "device"):
        assert u.unary_elementwise_backend_choice(1000, loc) in ("numpy", "cupy")


def test_variant_wrappers_agree_on_host_input():
    # numpy and cupy unary must produce the same values (the equiv gate relies on it).
    """Variant wrappers agree on host input."""
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    ref = u._unary_numpy(x)
    np.testing.assert_array_equal(ref, np.cos(x))
    assert ref.dtype == x.dtype


def _cuda_usable() -> bool:
    """True when cupy imports and a CUDA device answers."""
    if not u._HAS_CUPY:
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def test_cupy_variant_matches_numpy_on_host_input():
    """The cupy unary variant returns the numpy variant's values on a host array."""
    if not _cuda_usable():
        pytest.skip("no usable CUDA device")
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    got = u._unary_cupy(x)
    got = got.get() if hasattr(got, "get") else got
    assert got.shape == x.shape
    np.testing.assert_allclose(got, u._unary_numpy(x), rtol=1e-4, atol=1e-5)
