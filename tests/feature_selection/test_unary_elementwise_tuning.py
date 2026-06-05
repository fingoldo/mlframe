"""Residency-aware backend selection for the elementwise unary FE path."""
import numpy as np

from mlframe.feature_selection.filters import _unary_elementwise_tuning as u


def test_fallback_is_residency_aware():
    # DRAM-resident: cupy only above the breakeven; VRAM-resident: cupy always (if available).
    if u._HAS_CUPY:
        assert u._unary_fallback_choice(u._UNARY_DEFAULT_MIN_CELLS + 1, "host") == "cupy"
        assert u._unary_fallback_choice(1000, "host") == "numpy"
        assert u._unary_fallback_choice(1000, "device") == "cupy"  # no transfer to pay
    else:
        assert u._unary_fallback_choice(10_000_000, "device") == "numpy"  # no GPU -> numpy


def test_public_choice_returns_valid_backend():
    u._UNARY_SPEC._choice_cache.clear()  # the dispatch now memoizes via the spec
    for loc in ("host", "device"):
        assert u.unary_elementwise_backend_choice(1000, loc) in ("numpy", "cupy")


def test_variant_wrappers_agree_on_host_input():
    # numpy and cupy unary must produce the same values (the equiv gate relies on it).
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    ref = u._unary_numpy(x)
    if u._HAS_CUPY:
        got = u._unary_cupy(x)
        got = got.get() if hasattr(got, "get") else got
        assert np.allclose(ref, got, rtol=1e-4, atol=1e-5)
