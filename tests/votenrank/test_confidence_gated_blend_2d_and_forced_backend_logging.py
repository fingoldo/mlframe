"""confidence_gated_blend handles 2-D input above the dispatch size, and a failed forced cupy backend is logged above DEBUG."""
import logging

import numpy as np

from mlframe.votenrank.confidence_gated_blend import confidence_gated_blend


def test_large_two_dimensional_input_does_not_reach_the_one_dimensional_kernels():
    """5000 x 3 with the njit backend forced raised a numba TypingError (the kernels allocate a 1-D output); below 2000 rows the numpy path handled the same call."""
    rng = np.random.default_rng(0)
    e, a, c = (rng.random((5000, 3)) for _ in range(3))
    out = confidence_gated_blend(e, a, c, 0.5, 0.1, force_backend="njit")
    assert out.shape == (5000, 3)
    expected = confidence_gated_blend(e[:100], a[:100], c[:100], 0.5, 0.1)
    assert np.allclose(out[:100], expected)


def test_forced_cupy_failure_is_logged_at_warning(caplog, monkeypatch):
    """The fallback to numpy was DEBUG-only, hiding a failed GPU request."""
    import sys

    mod = sys.modules["mlframe.votenrank.confidence_gated_blend"]

    def boom(*args, **kwargs):
        """A cupy backend that fails."""
        raise RuntimeError("no gpu")

    monkeypatch.setattr(mod, "_blend_cupy", boom)
    rng = np.random.default_rng(0)
    e, a, c = (rng.random(3000) for _ in range(3))
    with caplog.at_level(logging.WARNING):
        out = confidence_gated_blend(e, a, c, 0.5, 0.1, force_backend="cupy")
    assert out.shape == (3000,)
    assert any("cupy backend" in r.message for r in caplog.records)
