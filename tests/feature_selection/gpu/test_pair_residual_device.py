"""The device pair residual scorer equals the numba pass; pair_interaction_screen returns the same pairs with the device forced on, and the device path demonstrably ran."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _pair_residual_gpu as g
from mlframe.feature_selection.filters import _pair_residual_screen as screen


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")


def test_device_equals_numba_pass_on_every_pair():
    """t statistics agree to 1e-9 relative, for an even mask that is not the alternating one and for an odd number of rows."""
    _needs_gpu()
    rng = np.random.default_rng(1)
    n, p, nb = 7001, 12, 8
    Q = rng.integers(0, nb, (p, n)).astype(np.int64)
    resid = rng.standard_normal(n) + 0.6 * (((Q[0] + Q[1]) % 2) - 0.5)
    even = rng.random(n) < 0.5
    pairs = np.array([(a, b) for a in range(p) for b in range(a + 1, p)], dtype=np.int64)
    want = screen._score_pairs(Q, resid, even, pairs, nb, 3.0)
    got = g.score_pairs_gpu(Q, resid, even, pairs, nb, 3.0)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    assert want[0] > 3.0, "the planted interaction pair must score high for the comparison to prove anything"


def test_screen_finds_the_same_pairs_with_the_device_forced(monkeypatch):
    """The reported (a, b, t) triples are the same with the device off and on, and the device scorer is called exactly once."""
    _needs_gpu()
    rng = np.random.default_rng(2)
    n = 6000
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(8)})
    y = np.sign(X["x0"].to_numpy() * X["x1"].to_numpy()) + 0.3 * rng.standard_normal(n) + 0.8 * X["x2"].to_numpy()
    monkeypatch.setattr(g, "pair_device_chosen", lambda n_, k: False)
    want = screen.pair_interaction_screen(X, y)
    monkeypatch.setattr(g, "pair_device_chosen", lambda n_, k: True)
    calls = {"n": 0}
    real = g.score_pairs_gpu

    def counting(*a, **k):
        """The device scorer, counting calls so the test proves the device path ran."""
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(g, "score_pairs_gpu", counting)
    got = screen.pair_interaction_screen(X, y)
    assert calls["n"] == 1
    assert [(a, b) for a, b, _ in got] == [(a, b) for a, b, _ in want] and ("x0", "x1") in [(a, b) for a, b, _ in got]
    np.testing.assert_allclose([t for *_, t in got], [t for *_, t in want], rtol=1e-9)
