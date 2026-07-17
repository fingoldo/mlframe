"""Dispatch-threshold boundary + scalar length-extreme coverage for
``mlframe.metrics.rank_correlation``.

The existing ``test_rank_correlation`` file checks value-equivalence across paths. This file pins the
exact numpy<->numba crossover: with the threshold set to 100, rows < 100 must take the numpy path
(numba kernel untouched) and rows >= 100 must take the numba path -- verified by spying on the
``spearmanr_batched_numba`` entry point. Also covers the scalar dispatcher at length extremes (0, 1, 2).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def restore_threshold():
    """Save/restore the module-global dispatch threshold so boundary tests don't leak state."""
    import mlframe.metrics.rank_correlation as rc

    saved = rc._DISPATCH_NUMBA_MIN_ROWS
    yield rc
    rc._DISPATCH_NUMBA_MIN_ROWS = saved


@pytest.mark.parametrize(
    "n_rows,expect_numba",
    [(99, False), (100, True), (101, True)],
)
def test_dispatch_threshold_boundary(n_rows, expect_numba, restore_threshold, monkeypatch):
    """Dispatch threshold boundary."""
    pytest.importorskip("numba")
    rc = restore_threshold
    rc.set_spearmanr_dispatch_threshold(100)

    calls = {"n": 0}
    orig = rc.spearmanr_batched_numba

    def spy(X, Y):
        """Spy."""
        calls["n"] += 1
        return orig(X, Y)

    monkeypatch.setattr(rc, "spearmanr_batched_numba", spy)

    rng = np.random.default_rng(n_rows)
    X = rng.normal(size=(n_rows, 11))
    Y = rng.normal(size=(n_rows, 11))
    got = rc.spearmanr_batched_dispatch(X, Y)

    # Whichever path ran, the answer must equal the numpy reference.
    ref = rc.spearmanr_batched(X, Y)
    np.testing.assert_allclose(got, ref, atol=1e-10, equal_nan=True)

    if expect_numba:
        assert calls["n"] == 1, f"n={n_rows}: expected numba path (>= threshold)"
    else:
        assert calls["n"] == 0, f"n={n_rows}: expected numpy path (< threshold)"


def test_dispatch_1d_input_uses_numpy_path_regardless_of_size(restore_threshold, monkeypatch):
    """Dispatch 1d input uses numpy path regardless of size."""
    pytest.importorskip("numba")
    rc = restore_threshold
    rc.set_spearmanr_dispatch_threshold(1)  # below any plausible size

    calls = {"n": 0}
    orig = rc.spearmanr_batched_numba

    def spy(X, Y):
        """Spy."""
        calls["n"] += 1
        return orig(X, Y)

    monkeypatch.setattr(rc, "spearmanr_batched_numba", spy)
    # A 1-D input has ndim != 2 -> dispatcher must route to numpy (which then raises), never numba.
    with pytest.raises(ValueError):
        rc.spearmanr_batched_dispatch(np.arange(500.0), np.arange(500.0))
    assert calls["n"] == 0


@pytest.mark.parametrize("n", [0, 1])
def test_scalar_dispatch_degenerate_length_is_nan(n):
    """Scalar dispatch degenerate length is nan."""
    from mlframe.metrics.rank_correlation import spearmanr_scalar_dispatch

    x = np.arange(float(n))
    y = np.arange(float(n))
    assert np.isnan(spearmanr_scalar_dispatch(x, y))


@pytest.mark.parametrize(
    "y,expected",
    [
        (np.array([1.0, 2.0]), 1.0),  # perfectly co-monotone
        (np.array([2.0, 1.0]), -1.0),  # perfectly anti-monotone
    ],
)
def test_scalar_dispatch_minimal_length_two(y, expected):
    """Scalar dispatch minimal length two."""
    from mlframe.metrics.rank_correlation import spearmanr_scalar_dispatch

    x = np.array([1.0, 2.0])
    assert spearmanr_scalar_dispatch(x, y) == pytest.approx(expected, abs=1e-9)
