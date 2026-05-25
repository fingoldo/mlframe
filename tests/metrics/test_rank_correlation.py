"""Tests for ``mlframe.metrics.rank_correlation``: batched-Spearman
correctness vs scipy.stats.spearmanr, NaN handling, ties, and
numpy-vs-numba equivalence."""
from __future__ import annotations

import numpy as np
import pytest

# Deterministic batched-Spearman correctness on synthetic n<=2000 rows; numba-prewarmed via session fixture.
pytestmark = [pytest.mark.fast]


def _scipy_per_row(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Reference: scipy.stats.spearmanr per row (the slow loop the
    batched API replaces)."""
    from scipy.stats import spearmanr
    n = X.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        a = X[i]
        b = Y[i]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        try:
            rho, _ = spearmanr(a, b)
        except Exception:
            continue
        if np.isfinite(rho):
            out[i] = rho
    return out


class TestSpearmanrBatchedNumpy:
    def test_random_pairs_match_scipy(self) -> None:
        from mlframe.metrics.rank_correlation import spearmanr_batched
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 31))
        Y = rng.normal(0, 1, (200, 31))
        ref = _scipy_per_row(X, Y)
        got = spearmanr_batched(X, Y)
        np.testing.assert_allclose(got, ref, atol=1e-10, equal_nan=True)

    def test_correlated_pairs_close_to_1(self) -> None:
        from mlframe.metrics.rank_correlation import spearmanr_batched
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 31))
        Y = X + 0.01 * rng.normal(0, 1, X.shape)
        got = spearmanr_batched(X, Y)
        assert (got > 0.95).all(), got

    def test_anti_correlated_close_to_neg1(self) -> None:
        from mlframe.metrics.rank_correlation import spearmanr_batched
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (50, 31))
        Y = -X + 0.01 * rng.normal(0, 1, X.shape)
        got = spearmanr_batched(X, Y)
        assert (got < -0.95).all(), got

    def test_ties_handled_via_average_rank(self) -> None:
        """Average-rank ties: ``spearmanr_batched`` must match scipy
        exactly on data with ties."""
        from mlframe.metrics.rank_correlation import spearmanr_batched
        X = np.array(
            [
                [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0],
                [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
            ]
        )
        Y = np.array(
            [
                [2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0],
            ]
        )
        ref = _scipy_per_row(X, Y)
        got = spearmanr_batched(X, Y)
        np.testing.assert_allclose(got, ref, atol=1e-10, equal_nan=True)

    def test_nan_row_yields_nan(self) -> None:
        from mlframe.metrics.rank_correlation import spearmanr_batched
        X = np.array([[1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 4.0]])
        Y = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        got = spearmanr_batched(X, Y)
        assert np.isnan(got[0])
        assert got[1] == pytest.approx(1.0)

    def test_constant_row_yields_nan(self) -> None:
        from mlframe.metrics.rank_correlation import spearmanr_batched
        X = np.array([[1.0, 1.0, 1.0, 1.0]])
        Y = np.array([[1.0, 2.0, 3.0, 4.0]])
        got = spearmanr_batched(X, Y)
        assert np.isnan(got[0])

    def test_shape_mismatch_raises(self) -> None:
        from mlframe.metrics.rank_correlation import spearmanr_batched
        X = np.ones((5, 10))
        Y = np.ones((5, 9))
        with pytest.raises(ValueError):
            spearmanr_batched(X, Y)

    def test_1d_input_raises(self) -> None:
        from mlframe.metrics.rank_correlation import spearmanr_batched
        X = np.arange(10.0)
        Y = np.arange(10.0)
        with pytest.raises(ValueError):
            spearmanr_batched(X, Y)


class TestSpearmanrBatchedNumba:
    def test_numpy_and_numba_agree(self) -> None:
        pytest.importorskip("numba")
        from mlframe.metrics.rank_correlation import (
            spearmanr_batched, spearmanr_batched_numba,
        )
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, (500, 31))
        Y = rng.normal(0, 1, (500, 31))
        got_np = spearmanr_batched(X, Y)
        got_nb = spearmanr_batched_numba(X, Y)
        np.testing.assert_allclose(got_np, got_nb, atol=1e-10, equal_nan=True)

    def test_numba_handles_ties_like_scipy(self) -> None:
        pytest.importorskip("numba")
        from mlframe.metrics.rank_correlation import spearmanr_batched_numba
        X = np.array(
            [
                [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0],
                [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
            ]
        )
        Y = np.array(
            [
                [2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0],
            ]
        )
        ref = _scipy_per_row(X, Y)
        got = spearmanr_batched_numba(X, Y)
        np.testing.assert_allclose(got, ref, atol=1e-10, equal_nan=True)

    def test_numba_nan_propagation(self) -> None:
        pytest.importorskip("numba")
        from mlframe.metrics.rank_correlation import spearmanr_batched_numba
        X = np.array([[1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 4.0]])
        Y = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        got = spearmanr_batched_numba(X, Y)
        assert np.isnan(got[0])
        assert got[1] == pytest.approx(1.0)


class TestDispatcher:
    def test_small_n_uses_numpy(self) -> None:
        """Below the threshold dispatch returns the numpy path -- check
        by running on tiny input and comparing to numpy."""
        from mlframe.metrics.rank_correlation import (
            spearmanr_batched, spearmanr_batched_dispatch,
            set_spearmanr_dispatch_threshold,
        )
        set_spearmanr_dispatch_threshold(5_000)
        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, (100, 21))
        Y = rng.normal(0, 1, (100, 21))
        got_disp = spearmanr_batched_dispatch(X, Y)
        got_np = spearmanr_batched(X, Y)
        np.testing.assert_allclose(got_disp, got_np, atol=1e-12, equal_nan=True)

    def test_large_n_uses_numba_when_available(self) -> None:
        pytest.importorskip("numba")
        from mlframe.metrics.rank_correlation import (
            spearmanr_batched, spearmanr_batched_dispatch,
            set_spearmanr_dispatch_threshold,
        )
        set_spearmanr_dispatch_threshold(100)
        rng = np.random.default_rng(13)
        X = rng.normal(0, 1, (500, 21))
        Y = rng.normal(0, 1, (500, 21))
        got_disp = spearmanr_batched_dispatch(X, Y)
        got_np = spearmanr_batched(X, Y)
        # Same numerical answer regardless of path.
        np.testing.assert_allclose(got_disp, got_np, atol=1e-10, equal_nan=True)
