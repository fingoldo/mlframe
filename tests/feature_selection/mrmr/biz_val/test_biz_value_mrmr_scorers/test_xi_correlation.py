"""Layer 72 biz_value: Chatterjee's Xi rank correlation for the auto-scorer pool
(mrmr_audit_2026-07-20 fe_expansion.md "Chatterjee's Xi rank correlation scorer").

Validates ``xi_correlation`` / ``xi_correlation_batch`` (``_orthogonal_xi_fe``) and its wiring into
the Layer 68 auto-scorer pool (``_orth_auto_scorer_fe.SCORER_NAMES`` / ``_score_xi``) and the
Layer 69 ensemble rank-fusion dispatch (``_orthogonal_scorer_auto_fe``).

Contracts pinned
-----------------
* ``TestXiMeasurableFunction``: a noiseless measurable function (y = x^2) gives xi close to 1.
* ``TestXiIndependence``: independent noise pairs give xi near 0 (within small-sample tail).
* ``TestXiBatchMatchesScalar``: the batched multi-column path matches the per-column scalar path.
* ``TestXiCatchesHighFrequencyOscillation`` (biz_value): on ``y = sin(20*x) + noise``, xi recovers
  materially more signal than plug-in MI at a realistic bin count -- the exact shape the audit's
  own rationale names as the gap xi closes that no existing scorer catches.
* ``TestScorerPoolWiring``: "xi" is in ``SCORER_NAMES`` and dispatches without raising.
* Degenerate inputs (constant y, too-few rows, NaN) return 0.0, never raise.
"""

from __future__ import annotations

import numpy as np
import pytest

SEEDS = (1, 7, 13, 42, 101)


def _import_xi():
    """Lazily import the Layer-72 Xi correlation primitives."""
    from mlframe.feature_selection.filters._orthogonal_xi_fe import xi_correlation, xi_correlation_batch

    return xi_correlation, xi_correlation_batch


class TestXiMeasurableFunction:
    """A noiseless measurable function must give xi close to its asymptotic ceiling of 1."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_quadratic_function_gives_high_xi(self, seed):
        """y = x^2 (non-monotone but measurable) must give xi close to 1 across seeds."""
        xi_correlation, _ = _import_xi()
        rng = np.random.default_rng(seed)
        n = 3000
        x = rng.uniform(-3, 3, n)
        y = x**2  # noiseless measurable function, non-monotone
        xi = xi_correlation(x, y, random_state=seed)
        assert xi > 0.85, f"seed={seed}: noiseless measurable function should give xi close to 1, got {xi:.4f}"


class TestXiIndependence:
    """Independent noise pairs must give xi near the null floor."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_independent_noise_gives_low_xi(self, seed):
        """Two independent gaussian arrays must give xi near 0 across seeds."""
        xi_correlation, _ = _import_xi()
        rng = np.random.default_rng(seed)
        n = 3000
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        xi = xi_correlation(x, y, random_state=seed)
        assert abs(xi) < 0.1, f"seed={seed}: independent pair should give xi near 0, got {xi:.4f}"


class TestXiBatchMatchesScalar:
    """The batched multi-column path's FIRST column must match the scalar path called with the
    same seed (both start from a freshly-seeded rng before any permutation draw); the batch must
    also be deterministic across repeated calls with the same random_state."""

    def test_batch_first_column_matches_scalar(self):
        """The batch path's first column must equal the scalar path called with the same seed."""
        xi_correlation, xi_correlation_batch = _import_xi()
        rng = np.random.default_rng(0)
        n = 1000
        x1 = rng.standard_normal(n)
        x2 = rng.uniform(-2, 2, n)
        y = np.sin(x2) + 0.1 * rng.standard_normal(n)
        X = np.column_stack([x1, x2])

        batch = xi_correlation_batch(X, y, random_state=7)
        scalar_col0 = xi_correlation(X[:, 0], y, random_state=7)
        assert batch[0] == pytest.approx(scalar_col0)

    def test_batch_is_deterministic_across_repeated_calls(self):
        """Two calls with the same random_state must return bit-identical arrays."""
        _xi_correlation, xi_correlation_batch = _import_xi()
        rng = np.random.default_rng(1)
        n = 800
        X = rng.standard_normal((n, 3))
        y = np.sin(X[:, 1]) + 0.1 * rng.standard_normal(n)
        batch = xi_correlation_batch(X, y, random_state=7)
        batch_repeat = xi_correlation_batch(X, y, random_state=7)
        np.testing.assert_array_equal(batch, batch_repeat)
        assert batch.shape == (3,)
        assert np.isfinite(batch).all()


class TestXiCatchesHighFrequencyOscillation:
    """biz_value: on a smooth-but-highly-oscillatory target, xi recovers materially more signal
    than plug-in MI at a realistic bin count -- the exact gap the audit names."""

    def test_xi_beats_plugin_mi_on_oscillatory_signal(self):
        """On a high-frequency oscillatory target, xi must recover >3x the plug-in MI signal."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

        xi_correlation, _ = _import_xi()
        rng = np.random.default_rng(3)
        n = 5000
        x = rng.uniform(0, 1, n)
        # 200 full cycles over [0, 1] with only nbins=20 quantile bins: each bin averages ~10
        # oscillation cycles, so plug-in MI's fixed quantile binning must collapse toward the null
        # floor even though the underlying x->y relationship is a perfectly deterministic function.
        y_cont = np.sin(200 * np.pi * x) + 0.05 * rng.standard_normal(n)
        y_binned = np.digitize(y_cont, np.quantile(y_cont, np.linspace(0, 1, 5)[1:-1])).astype(np.int64)

        xi = xi_correlation(x, y_cont, random_state=3)
        mi = float(_mi_classif_batch(x.reshape(-1, 1), y_binned, nbins=20)[0])

        assert xi > 0.5, f"xi should recover strong signal on the oscillatory target, got {xi:.4f}"
        assert xi > 3.0 * max(mi, 1e-6), f"xi ({xi:.4f}) should materially exceed plug-in MI ({mi:.4f}) on this high-frequency oscillatory shape"


class TestScorerPoolWiring:
    """xi must be a first-class member of the Layer 68/69 scorer pool, dispatchable without raising."""

    def test_xi_in_scorer_names(self):
        """Layer 68's SCORER_NAMES tuple must include 'xi'."""
        from mlframe.feature_selection.filters._orth_auto_scorer_fe import SCORER_NAMES

        assert "xi" in SCORER_NAMES, f"SCORER_NAMES missing 'xi': {SCORER_NAMES}"

    def test_score_xi_dispatches_without_raising(self):
        """The _score_xi wrapper must return a finite value without raising."""
        from mlframe.feature_selection.filters._orth_auto_scorer_fe import _score_xi

        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        y = np.sin(x) + 0.1 * rng.standard_normal(500)
        val = _score_xi(x, y, random_state=0)
        assert np.isfinite(val)


class TestDegenerateInputsReturnZero:
    """Constant y, too-few rows, and non-finite input must all return 0.0, never raise."""

    def test_constant_y_returns_zero(self):
        """A zero-variance y has nothing to detect regardless of x."""
        xi_correlation, _ = _import_xi()
        x = np.arange(100, dtype=np.float64)
        y = np.full(100, 5.0)
        assert xi_correlation(x, y) == 0.0

    def test_single_row_returns_zero(self):
        """n=1 hits the explicit n<2 early-return guard."""
        xi_correlation, _ = _import_xi()
        assert xi_correlation(np.array([1.0]), np.array([2.0])) == 0.0

    def test_nan_in_x_returns_zero(self):
        """A NaN in x hits the explicit finite-check guard rather than propagating into the sort."""
        xi_correlation, _ = _import_xi()
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert xi_correlation(x, y) == 0.0

    def test_batch_empty_columns_returns_empty(self):
        """K=0 columns hits the explicit early-return, not a zero-size loop crash."""
        _, xi_correlation_batch = _import_xi()
        out = xi_correlation_batch(np.empty((10, 0)), np.arange(10, dtype=np.float64))
        assert out.shape == (0,)
