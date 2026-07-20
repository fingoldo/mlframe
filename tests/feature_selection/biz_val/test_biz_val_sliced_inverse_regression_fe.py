"""biz_value + unit tests for Sliced Inverse Regression (SIR) (mrmr_audit_2026-07-20
fe_expansion.md "Sliced Inverse Regression (SIR) oblique-direction projection feature").

Validates ``sir_direction_features`` (``_sliced_inverse_regression_fe``): recovers the LINEAR
COMBINATION direction along which y varies most, catching an oblique (rotated) threshold spread
thinly across correlated columns that no per-column marginal MI or product-of-bases family reaches.

Contracts pinned
-----------------
* ``TestShapeAndDeterminism``: output shape is (n, n_directions); byte-identical across repeated
  calls on the same data (no hidden randomness).
* ``TestBizValueObliqueThreshold`` (biz_value): on y = 1{0.6*x1+0.5*x2+0.4*x3+0.3*x4+0.4*x5 > c},
  the top SIR direction correlates strongly with the true oblique combination w.x, while every
  individual column's marginal correlation with y is weak.
* Degenerate inputs (n<2, single slice / constant y, singular Sigma, non-finite input) return an
  (n, 0) array, never raise.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._sliced_inverse_regression_fe import sir_direction_features


class TestShapeAndDeterminism:
    """Output shape must match (n, n_directions); the computation is fully deterministic (no rng)."""

    def test_output_shape(self):
        """A (n, p) input with n_directions requested must produce a (n, n_directions) output."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 6))
        y = X[:, 0] + 0.1 * rng.standard_normal(2000)
        Z = sir_direction_features(X, y, n_slices=10, n_directions=2)
        assert Z.shape == (2000, 2)

    def test_deterministic_across_repeated_calls(self):
        """Two calls on the same data must be bit-identical (no internal randomness to seed)."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((1500, 5))
        y = X[:, 0] * 0.5 + X[:, 1] * 0.5 + 0.1 * rng.standard_normal(1500)
        Z1 = sir_direction_features(X, y, n_slices=8, n_directions=1)
        Z2 = sir_direction_features(X, y, n_slices=8, n_directions=1)
        np.testing.assert_array_equal(Z1, Z2)


class TestBizValueObliqueThreshold:
    """biz_value: SIR recovers a genuinely oblique linear combination spread thinly across
    correlated columns that no per-column marginal MI reaches."""

    def test_top_direction_correlates_with_true_oblique_combination(self):
        """The top SIR projection must correlate strongly with the TRUE oblique combination w.x,
        while every individual column's marginal |corr| with y is comparatively weak."""
        rng = np.random.default_rng(0)
        n = 8000
        # Independent columns, per the audit's own scenario: y depends on an OBLIQUE combination
        # spread thinly across 5 columns, none individually dominant -- exactly where no single
        # column's marginal correlation clears a useful floor but the combined direction does.
        w = np.array([0.6, 0.5, 0.4, 0.3, 0.4])
        X = rng.standard_normal((n, 5))
        combo = X @ w
        y = (combo > np.median(combo)).astype(float)

        Z = sir_direction_features(X, y, n_slices=10, n_directions=1)
        assert Z.shape == (n, 1)

        sir_corr = abs(float(np.corrcoef(Z[:, 0], combo)[0, 1]))
        marginal_corrs = [abs(float(np.corrcoef(X[:, j], y)[0, 1])) for j in range(5)]

        assert sir_corr > 0.8, f"top SIR direction should correlate strongly with the true oblique combination, got {sir_corr:.4f}"
        assert sir_corr > 2.0 * max(marginal_corrs), f"SIR direction corr ({sir_corr:.4f}) should materially exceed every column's marginal |corr| with y ({marginal_corrs})"


class TestDegenerateInputsReturnEmpty:
    """n<2, constant y (single slice), singular Sigma, and non-finite input must return (n, 0)."""

    def test_single_row_returns_empty(self):
        """n=1 hits the explicit n<2 early-return guard."""
        Z = sir_direction_features(np.array([[1.0, 2.0]]), np.array([1.0]))
        assert Z.shape == (1, 0)

    def test_constant_y_returns_empty(self):
        """A constant target has only one slice -- nothing to compute a between-slice covariance from."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((500, 3))
        y = np.full(500, 5.0)
        Z = sir_direction_features(X, y, n_slices=10)
        assert Z.shape == (500, 0)

    def test_singular_sigma_returns_empty_or_finite(self):
        """A constant (zero-variance) X column makes Sigma singular; must not raise -- either
        degrades gracefully to empty or (after ridge stabilization) returns a finite result."""
        rng = np.random.default_rng(3)
        n = 500
        X = np.column_stack([np.full(n, 1.0), rng.standard_normal(n)])
        y = X[:, 1] + 0.1 * rng.standard_normal(n)
        Z = sir_direction_features(X, y, n_slices=8, n_directions=1)
        assert Z.shape[0] == n
        assert Z.shape[1] in (0, 1)
        if Z.shape[1] == 1:
            assert np.isfinite(Z).all()

    def test_nan_input_returns_empty(self):
        """A NaN anywhere in X must return an (n, 0) array rather than propagating NaN."""
        X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
        y = np.array([1.0, 2.0, 3.0])
        Z = sir_direction_features(X, y)
        assert Z.shape == (3, 0)

    def test_n_directions_zero_returns_empty(self):
        """n_directions < 1 hits the explicit early-return guard."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((200, 3))
        y = rng.standard_normal(200)
        Z = sir_direction_features(X, y, n_directions=0)
        assert Z.shape == (200, 0)
