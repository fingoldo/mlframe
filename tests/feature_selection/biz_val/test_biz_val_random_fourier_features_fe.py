"""biz_value + unit tests for Random Fourier Features (mrmr_audit_2026-07-20 fe_expansion.md
"Random Fourier Features (random kitchen sinks) multi-column kernel-approximation block").

Validates ``random_fourier_features`` (``_random_fourier_features_fe``): a joint multi-column
kernel-approximation expansion, distinct from every existing PER-COLUMN or PRODUCT-of-per-column-
legs basis family in the catalog.

Contracts pinned
-----------------
* ``TestShapeAndDeterminism``: output shape is (n, m); the SAME random_state reproduces
  bit-identical features (load-bearing for a fit/transform replay contract).
* ``TestBizValueRadialGaussianBump`` (biz_value): on y = exp(-||x||^2/2) over p=10 jointly-
  informative columns, a linear model on the RFF-expanded features recovers the target far better
  than a linear model on the raw columns alone -- no pairwise/product-of-bases term can express a
  genuinely 10-way radial structure without combinatorial blow-up.
* Degenerate inputs (empty X, m=0, non-finite X) return an (n, 0) array, never raise.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._random_fourier_features_fe import random_fourier_features


class TestShapeAndDeterminism:
    """Output shape must match (n, m); the same seed must reproduce bit-identical features."""

    def test_output_shape(self):
        """A (n, p) input with m features requested must produce a (n, m) output."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 5))
        Z = random_fourier_features(X, m=32, random_state=0)
        assert Z.shape == (500, 32)

    def test_deterministic_across_repeated_calls(self):
        """Two calls with the same random_state must be bit-identical -- required for a
        fit-once/transform-many-times replay contract."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((300, 4))
        Z1 = random_fourier_features(X, m=16, random_state=7)
        Z2 = random_fourier_features(X, m=16, random_state=7)
        np.testing.assert_array_equal(Z1, Z2)

    def test_different_seeds_give_different_features(self):
        """Different random_state values must draw a genuinely different projection."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((300, 4))
        Z1 = random_fourier_features(X, m=16, random_state=1)
        Z2 = random_fourier_features(X, m=16, random_state=2)
        assert not np.allclose(Z1, Z2)

    def test_1d_input_treated_as_single_column(self):
        """A 1-D input array must be reshaped to (n, 1) rather than raising."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(200)
        Z = random_fourier_features(x, m=8, random_state=0)
        assert Z.shape == (200, 8)

    def test_output_bounded_by_amplitude(self):
        """Every RFF value must lie in [-sqrt(2/m), sqrt(2/m)] (cosine amplitude bound)."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((400, 3))
        m = 20
        Z = random_fourier_features(X, m=m, random_state=0)
        amp = np.sqrt(2.0 / m)
        assert np.all(np.abs(Z) <= amp + 1e-9)


class TestBizValueRadialGaussianBump:
    """biz_value: RFF recovers a genuinely joint 10-way radial structure that no per-column or
    pairwise/product-of-bases term can express without combinatorial blow-up."""

    def test_rff_recovers_radial_target_linear_raw_columns_cannot(self):
        """A linear model on RFF-expanded features must materially beat a linear model on the raw
        columns alone at recovering y = exp(-||x||^2/2) over p=10 columns."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split

        rng = np.random.default_rng(0)
        n, p = 4000, 10
        X = rng.standard_normal((n, p))
        y = np.exp(-0.5 * np.sum(X * X, axis=1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        raw_model = Ridge(alpha=1.0).fit(X_train, y_train)
        raw_r2 = raw_model.score(X_test, y_test)

        Z_train = random_fourier_features(X_train, m=200, bandwidth=1.5, random_state=0)
        Z_test = random_fourier_features(X_test, m=200, bandwidth=1.5, random_state=0)
        rff_model = Ridge(alpha=1.0).fit(Z_train, y_train)
        rff_r2 = rff_model.score(Z_test, y_test)

        assert rff_r2 > 0.5, f"RFF-expanded linear model should recover the radial target well, got R^2={rff_r2:.4f}"
        assert rff_r2 > raw_r2 + 0.3, f"RFF R^2 ({rff_r2:.4f}) should materially exceed raw-column linear R^2 ({raw_r2:.4f}) on this genuinely joint radial structure"


class TestDegenerateInputsReturnEmpty:
    """Empty X, m=0, and non-finite X must all return an (n, 0) array, never raise."""

    def test_m_zero_returns_empty_columns(self):
        """m=0 requests zero features -- must return (n, 0), not raise."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 3))
        Z = random_fourier_features(X, m=0, random_state=0)
        assert Z.shape == (100, 0)

    def test_empty_rows_returns_empty(self):
        """Zero rows must return an (0, 0) array rather than raising on an empty matmul."""
        X = np.empty((0, 3))
        Z = random_fourier_features(X, m=8, random_state=0)
        assert Z.shape == (0, 0)

    def test_nan_input_returns_empty(self):
        """A NaN anywhere in X must return an (n, 0) array rather than propagating NaN features."""
        X = np.array([[1.0, np.nan], [2.0, 3.0]])
        Z = random_fourier_features(X, m=8, random_state=0)
        assert Z.shape == (2, 0)
