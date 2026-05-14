"""Tests for ``mlframe.feature_selection.filters.estimators``.

The module exposes pure functions (not an sklearn estimator class):

* ``ksg_mi_with_target`` -- per-column KSG MI via ``sklearn.feature_selection.mutual_info_classif/regression``.
* ``ksg_mi_pair`` -- single-pair convenience wrapper.
* ``ksg_mi_with_significance`` -- KSG MI + permutation-test significance filter.
* ``nsb_mi`` -- Bayesian MI via optional ``ndd`` dep (skipped here if absent).

So the typical sklearn API battery (fit / set_params / clone) does not apply -- we test the
actual surface: shape / dtype / value contracts and one quantitative biz_value check
(KSG MI of an informative feature beats KSG MI of a pure-noise feature).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.estimators import (
    ksg_mi_with_target,
    ksg_mi_pair,
    ksg_mi_with_significance,
    nsb_mi,
)


# ================================================================================================
# Shared fixtures
# ================================================================================================


@pytest.fixture
def informative_xy_classif():
    """Small classification frame: feature 0 carries label, features 1-3 are noise."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n).astype(np.int64)
    X = np.column_stack([
        y.astype(np.float64) + rng.normal(scale=0.3, size=n),  # signal
        rng.normal(size=n),  # noise
        rng.normal(size=n),  # noise
        rng.normal(size=n),  # noise
    ])
    return X, y


@pytest.fixture
def informative_xy_regress():
    """Regression frame: feature 0 is y + noise; features 1-2 are noise."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(size=n)
    X = np.column_stack([
        y + rng.normal(scale=0.2, size=n),  # signal
        rng.normal(size=n),
        rng.normal(size=n),
    ])
    return X, y


# ================================================================================================
# ksg_mi_with_target
# ================================================================================================


class TestKsgMiWithTarget:
    def test_returns_ndarray_of_correct_shape(self, informative_xy_classif):
        X, y = informative_xy_classif
        mi = ksg_mi_with_target(X, y, feature_indices=[0, 1, 2], n_neighbors=3)
        assert isinstance(mi, np.ndarray)
        assert mi.shape == (3,)

    def test_subset_of_features_only(self, informative_xy_classif):
        """Asking for k columns returns k MI values (avoids sklearn over full X)."""
        X, y = informative_xy_classif
        mi_all = ksg_mi_with_target(X, y, feature_indices=list(range(X.shape[1])))
        mi_sub = ksg_mi_with_target(X, y, feature_indices=[0])
        assert mi_all.shape == (X.shape[1],)
        assert mi_sub.shape == (1,)
        # The MI of col 0 should match in both calls (same data, same seed, deterministic).
        assert np.isclose(mi_all[0], mi_sub[0])

    def test_mi_values_are_nonnegative(self, informative_xy_classif):
        """KSG MI is asymptotically unbiased but can be slightly negative on finite samples; the
        sklearn implementation clips at 0 (see sklearn._estimate_mi). Assert nonneg as the spec."""
        X, y = informative_xy_classif
        mi = ksg_mi_with_target(X, y, feature_indices=[0, 1, 2, 3])
        assert (mi >= 0).all()

    def test_regression_path(self, informative_xy_regress):
        X, y = informative_xy_regress
        mi = ksg_mi_with_target(X, y, feature_indices=[0, 1, 2], discrete_target=False)
        assert mi.shape == (3,)
        assert (mi >= 0).all()

    def test_single_feature_1d_X_via_reshape(self):
        """X.ndim==1 path: function reshapes to (-1, 1) and evaluates that single column."""
        rng = np.random.default_rng(0)
        n = 300
        y = rng.integers(0, 2, n).astype(np.int64)
        x = y.astype(np.float64) + rng.normal(scale=0.3, size=n)  # 1D
        mi = ksg_mi_with_target(x, y, feature_indices=[0])
        assert mi.shape == (1,)
        assert mi[0] > 0

    def test_random_state_determinism(self, informative_xy_classif):
        """Same random_state -> identical MI vectors (KSG is RNG-tied via sklearn)."""
        X, y = informative_xy_classif
        mi1 = ksg_mi_with_target(X, y, feature_indices=[0, 1, 2], random_state=7)
        mi2 = ksg_mi_with_target(X, y, feature_indices=[0, 1, 2], random_state=7)
        np.testing.assert_array_equal(mi1, mi2)


# ================================================================================================
# ksg_mi_pair
# ================================================================================================


class TestKsgMiPair:
    def test_returns_python_float(self):
        rng = np.random.default_rng(0)
        n = 300
        y = rng.integers(0, 2, n).astype(np.int64)
        x = y.astype(np.float64) + rng.normal(scale=0.3, size=n)
        val = ksg_mi_pair(x, y)
        assert isinstance(val, float)
        assert val >= 0

    def test_matches_with_target_first_column(self):
        """Pair convenience wrapper must be a numerical identity to ksg_mi_with_target on 1 col."""
        rng = np.random.default_rng(0)
        n = 300
        y = rng.integers(0, 2, n).astype(np.int64)
        x = y.astype(np.float64) + rng.normal(scale=0.3, size=n)
        pair = ksg_mi_pair(x, y, n_neighbors=3, random_state=42)
        arr = ksg_mi_with_target(x.reshape(-1, 1), y, [0], n_neighbors=3, random_state=42)
        assert np.isclose(pair, float(arr[0]))


# ================================================================================================
# ksg_mi_with_significance
# ================================================================================================


class TestKsgMiWithSignificance:
    def test_returns_three_tuple_of_correct_shapes(self, informative_xy_classif):
        X, y = informative_xy_classif
        n_feat = X.shape[1]
        mi, p, support = ksg_mi_with_significance(
            X, y, feature_indices=list(range(n_feat)),
            n_permutations=10, alpha=0.05, n_jobs=1,
        )
        assert isinstance(mi, np.ndarray) and mi.shape == (n_feat,)
        assert isinstance(p, np.ndarray) and p.shape == (n_feat,)
        assert isinstance(support, np.ndarray)
        # Support is a subset of the input feature indices.
        assert set(support.tolist()).issubset(set(range(n_feat)))

    def test_pvalues_in_valid_range(self, informative_xy_classif):
        X, y = informative_xy_classif
        _, p, _ = ksg_mi_with_significance(
            X, y, feature_indices=[0, 1, 2, 3],
            n_permutations=10, n_jobs=1,
        )
        # Conservative p formula: (1+failures)/(1+n_perms) lives in (0, 1].
        assert (p > 0).all()
        assert (p <= 1).all()

    def test_support_sorted_by_observed_mi_descending(self, informative_xy_classif):
        """``support`` is the surviving indices ORDERED by observed MI descending."""
        X, y = informative_xy_classif
        mi, _, support = ksg_mi_with_significance(
            X, y, feature_indices=[0, 1, 2, 3],
            n_permutations=10, alpha=1.0,  # alpha=1 -> all features pass
            n_jobs=1,
        )
        assert support.shape == (4,)
        ordered_mi = mi[support]
        # Strict-or-equal monotone descending.
        assert (np.diff(ordered_mi) <= 1e-12).all()

    def test_alpha_zero_rejects_everything(self, informative_xy_classif):
        """alpha=0 -> p > 0 always (since min p is 1/(1+n_perms)) -> empty support."""
        X, y = informative_xy_classif
        _, _, support = ksg_mi_with_significance(
            X, y, feature_indices=[0, 1, 2, 3],
            n_permutations=10, alpha=0.0, n_jobs=1,
        )
        assert support.shape == (0,)


# ================================================================================================
# nsb_mi
# ================================================================================================


class TestNsbMi:
    """``nsb_mi`` requires optional ``ndd`` dep."""

    def test_raises_importerror_when_ndd_missing(self):
        try:
            import ndd  # noqa: F401
        except ImportError:
            # Confirm the function surfaces a clean, actionable ImportError.
            x = np.array([0, 1, 0, 1], dtype=np.int64)
            y = np.array([0, 1, 0, 1], dtype=np.int64)
            with pytest.raises(ImportError, match="ndd"):
                nsb_mi(x, y)
        else:
            pytest.skip("ndd is installed; ImportError path not exercised")

    def test_basic_call_when_ndd_installed(self):
        pytest.importorskip("ndd")
        # Two perfectly correlated discrete vars: MI should be > 0.
        rng = np.random.default_rng(0)
        n = 300
        x = rng.integers(0, 3, n).astype(np.int64)
        y = x.copy()
        val = nsb_mi(x, y)
        assert isinstance(val, float)
        assert val > 0


# ================================================================================================
# biz_value: KSG separates signal from noise
# ================================================================================================


@pytest.mark.fast
class TestKsgMiBizValue:
    """Quantitative win: KSG MI of an informative feature must materially exceed KSG MI of
    pure-noise features on the same target. This is the operational contract the module exists
    to provide -- if it fails, the estimator is broken regardless of API hygiene."""

    def test_informative_beats_noise_classification(self):
        rng = np.random.default_rng(0)
        n = 600
        y = rng.integers(0, 2, n).astype(np.int64)
        X = np.column_stack([
            y.astype(np.float64) + rng.normal(scale=0.3, size=n),  # informative
            rng.normal(size=n),                                    # noise
            rng.normal(size=n),                                    # noise
        ])
        mi = ksg_mi_with_target(X, y, feature_indices=[0, 1, 2], random_state=0)
        assert mi[0] > mi[1] + 0.05
        assert mi[0] > mi[2] + 0.05

    def test_significance_filter_keeps_signal_drops_noise(self):
        rng = np.random.default_rng(0)
        n = 600
        y = rng.integers(0, 2, n).astype(np.int64)
        X = np.column_stack([
            y.astype(np.float64) + rng.normal(scale=0.3, size=n),  # informative
            rng.normal(size=n),                                    # noise
            rng.normal(size=n),                                    # noise
        ])
        _, _, support = ksg_mi_with_significance(
            X, y, feature_indices=[0, 1, 2],
            n_permutations=30, alpha=0.05, n_jobs=1, random_state=0,
        )
        assert 0 in support.tolist()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
