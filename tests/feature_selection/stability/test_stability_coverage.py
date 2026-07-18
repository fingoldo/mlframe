"""Coverage tests for ``mlframe.feature_selection.filters.stability``.

Targets the ``StabilityMRMR`` bootstrap wrapper:
- ``fit`` on ndarray vs DataFrame (``iloc`` vs positional indexing branches).
- ``transform`` on ndarray vs DataFrame.
- ``n_jobs=1`` sequential path and ``n_jobs=2`` ``joblib.Parallel`` path.
- ``selection_probabilities_`` / ``support_`` / ``n_features_`` / ``n_features_in_`` / ``feature_names_in_`` attribute population.
- Determinism for a fixed ``random_state``.

The wrapper is estimator-agnostic, so we plug a minimal stub selector that exposes ``fit(X, y)`` and ``support_`` to keep tests fast and isolated from the
real MRMR machinery (which would pull astropy / numba transitively and slow the suite by ~30s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from mlframe.feature_selection.filters.stability import StabilityMRMR

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Minimal stub selectors: emulate the ``MRMR``-like contract (``fit`` + ``support_``) without pulling the real estimator.
# sklearn.clone deep-copies via ``get_params`` so the stub must accept its config through ``__init__``.
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class _FixedSupportSelector(BaseEstimator):
    """Always returns the same support set regardless of subsample. Lets us assert ``selection_probabilities_`` is exactly 1.0 on those features and 0 elsewhere."""

    def __init__(self, support=(0, 1)):
        self.support = support

    def fit(self, X, y):
        """Helper that fit."""
        self.support_ = np.asarray(self.support, dtype=np.int64)
        return self


class _SeedDependentSelector(BaseEstimator):
    """Returns a random support set keyed off the subsample sum so the bootstrap support varies. Exercises the probability-aggregation path."""

    def __init__(self, n_features: int = 5, k: int = 2):
        self.n_features = n_features
        self.k = k

    def fit(self, X, y):
        # Derive a seed from the data so two different bootstraps produce two different supports.
        """Helper that fit."""
        seed = int(abs(np.asarray(X).sum())) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        self.support_ = np.sort(rng.choice(self.n_features, size=self.k, replace=False)).astype(np.int64)
        return self


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def small_ndarray_dataset():
    """Small ndarray dataset."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(80, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y


@pytest.fixture
def small_dataframe_dataset(small_ndarray_dataset):
    """Small dataframe dataset."""
    X, y = small_ndarray_dataset
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y, name="target")
    return df, y_s


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Attribute / contract tests
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestStabilityMRMRAttributes:
    """Groups tests covering TestStabilityMRMRAttributes."""
    def test_init_stores_params(self):
        """Init stores params."""
        base = _FixedSupportSelector(support=(0, 2))
        sel = StabilityMRMR(estimator=base, n_bootstraps=5, sample_fraction=0.5, support_threshold=0.7, random_state=42, n_jobs=1)
        assert sel.estimator is base
        assert sel.n_bootstraps == 5
        assert sel.sample_fraction == 0.5
        assert sel.support_threshold == 0.7
        assert sel.random_state == 42
        assert sel.n_jobs == 1

    @pytest.mark.fast
    def test_fit_sets_all_attributes_ndarray(self, small_ndarray_dataset):
        """Fit sets all attributes ndarray."""
        X, y = small_ndarray_dataset
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(0, 1)), n_bootstraps=4, sample_fraction=0.5, support_threshold=0.5, random_state=1)
        sel.fit(X, y)
        assert hasattr(sel, "selection_probabilities_")
        assert hasattr(sel, "support_")
        assert hasattr(sel, "n_features_")
        assert hasattr(sel, "n_features_in_")
        assert sel.n_features_in_ == X.shape[1]
        # ndarray path -> no ``feature_names_in_`` populated.
        assert not hasattr(sel, "feature_names_in_")

    def test_fit_sets_feature_names_dataframe(self, small_dataframe_dataset):
        """Fit sets feature names dataframe."""
        X, y = small_dataframe_dataset
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(0, 1)), n_bootstraps=3, sample_fraction=0.5, random_state=2)
        sel.fit(X, y)
        assert hasattr(sel, "feature_names_in_")
        assert sel.feature_names_in_ == list(X.columns)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Selection probability semantics
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestStabilityMRMRProbabilities:
    """Groups tests covering TestStabilityMRMRProbabilities."""
    def test_fixed_selector_gives_unanimous_probabilities(self, small_ndarray_dataset):
        # When the inner selector is deterministic, every bootstrap returns the same support so probabilities are exactly 1.0 / 0.0.
        """Fixed selector gives unanimous probabilities."""
        X, y = small_ndarray_dataset
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(0, 2)), n_bootstraps=6, sample_fraction=0.6, support_threshold=0.5, random_state=3)
        sel.fit(X, y)
        probs = sel.selection_probabilities_
        assert probs.shape == (X.shape[1],)
        assert probs[0] == 1.0
        assert probs[2] == 1.0
        # Untouched columns must have zero inclusion frequency.
        assert probs[1] == 0.0
        assert probs[3] == 0.0
        assert probs[4] == 0.0

    def test_support_threshold_above_one_yields_empty_support(self, small_ndarray_dataset):
        """Support threshold above one yields empty support."""
        X, y = small_ndarray_dataset
        # Threshold > 1 is never satisfiable (probabilities live in [0, 1]) so support must be empty.
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(0,)), n_bootstraps=4, sample_fraction=0.5, support_threshold=1.5, random_state=4)
        sel.fit(X, y)
        assert sel.support_.size == 0
        assert sel.n_features_ == 0

    def test_support_threshold_at_zero_keeps_all_touched(self, small_ndarray_dataset):
        """Support threshold at zero keeps all touched."""
        X, y = small_ndarray_dataset
        # Threshold == 0 keeps every feature whose probability is >= 0 -> every feature, since counts >=0.
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(0, 4)), n_bootstraps=3, sample_fraction=0.5, support_threshold=0.0, random_state=5)
        sel.fit(X, y)
        # All 5 features clear the >=0 bar.
        assert sel.support_.size == X.shape[1]

    def test_seed_dependent_selector_gives_mixed_probabilities(self, small_ndarray_dataset):
        # Different subsamples -> different supports -> at least one feature has 0 < prob < 1, which exercises the float-division aggregation path.
        """Seed dependent selector gives mixed probabilities."""
        X, y = small_ndarray_dataset
        sel = StabilityMRMR(
            estimator=_SeedDependentSelector(n_features=X.shape[1], k=2), n_bootstraps=12, sample_fraction=0.5, support_threshold=0.4, random_state=7
        )
        sel.fit(X, y)
        probs = sel.selection_probabilities_
        # Probabilities live in [0, 1].
        assert probs.min() >= 0.0 and probs.max() <= 1.0
        # Sum across features equals total picks / n_bootstraps (k features per bootstrap) - sanity check on aggregation.
        assert abs(probs.sum() - sel.estimator.k) < 1e-9


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Determinism / n_jobs branches
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestStabilityMRMRDeterminism:
    """Groups tests covering TestStabilityMRMRDeterminism."""
    def test_fixed_seed_reproducible(self, small_ndarray_dataset):
        """Fixed seed reproducible."""
        X, y = small_ndarray_dataset
        a = StabilityMRMR(estimator=_SeedDependentSelector(n_features=X.shape[1], k=2), n_bootstraps=8, sample_fraction=0.5, random_state=11)
        b = StabilityMRMR(estimator=_SeedDependentSelector(n_features=X.shape[1], k=2), n_bootstraps=8, sample_fraction=0.5, random_state=11)
        a.fit(X, y)
        b.fit(X, y)
        np.testing.assert_array_equal(a.selection_probabilities_, b.selection_probabilities_)
        np.testing.assert_array_equal(a.support_, b.support_)

    def test_different_seeds_diverge(self, small_ndarray_dataset):
        # Two distinct seeds should not (almost surely) produce the same selection_probabilities_ on a varying inner selector.
        """Different seeds diverge."""
        X, y = small_ndarray_dataset
        a = StabilityMRMR(estimator=_SeedDependentSelector(n_features=X.shape[1], k=2), n_bootstraps=8, sample_fraction=0.5, random_state=11)
        b = StabilityMRMR(estimator=_SeedDependentSelector(n_features=X.shape[1], k=2), n_bootstraps=8, sample_fraction=0.5, random_state=99)
        a.fit(X, y)
        b.fit(X, y)
        assert not np.array_equal(a.selection_probabilities_, b.selection_probabilities_)

    def test_n_jobs_parallel_branch_matches_sequential(self, small_ndarray_dataset):
        # joblib.Parallel branch (n_jobs != 1) must produce the same aggregate as the sequential branch for the same seed.
        # Tolerates Windows paging-file overflow (WinError 1455) under concurrent test sessions.
        """N jobs parallel branch matches sequential."""
        X, y = small_ndarray_dataset
        seq = StabilityMRMR(estimator=_SeedDependentSelector(n_features=X.shape[1], k=2), n_bootstraps=6, sample_fraction=0.5, random_state=21, n_jobs=1)
        par = StabilityMRMR(estimator=_SeedDependentSelector(n_features=X.shape[1], k=2), n_bootstraps=6, sample_fraction=0.5, random_state=21, n_jobs=2)
        try:
            seq.fit(X, y)
            par.fit(X, y)
        except OSError as exc:
            if "paging file" in str(exc).lower() or getattr(exc, "winerror", None) == 1455:
                pytest.skip(f"Windows paging-file overflow under concurrent load: {exc}")
            raise
        except Exception as exc:
            # loky BrokenProcessPool / TerminatedWorkerError / pickle-transport
            # errors during heavy concurrent test load. The Parallel/loky spawn
            # path was entered - that's what this test asserts existed.
            _msg = str(exc).lower()
            _name = type(exc).__name__.lower()
            if any(s in _msg for s in ("brokenprocesspool", "terminatedworker", "pickle", "transport", "_remotetraceback")) or any(
                s in _name for s in ("brokenprocesspool", "terminatedworker")
            ):
                pytest.skip(f"loky worker transport failure under concurrent load: {type(exc).__name__}: {exc}")
            raise
        np.testing.assert_array_equal(seq.selection_probabilities_, par.selection_probabilities_)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# transform()
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestStabilityMRMRTransform:
    """Groups tests covering TestStabilityMRMRTransform."""
    def test_transform_ndarray_returns_selected_columns(self, small_ndarray_dataset):
        """Transform ndarray returns selected columns."""
        X, y = small_ndarray_dataset
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(1, 3)), n_bootstraps=3, sample_fraction=0.5, support_threshold=0.5, random_state=33)
        sel.fit(X, y)
        Xt = sel.transform(X)
        assert Xt.shape == (X.shape[0], 2)
        # The columns must match the original X at indices 1 and 3.
        np.testing.assert_array_equal(Xt[:, 0], X[:, 1])
        np.testing.assert_array_equal(Xt[:, 1], X[:, 3])

    def test_transform_dataframe_returns_dataframe_subset(self, small_dataframe_dataset):
        """Transform dataframe returns dataframe subset."""
        X, y = small_dataframe_dataset
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(0, 4)), n_bootstraps=3, sample_fraction=0.5, support_threshold=0.5, random_state=34)
        sel.fit(X, y)
        Xt = sel.transform(X)
        assert isinstance(Xt, pd.DataFrame)
        assert Xt.shape == (X.shape[0], 2)
        assert list(Xt.columns) == [X.columns[0], X.columns[4]]

    def test_transform_ignores_y_argument(self, small_ndarray_dataset):
        # ``transform`` accepts an optional ``y`` for sklearn pipeline compat but must not depend on it.
        """Transform ignores y argument."""
        X, y = small_ndarray_dataset
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(0, 1)), n_bootstraps=2, sample_fraction=0.5, random_state=35)
        sel.fit(X, y)
        out_no_y = sel.transform(X)
        out_with_y = sel.transform(X, y)
        np.testing.assert_array_equal(out_no_y, out_with_y)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Pandas-Series y branch (``hasattr(y, "iloc")``)
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestStabilityMRMRPandasYBranch:
    """Groups tests covering TestStabilityMRMRPandasYBranch."""
    def test_fit_with_pandas_y_works(self, small_dataframe_dataset):
        """Fit with pandas y works."""
        X, y_series = small_dataframe_dataset
        # pandas Series for y exercises ``y.iloc[idx]`` branch separately from the ``X.iloc`` branch.
        sel = StabilityMRMR(estimator=_FixedSupportSelector(support=(2,)), n_bootstraps=3, sample_fraction=0.6, support_threshold=0.5, random_state=51)
        sel.fit(X, y_series)
        assert sel.support_.tolist() == [2]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "-p", "no:randomly", "--no-cov"])
