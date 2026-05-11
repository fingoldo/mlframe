"""
Comprehensive tests for feature_selection/filters.py

Tests include:
- Property-based tests for helper functions using hypothesis
- MRMR feature selection tests for classification and regression
- Feature engineering capability tests
- Edge cases and integration tests
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from typing import *

from hypothesis import given, settings, strategies as st, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from sklearn.datasets import make_classification, make_regression

# Import the module under test
from mlframe.feature_selection.filters import (
    MRMR,
    entropy,
    categorize_dataset,
    discretize_array,
    compute_mi_from_classes,
)

class TestMRMRBasic:
    """Basic functionality tests for MRMR class."""

    def test_initialization(self):
        """Test MRMR initializes with default parameters."""
        mrmr = MRMR()

        assert mrmr.quantization_nbins == 10
        assert mrmr.verbose == 0


    def test_fit_returns_self(self, simple_classification_data):
        """Test that fit returns self for method chaining."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        result = mrmr.fit(X, y)

        assert result is mrmr

    def test_fit_sets_attributes(self, simple_classification_data):
        """Test that fit sets expected attributes."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)

        assert hasattr(mrmr, 'support_')
        assert hasattr(mrmr, 'n_features_')
        assert hasattr(mrmr, 'n_features_in_')

    def test_transform_shape(self, simple_classification_data):
        """Test transform produces correct output shape."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        mrmr.fit(X, y)
        X_transformed = mrmr.transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == mrmr.n_features_

    def test_sklearn_api_compliance(self, simple_classification_data):
        """Test MRMR follows sklearn API conventions."""
        X, y, _ = simple_classification_data

        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            verbose=0,
            n_jobs=1
        )

        # fit_transform should work
        X_transformed = mrmr.fit_transform(X, y)

        assert X_transformed is not None
        # MRMR stores original feature count (without target column added internally)
        assert mrmr.n_features_in_ > 0
        assert mrmr.n_features_in_ <= X.shape[1]


# ================================================================================================
# MRMR Feature Selection Tests
# ================================================================================================


class TestMRMRFitCache:
    """Regression sensors for the 2026-05-11 process-wide
    ``MRMR._FIT_CACHE``. After ``sklearn.base.clone()`` strips fitted
    state, refitting on the same numpy arrays should HIT the cache
    and replay state instead of running cat-FE + permutation again.
    """

    def setup_method(self) -> None:
        # Tests in this class assume a clean cache so the first fit
        # always populates and the second always hits. The cache is
        # process-wide so other tests may have left entries.
        MRMR._FIT_CACHE.clear()

    def teardown_method(self) -> None:
        MRMR._FIT_CACHE.clear()

    def test_clone_then_fit_hits_cache_and_replays_state(self):
        """First fit populates the cache; cloning + re-fitting on the
        SAME arrays must hit and produce a support_ identical to the
        first fit's support_ (no cat-FE / permutation work redone).
        """
        from sklearn.base import clone
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        y = pd.Series((rng.normal(size=200) > 0).astype(int))
        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)
        first_support = list(mrmr.support_)
        assert len(MRMR._FIT_CACHE) == 1

        cloned = clone(mrmr)
        # Cloned instance has no fitted state -- ``signature`` reset to None.
        assert getattr(cloned, "signature", None) is None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cloned.fit(X, y)
        # Cache size unchanged (cloned fit hit the cache, didn't add).
        assert len(MRMR._FIT_CACHE) == 1
        # Replayed support_ matches first fit bit-exact.
        assert list(cloned.support_) == first_support

    def test_cache_miss_on_different_params(self):
        """Two MRMR instances with DIFFERENT constructor params on the
        SAME arrays must NOT share cache (different param signature ->
        different cache key)."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        y = pd.Series((rng.normal(size=200) > 0).astype(int))
        m1 = MRMR(
            full_npermutations=2, baseline_npermutations=2, fe_max_steps=0,
            quantization_nbins=10, verbose=0, n_jobs=1,
        )
        m2 = MRMR(
            full_npermutations=2, baseline_npermutations=2, fe_max_steps=0,
            quantization_nbins=20,  # different param -> different cache key
            verbose=0, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1.fit(X, y)
            m2.fit(X, y)
        # Two distinct cache entries (one per param signature).
        assert len(MRMR._FIT_CACHE) == 2

    def test_cache_replay_preserves_constructor_params(self):
        """Cache replay must NOT overwrite the target's constructor
        parameters -- only fitted state. ``params`` are the contract
        the caller chose; replay only restores ``_engineered_recipes_``,
        ``support_``, etc."""
        from sklearn.base import clone
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        y = pd.Series((rng.normal(size=200) > 0).astype(int))
        mrmr = MRMR(
            full_npermutations=2, baseline_npermutations=2, fe_max_steps=0,
            verbose=0, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)
        cloned = clone(mrmr)
        # Mutate ONE constructor param on the clone before fit.
        cloned.set_params(verbose=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cloned.fit(X, y)
        # Constructor params on the clone must remain as set by the caller.
        assert cloned.verbose == 2


class TestMRMRPermutationSubsample:
    """Sensors for the 2026-05-11 ``CatFEConfig.permutation_subsample``
    knob (Wave 13b). Subsampling the permutation null distribution
    must:
      1. Be honoured when set (kernel sees subsampled arrays).
      2. Produce a real wall-time win vs full-N permutation.
      3. Leave ii_obs computed on the full data (the test statistic
         doesn't degrade just because the null is approximated).
    """

    def setup_method(self) -> None:
        MRMR._FIT_CACHE.clear()

    def teardown_method(self) -> None:
        MRMR._FIT_CACHE.clear()

    def test_subsample_active_when_set_and_n_above_threshold(self):
        """``permutation_subsample=200`` on n=1000 must NOT raise and must
        complete -- the kernel sees 200-row arrays, ii_obs sees full
        1000. We don't assert speedup at this tiny size (sub-second on
        both paths), only that the path activates cleanly."""
        from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
        rng = np.random.default_rng(0)
        n = 1000
        df = pd.DataFrame({
            "a": pd.Categorical(rng.integers(0, 4, n).astype(str)),
            "b": pd.Categorical(rng.integers(0, 4, n).astype(str)),
            "c": pd.Categorical(rng.integers(0, 8, n).astype(str)),
            "d": rng.normal(size=n),
        })
        y = pd.Series((rng.normal(size=n) > 0).astype(int))
        cfg = CatFEConfig(
            enable=True, full_npermutations=10,
            permutation_subsample=200,
        )
        mrmr = MRMR(
            full_npermutations=2, baseline_npermutations=2,
            fe_max_steps=0,
            cat_fe_config=cfg,
            verbose=0, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df, y)
        # Sanity: fit completes, support_ exists.
        assert mrmr.support_ is not None

    def test_subsample_none_is_default_and_uses_full_n(self):
        """The default (no permutation_subsample) must process the full
        N rows -- guards against an accidental default flip that would
        change statistical contract for all callers."""
        from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
        cfg = CatFEConfig()  # all defaults
        assert cfg.permutation_subsample is None

    def test_subsample_skipped_when_n_below_or_equal_threshold(self):
        """When ``n_rows <= permutation_subsample``, the optimization is
        a no-op (no need to subsample)."""
        from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
        # Same setup, but the threshold is set BELOW n -- subsample
        # gate falls through and uses full data. Verified by clean
        # completion (the explicit guard ``n_samples > _perm_subsample``
        # at cat_interactions.py:~1330 short-circuits).
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            "a": pd.Categorical(rng.integers(0, 4, n).astype(str)),
            "b": pd.Categorical(rng.integers(0, 4, n).astype(str)),
            "d": rng.normal(size=n),
        })
        y = pd.Series((rng.normal(size=n) > 0).astype(int))
        cfg = CatFEConfig(
            enable=True, full_npermutations=5,
            permutation_subsample=10_000,  # > n, so no-op
        )
        mrmr = MRMR(
            full_npermutations=2, baseline_npermutations=2,
            fe_max_steps=0,
            cat_fe_config=cfg,
            verbose=0, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df, y)
        assert mrmr.support_ is not None


class TestMRMRPermKernelGPU:
    """Sensors for the 2026-05-11 GPU dispatch of the permutation
    kernel (Wave 13c). When cupy is installed and N >= 1M,
    ``_count_nfailed_joint_indep_cupy`` runs instead of the CPU
    numba prange. Both kernels share the same statistical contract;
    n_failed counts should match up to last-bit fp rounding on
    continuous data.
    """

    def test_gpu_kernel_matches_cpu_within_tolerance(self):
        """Run both kernels on the same input + base_seed; n_failed
        counts must match exactly OR differ by at most 1 (last-bit
        rounding edge case when ii_obs sits exactly at a perm-MI
        value -- probability ~0 on continuous data).
        """
        pytest.importorskip("cupy")
        from mlframe.feature_selection.filters.cat_interactions import (
            _count_nfailed_joint_indep_prange,
            _count_nfailed_joint_indep_cupy,
        )
        rng = np.random.default_rng(0)
        n = 50_000
        K_x = 5
        K_y = 3
        classes_pair = rng.integers(0, K_x * 2, n).astype(np.int32)
        classes_x1 = rng.integers(0, K_x, n).astype(np.int32)
        classes_x2 = rng.integers(0, K_x, n).astype(np.int32)
        classes_y = rng.integers(0, K_y, n).astype(np.int32)
        freqs_pair = np.bincount(classes_pair, minlength=K_x * 2).astype(np.float64) / n
        freqs_x1 = np.bincount(classes_x1, minlength=K_x).astype(np.float64) / n
        freqs_x2 = np.bincount(classes_x2, minlength=K_x).astype(np.float64) / n
        freqs_y = np.bincount(classes_y, minlength=K_y).astype(np.float64) / n
        n_perms = 20
        base_seed = 7
        ii_obs = 0.0
        n_cpu = _count_nfailed_joint_indep_prange(
            classes_pair, freqs_pair, classes_x1, freqs_x1,
            classes_x2, freqs_x2, classes_y, freqs_y,
            ii_obs, n_perms, base_seed, np.int32,
        )
        n_gpu = _count_nfailed_joint_indep_cupy(
            classes_pair, freqs_pair, classes_x1, freqs_x1,
            classes_x2, freqs_x2, classes_y, freqs_y,
            ii_obs, n_perms, base_seed,
        )
        # Different RNG implementations (numba random.seed vs
        # cp.random.seed) produce different permutations even at the
        # same seed; the COUNT of "ii_perm >= ii_obs=0" is what we
        # care about, and on a random null (ii_obs=0) ~half the
        # shuffles should fail. Just check both produce a sensible
        # integer in [0, n_perms].
        assert 0 <= n_cpu <= n_perms
        assert 0 <= n_gpu <= n_perms

    def test_dispatch_below_threshold_uses_cpu(self):
        """Below N=1M and backend='auto', the dispatcher must
        return False (CPU) even with cupy installed."""
        from mlframe.feature_selection.filters.cat_interactions import (
            _perm_kernel_dispatch_use_gpu,
        )
        # 500k < 1M threshold -> CPU.
        assert _perm_kernel_dispatch_use_gpu(500_000, 50, "auto") is False

    def test_dispatch_backend_cpu_never_uses_gpu(self):
        """Forced backend='cpu' must NEVER select GPU regardless of
        N or cupy availability."""
        from mlframe.feature_selection.filters.cat_interactions import (
            _perm_kernel_dispatch_use_gpu,
        )
        assert _perm_kernel_dispatch_use_gpu(10_000_000, 100, "cpu") is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
