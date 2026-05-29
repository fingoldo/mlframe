"""Unit tests for the chunked ANOVA F-statistic used by the SHAP-proxied prefilter stage-A.

Parity bar: numerical match with sklearn's ``f_classif`` / ``f_regression`` to float64 tolerance
on dense randn fixtures, since the prefilter consumes the RANKING and any non-trivial drift would
shift the survivor set. Auxiliary tests cover the constant-column / degenerate-K sentinel,
``resolve_batch_size`` priority order, and the psutil-missing fallback."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection._shap_proxy_prefilter_univariate import (
    _AUTO_BATCH_MAX,
    _AUTO_BATCH_MIN,
    f_classif_chunked,
    f_regression_chunked,
    resolve_batch_size,
)


@pytest.fixture
def small_classif():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 50)).astype(np.float64)
    y = rng.integers(0, 3, size=200)
    return X, y


@pytest.fixture
def small_regr():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 50)).astype(np.float64)
    y = X[:, 0] * 0.7 - X[:, 1] * 0.3 + rng.normal(scale=0.5, size=200)
    return X, y


def test_f_classif_chunked_matches_sklearn(small_classif):
    from sklearn.feature_selection import f_classif

    X, y = small_classif
    expected, _ = f_classif(X, y)
    got = f_classif_chunked(X, y, batch_size=8)
    assert got.shape == expected.shape
    # Bit-identical on randn dense (no constant cols); use relative tol for safety.
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_f_regression_chunked_matches_sklearn(small_regr):
    from sklearn.feature_selection import f_regression

    X, y = small_regr
    expected, _ = f_regression(X, y)
    got = f_regression_chunked(X, y, batch_size=8)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)


def test_f_classif_constant_column_sentinel():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 10)).astype(np.float64)
    X[:, 3] = 1.234  # constant column -> zero within-group SS -> -inf sentinel
    y = rng.integers(0, 2, size=100)
    got = f_classif_chunked(X, y, batch_size=4)
    assert got[3] == -np.inf
    assert np.isfinite(got[0])


def test_f_regression_constant_column_sentinel():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 10)).astype(np.float64)
    X[:, 7] = -2.5
    y = rng.normal(size=100)
    got = f_regression_chunked(X, y, batch_size=4)
    assert got[7] == -np.inf
    assert np.isfinite(got[0])


def test_f_classif_degenerate_N_le_K():
    # N <= K -> df_within == 0 -> -inf vector (matches sklearn's NaN -> our sentinel).
    X = np.arange(9, dtype=np.float64).reshape(3, 3)
    y = np.array([0, 1, 2])
    got = f_classif_chunked(X, y, batch_size=2)
    assert np.all(got == -np.inf)


def test_f_regression_constant_target_all_minus_inf():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 5)).astype(np.float64)
    y = np.full(50, 3.14)
    got = f_regression_chunked(X, y, batch_size=2)
    assert np.all(got == -np.inf)


def test_f_classif_chunk_independence(small_classif):
    """Batched output must be invariant to batch_size choice for a given input."""
    X, y = small_classif
    a = f_classif_chunked(X, y, batch_size=1)
    b = f_classif_chunked(X, y, batch_size=7)
    c = f_classif_chunked(X, y, batch_size=50)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a, c, rtol=1e-12, atol=1e-14)


def test_resolve_batch_size_user_value_clamps():
    # User value wins, clamped to [1, n_features].
    assert resolve_batch_size(100, 1000, user_value=50) == 50
    assert resolve_batch_size(100, 1000, user_value=999) == 100
    assert resolve_batch_size(100, 1000, user_value=0) == 1
    assert resolve_batch_size(100, 1000, user_value=-5) == 1


def test_resolve_batch_size_auto_bounds():
    # Auto path always returns a value in [_AUTO_BATCH_MIN, _AUTO_BATCH_MAX], capped by n_features.
    got = resolve_batch_size(20000, 10000, user_value=None)
    assert _AUTO_BATCH_MIN <= got <= _AUTO_BATCH_MAX
    assert got <= 20000
    # Small feature count -> capped by n_features even if min is 256.
    got_small = resolve_batch_size(50, 10000, user_value=None)
    assert got_small == 50


def test_resolve_batch_size_cache_hit(monkeypatch):
    """Kernel-tuning cache hit overrides the auto path."""

    class _StubCache:
        def lookup(self, kernel_name, **dims):
            assert kernel_name == "shap_proxy_prefilter_univariate_batch"
            return {"batch_size": 1024}

    import mlframe.feature_selection._shap_proxy_prefilter_univariate as mod

    monkeypatch.setattr(
        "mlframe.feature_selection.filters._kernel_tuning.get_kernel_tuning_cache",
        lambda: _StubCache(),
    )
    got = resolve_batch_size(20000, 10000, user_value=None)
    assert got == 1024
    # Cached value above n_features is still clamped down.
    got_clamp = mod.resolve_batch_size(500, 10000, user_value=None)
    assert got_clamp == 500


def test_resolve_batch_size_cache_miss_falls_through(monkeypatch):
    """Cache miss (None lookup) drops to the auto path."""

    class _StubCache:
        def lookup(self, kernel_name, **dims):
            return None

    monkeypatch.setattr(
        "mlframe.feature_selection.filters._kernel_tuning.get_kernel_tuning_cache",
        lambda: _StubCache(),
    )
    got = resolve_batch_size(20000, 10000, user_value=None)
    assert _AUTO_BATCH_MIN <= got <= _AUTO_BATCH_MAX


def test_resolve_batch_size_psutil_missing(monkeypatch):
    """ImportError on psutil falls through to the chunk-bytes cap alone, still bounded."""
    import mlframe.feature_selection._shap_proxy_prefilter_univariate as mod

    monkeypatch.setattr(mod, "_available_ram_bytes", lambda: None)
    got = resolve_batch_size(20000, 10000, user_value=None)
    assert _AUTO_BATCH_MIN <= got <= _AUTO_BATCH_MAX


def test_f_classif_dataframe_input(small_classif):
    pd = pytest.importorskip("pandas")
    X, y = small_classif
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(X.shape[1])])
    from sklearn.feature_selection import f_classif

    expected, _ = f_classif(X, y)
    got = f_classif_chunked(df, y, batch_size=8)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_f_classif_input_dtype_upcast(small_classif):
    X, y = small_classif
    X32 = X.astype(np.float32)
    from sklearn.feature_selection import f_classif

    expected, _ = f_classif(X, y)
    got = f_classif_chunked(X32, y, batch_size=8)
    # f32 inputs: sklearn upcasts internally too; allow a slightly looser tol.
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)
