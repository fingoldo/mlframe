"""
Shared fixtures for feature_selection tests.
Used by both test_wrappers.py and test_filters.py
"""
import os
import sys

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


_MRMR_MOD_NAME = "mlframe.feature_selection.filters.mrmr"


@pytest.fixture(autouse=True)
def _clear_mrmr_fit_cache_between_tests():
    """Reset the process-wide ``MRMR._FIT_CACHE`` before each test so the
    order-dependent cache state doesn't leak across the suite.

    Pre-fix (batch-2 random-ordering run): cache entries from earlier tests
    survived into ``test_mrmr_cache_does_not_collide_on_distinct_targets_with_shared_samples``
    (which probes negative collision semantics) and
    ``test_fix2_pandas_target_columns_cleaned_after_fit_exception`` (which checks
    post-fit state), causing both to fail intermittently under
    ``pytest-randomly`` ordering. Both pass in isolation. The cache is a class
    attribute on MRMR, so a fresh one is needed at every test boundary.

    Public ``MRMR.clear_fit_cache()`` is already documented as the
    suite-boundary drain hook (mrmr.py:511-519); we reuse it here so test
    isolation matches the production-mediated reset path exactly.

    Fast-path: ~54 / 118 files under tests/feature_selection/ never touch
    MRMR, so the autouse import would force the ~hundreds-of-ms numba+sklearn
    subgraph eagerly. Probe ``sys.modules`` first; if the mrmr module is not
    yet loaded, no test in scope has triggered a fit and the cache is
    guaranteed empty -- skip the import entirely.
    """
    _mrmr_mod = sys.modules.get(_MRMR_MOD_NAME)
    if _mrmr_mod is None:
        yield
        return
    try:
        _mrmr_mod.MRMR.clear_fit_cache()
        # Module-level cross-target identity cache must also drain: a prior
        # test storing an "identity" fingerprint can short-circuit fit() in a
        # later test via _fit_identity_shortcut, bypassing the target-column
        # injection and cleanup -- the order-dependent flake in
        # test_fix2_pandas_target_columns_cleaned_after_fit_exception under
        # random ordering.
        with getattr(_mrmr_mod, "_MRMR_IDENTITY_FP_LOCK", _NullCtx()):
            _mrmr_mod._MRMR_IDENTITY_FP_CACHE.clear()
    except Exception:
        # If clearing fails (e.g. coverage-active subprocess that skips
        # heavy fixtures), don't block the test from running its own setup.
        pass
    yield


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


# Re-export the canonical fast-mode flag and helper from the root conftest. Subdir tests historically imported ``IS_FAST_MODE`` from this conftest; preserve that name
# but make the import-time snapshot a thin re-export of the live root value. ``is_fast_mode()`` is the authoritative live check used inside the collection hook below;
# the constant captures the value at import time for callers that only need a parametrize-decorator-time snapshot.
from tests.conftest import IS_FAST_MODE, is_fast_mode  # noqa: F401, E402


def _coverage_active() -> bool:
    """True iff coverage.py is currently tracing this process. Used to skip tests that spawn threads via joblib.Parallel /
    multiprocessing.dummy - those interact badly with coverage's sys.settrace on Windows (RuntimeError: can't start new
    thread + AttributeError: 'DummyProcess' has no 'terminate') and break the coverage run. Tests still pass in standard
    pytest invocations; only the coverage-measurement path skips them."""
    try:
        import coverage as _cov
        return _cov.Coverage.current() is not None
    except Exception:
        return False


COVERAGE_ACTIVE = _coverage_active()


# NOTE: a richer ``fast_subset`` lives in tests/conftest.py (top-level).
# This local copy preserves the historical ``n=`` kwarg used by callers in
# ``feature_selection``. Both share the same MLFRAME_FAST env-var gate via
# the live ``is_fast_mode()`` check (not the import-time snapshot, so a
# mid-session env mutation is observed correctly).
def fast_subset(seq, n: int = 1):
    items = list(seq)
    if is_fast_mode() and len(items) > n:
        return items[:n]
    return items


# Marker for tests that are too slow to run under fast mode (e.g. multi-thousand-row benches, multi-seed h2h vs sklearn). Test bodies stay
# untouched; the collection hook below skips them when MLFRAME_FAST=1.
def _register_slow_marker(config):
    config.addinivalue_line(
        "markers",
        "slow: skip when MLFRAME_FAST=1 (heavy bench / multi-seed / etc.)",
    )


# Register `no_xdist` marker. The marker alone is a no-op without the
# collection hook below; tests that rely on stable file-system state (golden
# tests wiping numba __pycache__) must skip when pytest-xdist parallelism is
# active so workers don't compete for the same cache directory.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_xdist: skip when pytest-xdist is collecting workers in parallel",
    )
    _register_slow_marker(config)


def pytest_collection_modifyitems(config, items):
    skip_xdist = pytest.mark.skip(reason="requires sequential execution (numba cache lock)")
    skip_slow = pytest.mark.skip(reason="MLFRAME_FAST=1 set; slow-marked test skipped for fast iteration")
    dist = getattr(config.option, "dist", "no")
    # ``is_fast_mode()`` is live (re-reads MLFRAME_FAST) and reflects ``--fast`` set by the root conftest's ``pytest_configure`` even when this hook runs before the snapshot updates.
    fast = is_fast_mode()
    for item in items:
        if dist != "no" and "no_xdist" in item.keywords:
            item.add_marker(skip_xdist)
        if fast and "slow" in item.keywords:
            item.add_marker(skip_slow)


# ================================================================================================
# Common Classification Fixtures
# ================================================================================================

@pytest.fixture
def simple_classification_data():
    """Generate simple classification data with known informative features. Delegates to the centralized ``make_informative_noise_classification`` builder in ``tests/training/synthetic.py`` so seed handling + feature naming match the project-wide convention."""
    from tests.training.synthetic import make_informative_noise_classification
    return make_informative_noise_classification(n_samples=200, n_informative=5, n_noise=15, seed=42)


@pytest.fixture
def simple_regression_data():
    """Generate simple regression data with known informative features. Local RNG (no global mutation)."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_informative = 5
    n_noise = 15

    X_informative = rng.standard_normal(size=(n_samples, n_informative))
    y = 3 * X_informative[:, 0] + 2 * X_informative[:, 1] - X_informative[:, 2] + \
        0.5 * X_informative[:, 3] + rng.standard_normal(n_samples) * 0.1

    X_noise = rng.standard_normal(size=(n_samples, n_noise))
    X = np.hstack([X_informative, X_noise])

    feature_names = [f'informative_{i}' for i in range(n_informative)] + \
                   [f'noise_{i}' for i in range(n_noise)]

    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, list(range(n_informative))


@pytest.fixture
def imbalanced_classification_data():
    """Generate imbalanced classification data (95/5 split). Local RNG (no global mutation)."""
    rng = np.random.default_rng(42)
    n_samples = 400
    n_informative = 5
    n_noise = 15

    X_informative = rng.standard_normal(size=(n_samples, n_informative))

    scores = X_informative[:, 0] + X_informative[:, 1] - X_informative[:, 2]
    threshold = np.percentile(scores, 95)
    y = (scores > threshold).astype(int)

    X_noise = rng.standard_normal(size=(n_samples, n_noise))
    X = np.hstack([X_informative, X_noise])

    feature_names = [f'informative_{i}' for i in range(n_informative)] + \
                   [f'noise_{i}' for i in range(n_noise)]

    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, list(range(n_informative))


@pytest.fixture
def multiclass_data():
    """Generate multiclass classification data."""
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=5,
        n_redundant=0,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(20)]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, list(range(5))


@pytest.fixture
def high_dimensional_data():
    """Generate high-dimensional data (p > n). Local RNG (no global mutation). Kept inline rather than migrated to ``make_informative_noise_classification`` because the legacy signal uses x0+x1 only (n_informative=2 score) while reporting informative_indices=range(3); the centralized builder couples score-arity to n_informative, so a drop-in migration would silently change y for existing consumers."""
    rng = np.random.default_rng(42)
    n_samples = 50
    n_informative = 3
    n_noise = 100

    X_informative = rng.standard_normal(size=(n_samples, n_informative))
    y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)

    X_noise = rng.standard_normal(size=(n_samples, n_noise))
    X = np.hstack([X_informative, X_noise])

    feature_names = [f'informative_{i}' for i in range(n_informative)] + \
                   [f'noise_{i}' for i in range(n_noise)]

    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, list(range(n_informative))


@pytest.fixture
def correlated_features_data():
    """Generate data with correlated informative features. Local RNG (no global mutation)."""
    rng = np.random.default_rng(42)
    n_samples = 200

    base1 = rng.standard_normal(n_samples)
    base2 = rng.standard_normal(n_samples)

    X_informative = np.column_stack([
        base1,
        base1 + rng.standard_normal(n_samples) * 0.1,
        base2,
        base2 + rng.standard_normal(n_samples) * 0.1,
        base1 + base2
    ])

    y = (base1 + base2 > 0).astype(int)

    X_noise = rng.standard_normal(size=(n_samples, 15))
    X = np.hstack([X_informative, X_noise])

    feature_names = [f'informative_{i}' for i in range(5)] + \
                   [f'noise_{i}' for i in range(15)]

    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, list(range(5))


# ================================================================================================
# MRMR-Specific Fixtures
# ================================================================================================

@pytest.fixture
def synergistic_features_data():
    """
    Generate data with synergistic feature relationships.
    y = a^2/b + log(c)*sin(d)
    MRMR should detect all 4 features and recommend engineered features.
    """
    rng = np.random.default_rng(42)
    n = 10_000  # Smaller for faster tests

    a = rng.random(n) + 0.1  # Avoid zero
    b = rng.random(n) + 0.1
    c = rng.random(n) + 0.1
    d = rng.random(n) * 2 * np.pi
    e = rng.random(n)  # Noise feature

    y = a**2/b + np.log(c)*np.sin(d)

    df = pd.DataFrame({
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
    })

    return df, y, ['a', 'b', 'c', 'd']


@pytest.fixture
def multiplicative_synergy_data():
    """Data where y = a * b + noise, testing multiplicative synergy detection."""
    rng = np.random.default_rng(42)
    n = 5_000

    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)  # Noise

    y = a * b + rng.standard_normal(n) * 0.1

    df = pd.DataFrame({
        'a': a,
        'b': b,
        'c': c,
    })

    return df, y, ['a', 'b']


@pytest.fixture
def additive_synergy_data():
    """Data where y = a + b + noise, testing additive relationship."""
    rng = np.random.default_rng(42)
    n = 5_000

    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)  # Noise

    y = a + b + rng.standard_normal(n) * 0.1

    df = pd.DataFrame({
        'a': a,
        'b': b,
        'c': c,
    })

    return df, y, ['a', 'b']


@pytest.fixture
def nonlinear_transform_data():
    """Data with nonlinear transforms: y = sin(a) + log(b+1) + c^2"""
    rng = np.random.default_rng(42)
    n = 5_000

    a = rng.random(n) * 2 * np.pi
    b = rng.random(n)
    c = rng.standard_normal(n)
    d = rng.standard_normal(n)  # Noise

    y = np.sin(a) + np.log(b + 1) + c**2 + rng.standard_normal(n) * 0.1

    df = pd.DataFrame({
        'a': a,
        'b': b,
        'c': c,
        'd': d,
    })

    return df, y, ['a', 'b', 'c']


@pytest.fixture
def feature_engineering_example_data():
    """
    The user's exact example from the ticket:
    y = a^2/b + f/5 + log(c)*sin(d)

    Note: 'f' is not in DataFrame, so effectively y = a^2/b + log(c)*sin(d)
    MRMR should select a, b, c, d and recommend:
    - mul(log(c), sin(d))
    - mul(squared(a), reciproc(b))
    """
    rng = np.random.default_rng(42)
    n = 10_000

    a = rng.random(n) + 0.1
    b = rng.random(n) + 0.1
    c = rng.random(n) + 0.1
    d = rng.random(n) * 2 * np.pi
    e = rng.random(n)  # Noise

    y = a**2/b + np.log(c)*np.sin(d)

    df = pd.DataFrame({
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
    })

    return df, y, ['a', 'b', 'c', 'd']


@pytest.fixture
def known_mi_data():
    """Data with known mutual information for testing MI computation."""
    rng = np.random.default_rng(42)
    n = 1000

    # Create perfectly predictable relationship
    x = rng.integers(0, 5, n)
    y = x.copy()  # MI(X,Y) = H(X) for identical variables

    return x, y
