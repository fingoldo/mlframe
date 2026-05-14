"""
Shared fixtures for feature_selection tests.
Used by both test_wrappers.py and test_filters.py
"""
import os

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


# Fast-mode flag: when MLFRAME_FAST=1 (or "true"/"yes"), parametric / loop-heavy tests collapse to a single representative case so the suite runs
# in a fraction of full-suite time during local dev iteration. Each fast-mode-aware test still hits every code branch with one input.
IS_FAST_MODE = os.environ.get("MLFRAME_FAST", "").strip().lower() in ("1", "true", "yes", "on")


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


def fast_subset(seq, n: int = 1):
    """Return the first ``n`` items of ``seq`` when ``MLFRAME_FAST=1``; otherwise return ``seq`` unchanged.

    Use to slim parametric coverage for fast-mode iteration:
        @pytest.mark.parametrize("cv_n", fast_subset([2, 3, 5, 10]))
    Returns a list so pytest's parametrize can introspect it; sequences shorter than ``n`` pass through unchanged.
    """
    items = list(seq)
    if IS_FAST_MODE and len(items) > n:
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
    for item in items:
        if dist != "no" and "no_xdist" in item.keywords:
            item.add_marker(skip_xdist)
        if IS_FAST_MODE and "slow" in item.keywords:
            item.add_marker(skip_slow)


# ================================================================================================
# Common Classification Fixtures
# ================================================================================================

@pytest.fixture
def simple_classification_data():
    """Generate simple classification data with known informative features."""
    np.random.seed(42)
    n_samples = 200
    n_informative = 5
    n_noise = 15

    X_informative = np.random.randn(n_samples, n_informative)
    y = (X_informative[:, 0] + X_informative[:, 1] - X_informative[:, 2] > 0).astype(int)

    X_noise = np.random.randn(n_samples, n_noise)
    X = np.hstack([X_informative, X_noise])

    feature_names = [f'informative_{i}' for i in range(n_informative)] + \
                   [f'noise_{i}' for i in range(n_noise)]

    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, list(range(n_informative))


@pytest.fixture
def simple_regression_data():
    """Generate simple regression data with known informative features."""
    np.random.seed(42)
    n_samples = 200
    n_informative = 5
    n_noise = 15

    X_informative = np.random.randn(n_samples, n_informative)
    y = 3 * X_informative[:, 0] + 2 * X_informative[:, 1] - X_informative[:, 2] + \
        0.5 * X_informative[:, 3] + np.random.randn(n_samples) * 0.1

    X_noise = np.random.randn(n_samples, n_noise)
    X = np.hstack([X_informative, X_noise])

    feature_names = [f'informative_{i}' for i in range(n_informative)] + \
                   [f'noise_{i}' for i in range(n_noise)]

    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, list(range(n_informative))


@pytest.fixture
def imbalanced_classification_data():
    """Generate imbalanced classification data (95/5 split)."""
    np.random.seed(42)
    n_samples = 400
    n_informative = 5
    n_noise = 15

    X_informative = np.random.randn(n_samples, n_informative)

    scores = X_informative[:, 0] + X_informative[:, 1] - X_informative[:, 2]
    threshold = np.percentile(scores, 95)
    y = (scores > threshold).astype(int)

    X_noise = np.random.randn(n_samples, n_noise)
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
    """Generate high-dimensional data (p > n)."""
    np.random.seed(42)
    n_samples = 50
    n_informative = 3
    n_noise = 100

    X_informative = np.random.randn(n_samples, n_informative)
    y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)

    X_noise = np.random.randn(n_samples, n_noise)
    X = np.hstack([X_informative, X_noise])

    feature_names = [f'informative_{i}' for i in range(n_informative)] + \
                   [f'noise_{i}' for i in range(n_noise)]

    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, y, list(range(n_informative))


@pytest.fixture
def correlated_features_data():
    """Generate data with correlated informative features."""
    np.random.seed(42)
    n_samples = 200

    base1 = np.random.randn(n_samples)
    base2 = np.random.randn(n_samples)

    X_informative = np.column_stack([
        base1,
        base1 + np.random.randn(n_samples) * 0.1,
        base2,
        base2 + np.random.randn(n_samples) * 0.1,
        base1 + base2
    ])

    y = (base1 + base2 > 0).astype(int)

    X_noise = np.random.randn(n_samples, 15)
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
    np.random.seed(42)
    n = 10_000  # Smaller for faster tests

    a = np.random.rand(n) + 0.1  # Avoid zero
    b = np.random.rand(n) + 0.1
    c = np.random.rand(n) + 0.1
    d = np.random.rand(n) * 2 * np.pi
    e = np.random.rand(n)  # Noise feature

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
    np.random.seed(42)
    n = 5_000

    a = np.random.randn(n)
    b = np.random.randn(n)
    c = np.random.randn(n)  # Noise

    y = a * b + np.random.randn(n) * 0.1

    df = pd.DataFrame({
        'a': a,
        'b': b,
        'c': c,
    })

    return df, y, ['a', 'b']


@pytest.fixture
def additive_synergy_data():
    """Data where y = a + b + noise, testing additive relationship."""
    np.random.seed(42)
    n = 5_000

    a = np.random.randn(n)
    b = np.random.randn(n)
    c = np.random.randn(n)  # Noise

    y = a + b + np.random.randn(n) * 0.1

    df = pd.DataFrame({
        'a': a,
        'b': b,
        'c': c,
    })

    return df, y, ['a', 'b']


@pytest.fixture
def nonlinear_transform_data():
    """Data with nonlinear transforms: y = sin(a) + log(b+1) + c^2"""
    np.random.seed(42)
    n = 5_000

    a = np.random.rand(n) * 2 * np.pi
    b = np.random.rand(n)
    c = np.random.randn(n)
    d = np.random.randn(n)  # Noise

    y = np.sin(a) + np.log(b + 1) + c**2 + np.random.randn(n) * 0.1

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
    np.random.seed(42)
    n = 10_000

    a = np.random.rand(n) + 0.1
    b = np.random.rand(n) + 0.1
    c = np.random.rand(n) + 0.1
    d = np.random.rand(n) * 2 * np.pi
    e = np.random.rand(n)  # Noise

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
    np.random.seed(42)
    n = 1000

    # Create perfectly predictable relationship
    x = np.random.randint(0, 5, n)
    y = x.copy()  # MI(X,Y) = H(X) for identical variables

    return x, y
