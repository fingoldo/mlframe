"""
Shared fixtures for feature_selection tests.
Used by both test_wrappers.py and test_filters.py
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


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
