"""
Shared pytest fixtures for training module tests.
"""

import gc
import pytest
import numpy as np
import pandas as pd
import polars as pl
import warnings
from pathlib import Path


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test to prevent OOM issues."""
    yield
    gc.collect()


@pytest.fixture
def sample_regression_data():
    """Generate synthetic regression dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # True relationship: y = 2*x0 + 3*x1 - 1.5*x2 + noise
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(n_samples) * 0.5

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df, feature_names, y


@pytest.fixture
def sample_classification_data():
    """Generate synthetic binary classification dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # True relationship: logistic decision boundary
    logits = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2]
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df, feature_names, y


@pytest.fixture
def sample_polars_data(sample_regression_data):
    """Convert sample data to Polars DataFrame."""
    df, feature_names, y = sample_regression_data
    pl_df = pl.from_pandas(df)
    return pl_df, feature_names, y


@pytest.fixture
def sample_timeseries_data():
    """Generate synthetic time series dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1h')
    X = np.random.randn(n_samples, n_features)
    y = np.sin(np.arange(n_samples) / 50) + np.random.randn(n_samples) * 0.1

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['timestamp'] = dates
    df['target'] = y

    return df, feature_names, dates, y


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create a temporary directory for models."""
    models_dir = tmp_path / "test_models"
    models_dir.mkdir()
    return str(models_dir)


# ================================================================================================
# Warning Suppression
# ================================================================================================

@pytest.fixture(autouse=True)
def suppress_convergence_warnings():
    """Suppress convergence warnings during tests."""
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
    warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
    warnings.filterwarnings("ignore", message=".*Objective did not converge.*")
    yield
    warnings.resetwarnings()


# ================================================================================================
# Categorical Data Fixtures
# ================================================================================================

@pytest.fixture
def sample_categorical_data():
    """Generate dataset with high-cardinality categorical features."""
    np.random.seed(42)
    n_samples = 500  # Small dataset as per user requirement
    n_numeric = 5

    # Numeric features
    X_numeric = np.random.randn(n_samples, n_numeric)

    # High-cardinality categorical features
    cat_feature_1 = np.random.choice([f'cat_A_{i}' for i in range(100)], n_samples)  # 100 categories
    cat_feature_2 = np.random.choice([f'cat_B_{i}' for i in range(50)], n_samples)   # 50 categories
    cat_feature_3 = np.random.choice(['X', 'Y', 'Z'], n_samples)  # 3 categories

    # Create target (influenced by both numeric and categorical)
    y = (2 * X_numeric[:, 0] +
         3 * X_numeric[:, 1] +
         (cat_feature_1 == 'cat_A_10').astype(float) * 5 +
         np.random.randn(n_samples) * 0.5)

    # Create DataFrame
    df = pd.DataFrame(X_numeric, columns=[f'num_{i}' for i in range(n_numeric)])
    df['cat_1'] = cat_feature_1
    df['cat_2'] = cat_feature_2
    df['cat_3'] = cat_feature_3
    df['target'] = y

    feature_names = list(df.columns[:-1])
    cat_features = ['cat_1', 'cat_2', 'cat_3']

    return df, feature_names, cat_features, y


@pytest.fixture
def sample_categorical_classification_data():
    """Generate classification dataset with high-cardinality categorical features."""
    np.random.seed(42)
    n_samples = 500
    n_numeric = 5

    # Numeric features
    X_numeric = np.random.randn(n_samples, n_numeric)

    # High-cardinality categorical features
    cat_feature_1 = np.random.choice([f'cat_A_{i}' for i in range(100)], n_samples)
    cat_feature_2 = np.random.choice([f'cat_B_{i}' for i in range(50)], n_samples)
    cat_feature_3 = np.random.choice(['X', 'Y', 'Z'], n_samples)

    # Create binary target
    logits = (2 * X_numeric[:, 0] +
              3 * X_numeric[:, 1] +
              (cat_feature_1 == 'cat_A_10').astype(float) * 5)
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    # Create DataFrame
    df = pd.DataFrame(X_numeric, columns=[f'num_{i}' for i in range(n_numeric)])
    df['cat_1'] = cat_feature_1
    df['cat_2'] = cat_feature_2
    df['cat_3'] = cat_feature_3
    df['target'] = y

    feature_names = list(df.columns[:-1])
    cat_features = ['cat_1', 'cat_2', 'cat_3']

    return df, feature_names, cat_features, y


# ================================================================================================
# Special Dataset Fixtures
# ================================================================================================

@pytest.fixture
def sample_large_regression_data():
    """Generate larger dataset for SGD testing (better convergence)."""
    np.random.seed(42)
    n_samples = 2000  # Larger for SGD convergence
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(n_samples) * 0.5

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df, feature_names, y


@pytest.fixture
def sample_outlier_data():
    """Generate dataset with outliers for RANSAC and Huber testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(n_samples) * 0.5

    # Add outliers (10% of data)
    n_outliers = int(0.1 * n_samples)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    y[outlier_indices] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(10, 20, n_outliers)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df, feature_names, y


# ================================================================================================
# GPU Availability Fixtures
# ================================================================================================

@pytest.fixture
def check_lgb_gpu_available():
    """Check if LightGBM GPU is available."""
    try:
        import lightgbm as lgb

        # Try to create and fit a tiny GPU model
        model = lgb.LGBMClassifier(
            n_estimators=1,
            device_type="cuda",
            verbose=-1
        )
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

        return True
    except Exception as e:
        if "CUDA Tree Learner" in str(e):
            return False
        # Other errors might be real issues, so raise
        raise


@pytest.fixture
def check_gpu_available():
    """Check if CUDA GPU is available for testing."""
    try:
        from numba.cuda import is_available as is_cuda_available
        return is_cuda_available()
    except:
        return False


@pytest.fixture
def common_init_params():
    """Common init_common_params to suppress matplotlib figures in tests."""
    return {'show_perf_chart': False, 'show_fi': False}
