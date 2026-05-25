"""
Shared pytest fixtures for training module tests.

Note: autouse fixtures (cleanup_memory, suppress_convergence_warnings) live in
tests/conftest.py and apply to all test modules automatically.
"""

# Set matplotlib backend to 'Agg' BEFORE any matplotlib import to prevent plt.show() from blocking
import matplotlib
matplotlib.use('Agg')

# Pre-warm heavy optional dependencies BEFORE the per-test pytest-timeout
# clock starts. On Windows cold cache, the first import of
# ``flaml.default`` / ``mlframe.training.neural`` (lightning + torchmetrics)
# / networkx / matplotlib stylesheet pulls in tens of submodules off
# disk and easily exceeds 180 s on the first test that hits them — that
# blew up the fuzz-suite run on 2026-04-27. Lazy-import getters in
# trainer.py defer the actual classes; this prewarm here just touches
# disk so the OS file cache is hot before the first test starts. We
# swallow ImportError because some of these are optional extras.
try:
    import flaml.default  # noqa: F401
except (ImportError, OSError):
    pass
try:
    import mlframe.training.neural  # noqa: F401
except (ImportError, OSError):
    pass
try:
    # networkx is pulled in transitively by some sklearn / pyutilz paths;
    # touching it here lets the cold-cache wallclock land outside the
    # per-test timeout window.
    import networkx  # noqa: F401
    import networkx.algorithms  # noqa: F401
except (ImportError, OSError):
    pass

import pytest
import numpy as np
import pandas as pd
import polars as pl
import warnings
from pathlib import Path

from .synthetic import (
    make_categorical_classification_data,
    make_outlier_regression_data,
    make_simple_classification_data,
    make_simple_regression_data,
)


@pytest.fixture(scope="session")
def _session_monkeypatch(request):
    """Session-scoped monkeypatch helper (pytest's built-in monkeypatch is function-scoped).

    Provides safe per-session attribute patching that is auto-reverted on session teardown
    even if a worker yields mid-test under pytest-xdist (memory feedback rule about xdist safety).
    """
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def _force_cpu_training_defaults(_session_monkeypatch):
    """Force prefer_gpu_configs=False during tests.

    CI and many dev boxes have limited GPU memory; parameterized CatBoost
    test sweeps can accumulate CUDA allocations across the session and
    eventually OOM mid-suite (Windows fatal "bad allocation" out of the
    CatBoost cython layer is unrecoverable -- kills the pytest process).
    CPU fallback is slower per-test but keeps the suite runnable end-to-end.
    Individual GPU-specific tests remain opt-in via @pytest.mark.gpu.

    Uses session-scoped MonkeyPatch to guarantee restoration on teardown
    under pytest-xdist (raw try/finally on model_fields was leaking mutated
    defaults to sibling workers if a worker died mid-test).
    """
    from mlframe.training.configs import TrainingBehaviorConfig
    _field = TrainingBehaviorConfig.model_fields["prefer_gpu_configs"]
    _session_monkeypatch.setattr(_field, "default", False)
    TrainingBehaviorConfig.model_rebuild(force=True)
    yield
    TrainingBehaviorConfig.model_rebuild(force=True)


@pytest.fixture(scope="session", autouse=True)
def _prewarm_numba_once():
    """Pre-warm numba JIT cache ONCE per session.

    Under pytest-xdist multiple workers race on .nbc/.nbi cache files at first
    compile -> lock contention and occasional Windows access violations; the
    session-scoped prewarm forces a single compile pass while workers are still
    serial-importing the conftest. Serial runs ALSO benefit because the first
    test hitting numba kernels otherwise pays the cold-compile cost inside its
    own per-test timeout window; doing the compile in conftest setup amortises
    it across the session.

    Skip only on the xdist coordinator (PYTEST_XDIST_WORKER absent AND
    PYTEST_XDIST_TESTRUNUID present) where each worker will run this fixture
    itself; otherwise the coordinator would double-warm.

    Skip env: MLFRAME_SKIP_NUMBA_PREWARM=1 disables the prewarm entirely.
    Used by the pre-commit mlframe-meta-tests hook (meta-tests are AST /
    config / structure checks that never touch numba kernels) so a CUDA
    driver init hang in conftest can't block local commits. CI runs leave
    the env unset so the full prewarm fires there.
    """
    import os
    if os.environ.get("MLFRAME_SKIP_NUMBA_PREWARM", "").strip() in ("1", "true", "True", "yes"):
        yield
        return
    is_xdist_coordinator = (
        os.environ.get("PYTEST_XDIST_TESTRUNUID") is not None
        and not os.environ.get("PYTEST_XDIST_WORKER")
    )
    if is_xdist_coordinator:
        yield
        return
    try:
        from mlframe.metrics.core import prewarm_numba_cache
        prewarm_numba_cache()
    except Exception:
        pass
    yield


_SESSION_FIXTURE_SHAPES: dict[str, tuple] = {}
_SESSION_FIXTURE_REFS: dict[str, object] = {}


@pytest.fixture(scope="session", autouse=True)
def _session_fixture_immutability_sensor():
    """Tripwire for unintended mutation of session-scope DataFrame fixtures.

    The session-scope ``sample_*_data`` fixtures share one DataFrame across hundreds of tests; mutating one
    (e.g. ``df["new_col"] = ...``) silently changes the fixture for every later consumer. This sensor records the
    fixture's shape signature (rows / cols / dtype tuple) at first use and re-checks at teardown. A signature
    mismatch surfaces the leak with the fixture name, even if the mutating test itself passed.

    Cheap (only re-reads tuples at session end) and informational (logs to stderr; does not fail the run because
    a mutation may have legitimately occurred in a test that explicitly took a ``.copy()`` after fixture access)."""
    yield
    leaked = []
    for name, original in _SESSION_FIXTURE_SHAPES.items():
        ref = _SESSION_FIXTURE_REFS.get(name)
        if ref is None:
            continue
        current = _df_shape_signature(ref)
        if current != original:
            leaked.append((name, original, current))
    if leaked:
        import sys
        sys.stderr.write(
            "[session-fixture-immutability-sensor] DETECTED MUTATION of session-scope fixtures:\n"
        )
        for name, before, after in leaked:
            sys.stderr.write(f"  {name}: shape/cols/dtypes before={before} after={after}\n")
        sys.stderr.write(
            "[session-fixture-immutability-sensor] Tests should .copy() before mutating. "
            "See tests/training/conftest.py docstrings on sample_regression_data / sample_classification_data.\n"
        )


def _df_shape_signature(df) -> tuple:
    """Cheap shape+dtypes+column-names tuple used by the session-fixture immutability sensor at teardown.

    Catches gross mutation (rename, dtype flip, row insertion) without the cost of hashing every cell. A test that
    silently mutates a session-scope fixture would shift one of: ``df.shape``, ``df.columns.tolist()``, or the
    dtypes tuple."""
    try:
        return (tuple(df.shape), tuple(df.columns.tolist()), tuple(str(t) for t in df.dtypes.tolist()))
    except Exception:
        return (None, None, None)


@pytest.fixture(scope="session")
def sample_regression_data():
    """Synthetic regression dataset; session-scoped (deterministic via default_rng).

    DO NOT MUTATE. Tests that need to add / rename / dtype-cast columns must do so on a ``.copy()`` first. The
    session-scope sharing depends on stability across all consuming tests; mutating in-place silently changes the
    fixture for every later consumer."""
    df, feature_names, y = make_simple_regression_data(n_samples=1000, n_features=10, seed=42)
    _SESSION_FIXTURE_SHAPES["sample_regression_data"] = _df_shape_signature(df)
    _SESSION_FIXTURE_REFS["sample_regression_data"] = df
    return df, feature_names, y


@pytest.fixture(scope="session")
def sample_classification_data():
    """Synthetic binary classification dataset; session-scoped. DO NOT MUTATE (see ``sample_regression_data``)."""
    df, feature_names, y = make_simple_classification_data(n_samples=1000, n_features=10, seed=42)
    _SESSION_FIXTURE_SHAPES["sample_classification_data"] = _df_shape_signature(df)
    _SESSION_FIXTURE_REFS["sample_classification_data"] = df
    return df, feature_names, y


@pytest.fixture(scope="session")
def sample_polars_data(sample_regression_data):
    """Convert sample regression data to a Polars DataFrame; session-scoped."""
    df, feature_names, y = sample_regression_data
    pl_df = pl.from_pandas(df)
    return pl_df, feature_names, y


@pytest.fixture(scope="session")
def sample_timeseries_data():
    """Synthetic time-series dataset; session-scoped (deterministic Generator)."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_features = 5

    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1h')
    X = rng.standard_normal((n_samples, n_features))
    y = np.sin(np.arange(n_samples) / 50) + rng.standard_normal(n_samples) * 0.1

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
# Categorical Data Fixtures
# ================================================================================================

@pytest.fixture(scope="session")
def sample_categorical_data():
    """High-cardinality categorical-feature regression-style dataset; session-scoped.

    Returns (df, feature_names, cat_features, y_continuous). Built via deterministic
    Generator; no mutation of global numpy RNG state.
    """
    rng = np.random.default_rng(42)
    n_samples = 500
    n_numeric = 5
    X_numeric = rng.standard_normal((n_samples, n_numeric))
    cat_1 = rng.choice([f"cat_A_{i}" for i in range(100)], n_samples)
    cat_2 = rng.choice([f"cat_B_{i}" for i in range(50)], n_samples)
    cat_3 = rng.choice(["X", "Y", "Z"], n_samples)
    y = (
        2 * X_numeric[:, 0]
        + 3 * X_numeric[:, 1]
        + (cat_1 == "cat_A_10").astype(float) * 5
        + rng.standard_normal(n_samples) * 0.5
    )
    df = pd.DataFrame(X_numeric, columns=[f"num_{i}" for i in range(n_numeric)])
    df["cat_1"] = cat_1
    df["cat_2"] = cat_2
    df["cat_3"] = cat_3
    df["target"] = y
    feature_names = list(df.columns[:-1])
    cat_features = ["cat_1", "cat_2", "cat_3"]
    return df, feature_names, cat_features, y


@pytest.fixture(scope="session")
def sample_categorical_classification_data():
    """Classification with high-cardinality categoricals; session-scoped via Generator."""
    return make_categorical_classification_data(n_samples=500, n_numeric=5, seed=42)


# ================================================================================================
# Special Dataset Fixtures
# ================================================================================================

@pytest.fixture(scope="session")
def sample_large_regression_data():
    """Larger regression dataset for SGD convergence; session-scoped."""
    return make_simple_regression_data(n_samples=2000, n_features=10, seed=42)


@pytest.fixture(scope="session")
def sample_outlier_data():
    """Regression with 10% outliers for RANSAC and Huber tests; session-scoped."""
    return make_outlier_regression_data(n_samples=500, n_features=10, seed=42)


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
def check_catboost_gpu_available():
    """Check whether CatBoost specifically can use GPU.

    `check_gpu_available` is too permissive — it only verifies that *some*
    CUDA device exists via numba. CatBoost ships its own GPU runtime that
    may not be installed on all dev boxes (error observed:
    ``Environment for task type [GPU] not found``). Gate CatBoost-GPU tests
    on this check instead.
    """
    try:
        from catboost.utils import get_gpu_device_count
        return get_gpu_device_count() > 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def common_init_params():
    """Common ReportingConfig to suppress matplotlib figures in tests.

    Returns a typed ReportingConfig (was a dict pre-2026-04-27). Tests that
    use this fixture should pass it as ``reporting_config=common_init_params``,
    not as the deleted ``init_common_params=`` legacy kwarg.
    """
    from mlframe.training.configs import ReportingConfig
    return ReportingConfig(show_perf_chart=False, show_fi=False)


@pytest.fixture(scope="session")
def fast_iterations():
    """Low iteration count for fast test execution.

    Use this to override the default 5000 iterations in TreeModelConfig.
    With 10 iterations, early_stopping_rounds will be ~3 instead of ~1666.
    """
    return 10


@pytest.fixture(scope="session")
def fast_config_override(fast_iterations):
    """Config params override for fast test execution."""
    return {'iterations': fast_iterations}


@pytest.fixture(scope="session")
def cpu_only_hyperparams(fast_iterations):
    """Shared CPU-only kwargs for cb / xgb / lgb / mlp boosters.

    Replaces the repeated inline blocks in test_all_models.py and
    test_catboost_polars.py that set task_type=CPU / device=cpu / device_type=cpu
    per booster family. Pair with `fast_iterations` for fast test execution.
    """
    return {
        "iterations": fast_iterations,
        "cb_kwargs": {"task_type": "CPU"},
        "xgb_kwargs": {"device": "cpu"},
        "lgb_kwargs": {"device_type": "cpu"},
        "mlp_kwargs": {"trainer_params": {"devices": 1}},
    }


@pytest.fixture(scope="session")
def trained_suite_regression(sample_regression_data, common_init_params, fast_iterations):
    """One trained suite per session for regression target.

    Heavy fixture: runs `train_mlframe_models_suite` once with a tiny iteration
    budget so that downstream behaviour tests that only need to inspect the
    returned suite object (e.g. attribute presence, metric dict shape) can
    consume it without re-training.
    """
    from mlframe.training.core import train_mlframe_models_suite
    from .shared import SimpleFeaturesAndTargetsExtractor

    df, feature_names, _y = sample_regression_data
    extractor = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
    suite = train_mlframe_models_suite(
        df=df,
        target_name="test_target",
        model_name="trained_suite_regression",
        features_and_targets_extractor=extractor,
        reporting_config=common_init_params,
        hyperparams_config={
            "iterations": fast_iterations,
            "cb_kwargs": {"task_type": "CPU"},
            "xgb_kwargs": {"device": "cpu"},
            "lgb_kwargs": {"device_type": "cpu"},
        },
    )
    return suite


@pytest.fixture(scope="session")
def trained_suite_binary(sample_classification_data, common_init_params, fast_iterations):
    """One trained suite per session for binary-classification target.

    Companion to ``trained_suite_regression`` (extends W4E shared-fixture work).
    Tests that only inspect the returned suite object (attribute presence,
    metric dict shape, predict_proba round-trip) should consume this fixture
    instead of re-fitting per test.
    """
    from mlframe.training.core import train_mlframe_models_suite
    from .shared import SimpleFeaturesAndTargetsExtractor

    df, feature_names, *_, _y = sample_classification_data
    extractor = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    suite = train_mlframe_models_suite(
        df=df,
        target_name="test_target",
        model_name="trained_suite_binary",
        features_and_targets_extractor=extractor,
        reporting_config=common_init_params,
        hyperparams_config={
            "iterations": fast_iterations,
            "cb_kwargs": {"task_type": "CPU"},
            "xgb_kwargs": {"device": "cpu"},
            "lgb_kwargs": {"device_type": "cpu"},
        },
    )
    return suite
