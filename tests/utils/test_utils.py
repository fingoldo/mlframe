"""Tests for mlframe.utils.misc."""

import os
import random

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.utils.misc import (
    get_full_classifier_name,
    get_pipeline_last_element,
    set_random_seed,
)


# --- set_random_seed ---


def test_set_random_seed_deterministic_numpy():
    """Set random seed deterministic numpy."""
    set_random_seed(123)
    a = np.random.random(5)
    set_random_seed(123)
    b = np.random.random(5)
    np.testing.assert_array_equal(a, b)


def test_set_random_seed_deterministic_stdlib():
    """Set random seed deterministic stdlib."""
    set_random_seed(77)
    a = [random.random() for _ in range(5)]  # nosec B311 -- deterministic-seed test PRNG, not used for security/crypto purposes
    set_random_seed(77)
    b = [random.random() for _ in range(5)]  # nosec B311 -- deterministic-seed test PRNG, not used for security/crypto purposes
    assert a == b


def test_set_random_seed_hash_seed():
    """Set random seed hash seed."""
    set_random_seed(42, set_hash_seed=True)
    assert os.environ.get("PYTHONHASHSEED") == "42"


# --- get_pipeline_last_element ---


def test_get_pipeline_last_element_returns_last():
    """Get pipeline last element returns last."""
    ridge = Ridge()
    pipe = Pipeline([("scaler", StandardScaler()), ("model", ridge)])
    assert get_pipeline_last_element(pipe) is ridge


def test_get_pipeline_last_element_single_step():
    """Get pipeline last element single step."""
    ridge = Ridge()
    pipe = Pipeline([("model", ridge)])
    assert get_pipeline_last_element(pipe) is ridge


# --- get_full_classifier_name ---


def test_classifier_name_regular_estimator():
    """Classifier name regular estimator."""
    assert get_full_classifier_name(Ridge()) == "Ridge"


def test_classifier_name_pipeline():
    """Classifier name pipeline."""
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    assert get_full_classifier_name(pipe) == "pipe[Ridge]"


def test_classifier_name_dummy():
    """Classifier name dummy."""
    clf = DummyClassifier(strategy="most_frequent")
    assert get_full_classifier_name(clf) == "DummyClassifier[most_frequent]"


# --- is_cuda_available ---


def test_is_cuda_available():
    """Is cuda available."""
    pytest.importorskip("numba")
    from mlframe.utils.misc import is_cuda_available

    result = is_cuda_available()
    assert isinstance(result, bool)


# --- check_cpu_flag ---


def test_check_cpu_flag():
    """Check cpu flag."""
    pytest.importorskip("cpuinfo")
    from mlframe.utils.misc import check_cpu_flag

    result = check_cpu_flag("sse2")
    assert isinstance(result, bool)
