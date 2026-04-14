"""Tests for mlframe.utils."""

import os
import random

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.utils import (
    get_full_classifier_name,
    get_pipeline_last_element,
    set_random_seed,
)


# --- set_random_seed ---

def test_set_random_seed_deterministic_numpy():
    set_random_seed(123)
    a = np.random.random(5)
    set_random_seed(123)
    b = np.random.random(5)
    np.testing.assert_array_equal(a, b)


def test_set_random_seed_deterministic_stdlib():
    set_random_seed(77)
    a = [random.random() for _ in range(5)]
    set_random_seed(77)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_random_seed_hash_seed():
    set_random_seed(42, set_hash_seed=True)
    assert os.environ.get("PYTHONHASHSEED") == "42"


# --- get_pipeline_last_element ---

def test_get_pipeline_last_element_returns_last():
    ridge = Ridge()
    pipe = Pipeline([("scaler", StandardScaler()), ("model", ridge)])
    assert get_pipeline_last_element(pipe) is ridge


def test_get_pipeline_last_element_single_step():
    ridge = Ridge()
    pipe = Pipeline([("model", ridge)])
    assert get_pipeline_last_element(pipe) is ridge


# --- get_full_classifier_name ---

def test_classifier_name_regular_estimator():
    assert get_full_classifier_name(Ridge()) == "Ridge"


def test_classifier_name_pipeline():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    assert get_full_classifier_name(pipe) == "pipe[Ridge]"


def test_classifier_name_dummy():
    clf = DummyClassifier(strategy="most_frequent")
    assert get_full_classifier_name(clf) == "DummyClassifier[most_frequent]"


# --- is_cuda_available ---

def test_is_cuda_available():
    numba = pytest.importorskip("numba")
    from mlframe.utils import is_cuda_available

    result = is_cuda_available()
    assert isinstance(result, bool)


# --- check_cpu_flag ---

def test_check_cpu_flag():
    pytest.importorskip("cpuinfo")
    from mlframe.utils import check_cpu_flag

    result = check_cpu_flag("sse2")
    assert isinstance(result, bool)
