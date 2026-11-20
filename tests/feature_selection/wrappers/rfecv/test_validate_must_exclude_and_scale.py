"""Regression tests for RFECV input sanitising: must_exclude ordering and scale-free zero-variance detection."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers.rfecv._validate import _sanitize_X_inputs


def _self(**kw):
    """Minimal RFECV stand-in carrying the attributes the sanitiser reads."""
    base = dict(max_nfeatures=None, max_runtime_mins=None, cv=3, feature_groups=None, must_exclude=None, must_include=None,
                leakage_corr_threshold=None, verbose=0, must_exclude_strict=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_must_exclude_of_constant_column_does_not_raise():
    """A must_exclude name that is also constant is excluded, not reported as missing."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100), "const": 1.0})
    out = _sanitize_X_inputs(_self(must_exclude=["const"]), X, rng.normal(size=100))
    assert list(out.columns) == ["a"]


def test_must_exclude_column_does_not_swallow_its_duplicate():
    """An excluded column must not cause its exact duplicate to be dropped as the copy."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=100)
    X = pd.DataFrame({"a": v, "b": v.copy(), "c": rng.normal(size=100)})
    out = _sanitize_X_inputs(_self(must_exclude=["a"]), X, rng.normal(size=100))
    assert list(out.columns) == ["b", "c"]


def test_small_scale_informative_column_is_not_dropped_as_zero_variance():
    """A genuinely varying column in tiny units (std 1e-7) is kept."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"tiny": rng.normal(size=100) * 1e-7, "b": rng.normal(size=100)})
    out = _sanitize_X_inputs(_self(), X, rng.normal(size=100))
    assert "tiny" in out.columns


def test_relabeled_categorical_is_not_dropped_as_exact_duplicate():
    """Two categoricals with different labels but the same code pattern are distinct features."""
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 3, size=100)
    X = pd.DataFrame({"a": np.array(["x", "y", "z"])[idx], "b": np.array(["p", "q", "r"])[idx], "c": rng.normal(size=100)})
    out = _sanitize_X_inputs(_self(), X, rng.normal(size=100))
    assert {"a", "b"} <= set(out.columns)
