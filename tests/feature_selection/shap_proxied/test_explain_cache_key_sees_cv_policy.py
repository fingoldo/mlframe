"""The SHAP disk-cache key and rng replay honour the shared CV policy, the estimator class and string-column boundaries."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.feature_selection.cv_policy import CVPolicy
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain_cache as cache_mod
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain_cache import _open_shap_disk_cache, _store_shap_cache, content_digest
from mlframe.utils.log_throttle import reset_throttle_counts


def _open(tmp_path, X, y, *, template=None, cv_policy=None, rng=None, out_of_fold=True):
    """Open the cache for (X, y) with fixed settings and return ``(cache, key, hit)``."""
    return _open_shap_disk_cache(
        tmp_path, template if template is not None else LinearRegression(), X, y, rng if rng is not None else np.random.default_rng(0),
        classification=False, out_of_fold=out_of_fold, n_splits=3, n_models=2, config_jitter=False, return_variance=False,
        return_per_fold_phi_mean=False, shap_backend="auto", n_estimators_cap=None, inner_n_jobs_cap=False, cv_policy=cv_policy,
    )


def _data(n=60):
    """A small regression frame."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc")), rng.normal(size=n)


def test_key_differs_when_a_temporal_policy_changes_the_fold_partition(tmp_path):
    """A temporal policy partitions rows into time blocks, so it must not share a key with the shuffled-KFold call."""
    X, y = _data()
    policy = CVPolicy("temporal", "test", timestamps=np.arange(len(X)))
    assert _open(tmp_path, X, y)[1] != _open(tmp_path, X, y, cv_policy=policy)[1]


def test_key_differs_between_two_different_policies(tmp_path):
    """Two temporal policies with different row orders give different folds and different keys."""
    X, y = _data()
    a = CVPolicy("temporal", "a", timestamps=np.arange(len(X)))
    b = CVPolicy("temporal", "b", timestamps=np.arange(len(X))[::-1].copy())
    assert _open(tmp_path, X, y, cv_policy=a)[1] != _open(tmp_path, X, y, cv_policy=b)[1]


def test_key_differs_between_estimator_classes_with_equal_params(tmp_path):
    """Two estimator classes exposing the same get_params must not share a key."""

    class Sub(LinearRegression):
        """Subclass with identical params."""

    X, y = _data()
    assert _open(tmp_path, X, y, template=LinearRegression())[1] != _open(tmp_path, X, y, template=Sub())[1]


def test_cache_hit_replays_the_rng_the_miss_path_consumes_under_a_policy(tmp_path):
    """Under a policy the OOF stage draws no splitter seed, so the hit replay must leave the rng where the miss path would."""
    X, y = _data()
    policy = CVPolicy("temporal", "test", timestamps=np.arange(len(X)))
    cache, key, _ = _open(tmp_path, X, y, cv_policy=policy, rng=np.random.default_rng(7))
    cache.put(key, "payload")
    replayed = np.random.default_rng(7)
    assert _open(tmp_path, X, y, cv_policy=policy, rng=replayed)[2] == "payload"
    expected = np.random.default_rng(7)
    for _ in range(3 * 2):  # n_splits * n_models fold seeds, no splitter seed
        expected.integers(0, 2**31 - 1)
    assert replayed.integers(0, 2**31 - 1) == expected.integers(0, 2**31 - 1)


def test_object_column_digest_sees_where_the_strings_split():
    """['ab', 'c'] and ['a', 'bc'] are different columns."""
    a = pd.DataFrame({"s": np.array(["ab", "c"], dtype=object)})
    b = pd.DataFrame({"s": np.array(["a", "bc"], dtype=object)})
    assert content_digest(a) != content_digest(b)


def test_failed_cache_write_warns(caplog):
    """A cache.put failure is a visible WARNING, not a debug line."""

    class Boom:
        """Cache whose put always fails."""

        def put(self, key, value):
            """Fail."""
            raise OSError("disk full")

    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=cache_mod.logger.name):
        assert _store_shap_cache(Boom(), "k", 5) == 5
    assert any("cache write failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
