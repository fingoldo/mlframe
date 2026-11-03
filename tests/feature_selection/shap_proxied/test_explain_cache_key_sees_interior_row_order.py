"""The SHAP disk-cache key changes when rows are reordered inside the frame's middle, so a hit cannot attach phi to the wrong samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain_cache import _open_shap_disk_cache


def _key(tmp_path, X, y):
    """Cache key for (X, y) under fixed settings."""
    _cache, key, _hit = _open_shap_disk_cache(
        tmp_path, LinearRegression(), X, y, np.random.default_rng(0),
        classification=False, out_of_fold=False, n_splits=3, n_models=1, config_jitter=False, return_variance=False,
        return_per_fold_phi_mean=False, shap_backend="auto", n_estimators_cap=None, inner_n_jobs_cap=False,
    )
    return key


def test_swapping_two_middle_rows_changes_the_key(tmp_path):
    """Rows 500 and 501 swapped (together with y): shape, ends and column sums are unchanged, the key must still differ."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list("abc"))
    y = rng.normal(size=1000)
    order = np.arange(1000)
    order[[500, 501]] = order[[501, 500]]
    X2, y2 = X.iloc[order].reset_index(drop=True), y[order]
    assert _key(tmp_path, X, y) is not None
    assert _key(tmp_path, X, y) != _key(tmp_path, X2, y2)


def test_identical_inputs_share_a_key(tmp_path):
    """Same data, same key (negative control: the cache still works)."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=list("ab"))
    y = rng.normal(size=200)
    assert _key(tmp_path, X, y) == _key(tmp_path, X.copy(), y.copy())
