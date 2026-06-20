"""Production-surface tests for the promoted HybridSelector (mlframe.feature_selection.hybrid_selector).

Asserts the public import paths resolve to one class, the sklearn-style selector interface (fit / transform /
get_support / get_feature_names_out / selected_features_ / n_features_in_), and that a fitted estimator pickles
small (the __getstate__ drop of the transient fit-time X_aug/y) and still transforms after a round-trip.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest


def _data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-(z @ np.array([1.5, -1.2, 1.0, 0.9]))))).astype(int))
    cols = {f"f{i}": z[:, i] for i in range(4)}
    cols.update({f"n{i}": rng.standard_normal(n) for i in range(6)})
    return pd.DataFrame(cols), y


def test_public_import_paths_resolve_to_one_class():
    from mlframe.feature_selection import HybridSelector as H_pkg
    from mlframe.feature_selection.hybrid_selector import HybridSelector as H_prod
    from mlframe.feature_selection._benchmarks.fs_hybrid.hybrid_selector import HybridSelector as H_bench
    assert H_pkg is H_prod is H_bench  # package export == production module == benchmark re-export


def test_selector_interface_and_defaults():
    import inspect
    from mlframe.feature_selection.hybrid_selector import HybridSelector
    p = inspect.signature(HybridSelector.__init__).parameters
    # the shipped, bench-validated config
    assert p["vote"].default == 1 and p["use_fe"].default is True
    assert p["anchor_fe"].default is False and p["boruta_driver"].default == "gini"
    assert p["fi_guard"].default is False


@pytest.mark.timeout(900)
def test_fit_transform_support_and_pickle_roundtrip():
    from mlframe.feature_selection.hybrid_selector import HybridSelector
    X, y = _data()
    h = HybridSelector(use_fe=False, random_state=0).fit(X, y)   # use_fe=False keeps this test fast + deterministic
    # fitted attributes
    assert h.n_features_in_ == X.shape[1] and h.feature_names_in_ == list(X.columns)
    assert h.selected_features_ == list(h.raw_selected_) and len(h.selected_features_) >= 1
    # get_support mask aligns with the original columns; get_feature_names_out matches the selection
    mask = h.get_support()
    assert mask.shape == (X.shape[1],) and mask.sum() == len([c for c in h.raw_selected_ if c in X.columns])
    assert list(h.get_feature_names_out()) == list(h.raw_selected_)
    # transform returns exactly the selected columns
    Z = h.transform(X)
    assert list(Z.columns) == [c for c in h.raw_selected_ if c in Z.columns] and Z.shape[0] == X.shape[0]
    # pickle drops the transient fit data (__getstate__) and still transforms after reload
    h2 = pickle.loads(pickle.dumps(h))
    assert not hasattr(h2, "_Xaug_") and not hasattr(h2, "_y_")
    assert list(h2.transform(X).columns) == list(Z.columns)
