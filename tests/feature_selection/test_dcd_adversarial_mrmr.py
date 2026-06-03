"""Adversarial MRMR-level edge cases for DCD clustering (2026-06-03):
determinism, pickle/transform parity (stresses the new dcd_swap_npermutations
state + the post-swap _fn_arr_cached reset + recipe replay), and degenerate
inputs (perfect duplicates, tiny n, NaN columns). Failures are prod bugs.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _dup_cluster_frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame({
        "strong": other,
        "dup_a": latent + 0.01 * rng.standard_normal(n),
        "dup_b": latent + 0.01 * rng.standard_normal(n),
        "dup_c": latent + 0.01 * rng.standard_normal(n),
        "noise": rng.standard_normal(n),
    })
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _fit(X, y, **kw):
    from mlframe.feature_selection.filters.mrmr import MRMR
    base = dict(dcd_enable=True, dcd_tau_cluster=0.5, dcd_cluster_size_threshold=2,
                verbose=0, random_seed=0)
    base.update(kw)
    return MRMR(**base).fit(X, y)


def test_dcd_fit_is_deterministic():
    X, y = _dup_cluster_frame()
    m1, m2 = _fit(X, y), _fit(X, y)
    assert list(m1.get_feature_names_out()) == list(m2.get_feature_names_out())
    s1 = (m1.dcd_ or {}).get("swap_log", [])
    s2 = (m2.dcd_ or {}).get("swap_log", [])
    assert [e.get("aggregate_name") for e in s1] == [e.get("aggregate_name") for e in s2]
    assert (m1.dcd_ or {}).get("n_swaps") == (m2.dcd_ or {}).get("n_swaps")


def test_dcd_pickle_transform_parity():
    # fit -> pickle round-trip -> transform must be bit-identical. Stresses the
    # new swap_npermutations field, _fn_arr_cached (excluded/rebuilt), and the
    # aggregate recipe replay surviving serialization.
    X, y = _dup_cluster_frame()
    Xte = X.iloc[:400]
    m = _fit(X, y)
    out0 = np.asarray(m.transform(Xte))
    m2 = pickle.loads(pickle.dumps(m))
    out1 = np.asarray(m2.transform(Xte))
    assert out0.shape == out1.shape
    assert np.array_equal(np.nan_to_num(out0), np.nan_to_num(out1)), (
        "transform output changed across pickle round-trip"
    )


def test_dcd_fit_on_perfect_duplicates_no_crash():
    rng = np.random.default_rng(3)
    n = 1200
    col = rng.standard_normal(n)
    X = pd.DataFrame({f"d{i}": col for i in range(4)})  # 4 byte-identical columns
    X["sig"] = (col > 0) * 1.0 + 0.5 * rng.standard_normal(n)
    y = pd.Series((col > 0).astype(int))
    m = _fit(X, y)
    assert len(list(m.get_feature_names_out())) >= 1


def test_dcd_fit_tiny_n_no_crash():
    rng = np.random.default_rng(4)
    n = 40
    z = rng.standard_normal(n)
    X = pd.DataFrame({"a": z + 0.05 * rng.standard_normal(n),
                      "b": z + 0.05 * rng.standard_normal(n),
                      "c": rng.standard_normal(n)})
    y = pd.Series((z > 0).astype(int))
    m = _fit(X, y)
    assert len(list(m.get_feature_names_out())) >= 1


def test_dcd_fit_with_nan_columns_no_crash():
    rng = np.random.default_rng(5)
    n = 1000
    z = rng.standard_normal(n)
    a = z + 0.05 * rng.standard_normal(n)
    a[::50] = np.nan  # scattered NaNs in a cluster member
    X = pd.DataFrame({"a": a, "b": z + 0.05 * rng.standard_normal(n),
                      "c": z + 0.05 * rng.standard_normal(n),
                      "noise": rng.standard_normal(n)})
    y = pd.Series((z > 0).astype(int))
    m = _fit(X, y)
    assert len(list(m.get_feature_names_out())) >= 1
