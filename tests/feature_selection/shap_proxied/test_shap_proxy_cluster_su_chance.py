"""Chance-corrected SU clustering: independent high-cardinality columns must not merge; true duplicates and low-card pairs must."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import cluster_correlated_features_su


def _cols(n=20_000, K=2_000, seed=0):
    """Two independent high-cardinality columns plus a random recoding of the first."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, K, n)
    return {"a": a, "b": rng.integers(0, K, n), "a_recode": rng.permutation(K)[a]}


def _lab(cols, **kw):
    """Cluster the columns by symmetric uncertainty on CPU and return the labels."""
    return cluster_correlated_features_su(cols, feature_names=list(cols), use_gpu=False, **kw)


def test_independent_high_card_columns_not_merged_but_plugin_merged_them():
    """Independent high card columns not merged but plugin merged them."""
    cols = _cols()
    raw = _lab(cols, chance_correct=False)
    assert raw[0] == raw[1], "premise: plug-in SU falsely merges independent high-card columns"
    lab = _lab(cols)
    assert lab[0] != lab[1] and lab[2] != lab[1]


def test_true_high_card_recoding_still_merged():
    """True high card recoding still merged."""
    lab = _lab(_cols())
    assert lab[0] == lab[2]


def test_low_card_labels_bit_identical_to_plugin():
    """Low card labels bit identical to plugin."""
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.integers(0, 10, n)
    cols = {"x": x, "y": np.where(rng.random(n) < 0.1, rng.integers(0, 10, n), x), "z": rng.integers(0, 10, n)}
    assert np.array_equal(_lab(cols), _lab(cols, chance_correct=False))
    assert _lab(cols)[0] == _lab(cols)[1]


def test_near_unique_columns_unlinked():
    """Near unique columns unlinked."""
    rng = np.random.default_rng(2)
    n = 5_000
    a = rng.permutation(n)
    cols = {"a": a, "b": rng.permutation(n)}
    assert _lab(cols)[0] != _lab(cols)[1]
