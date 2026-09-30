"""biz_val: chance-corrected SU removes false merges of independent high-cardinality columns without losing true merges.

Measured (n=50k, K=10k, 5 seeds): plug-in SU falsely merges 7/7 unrelated pairs and chance-corrected SU 0/7; true pairs (recoding, noisy copy,
low-card correlated) missed 0/4 for both.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import cluster_correlated_features_su


def _build(seed, n=50_000, K=10_000):
    """Build high-cardinality ID-like columns with a recoded duplicate, a noisy copy and low-cardinality pair, plus the set of truly related pairs."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, K, n)
    noisy = np.where(rng.random(n) < 0.1, rng.integers(0, K, n), a)
    lo = rng.integers(0, 8, n)
    cols = {"i0": a, "i1": rng.integers(0, K, n), "i2": rng.integers(0, K, n), "dup": rng.permutation(K)[a], "noisy": noisy,
            "lo0": lo, "lo1": np.where(rng.random(n) < 0.05, rng.integers(0, 8, n), lo)}
    true = {("i0", "dup"), ("i0", "noisy"), ("dup", "noisy"), ("lo0", "lo1")}
    return cols, true


def _rates(chance):
    """Return the summed false-merge and missed-merge counts over five seeds for the chance-correction setting."""
    fm = mm = 0
    for seed in range(5):
        cols, true = _build(seed)
        names = list(cols)
        lab = cluster_correlated_features_su(cols, feature_names=names, use_gpu=False, chance_correct=chance)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                same = lab[i] == lab[j]
                if (names[i], names[j]) in true:
                    mm += not same
                else:
                    fm += same
    return fm, mm


def test_biz_val_cluster_su_chance_false_merges_eliminated_no_missed_merges():
    """Biz val cluster su chance false merges eliminated no missed merges."""
    old_fm, _ = _rates(False)
    fm, mm = _rates(True)
    assert old_fm >= 25, old_fm
    assert fm == 0
    assert mm == 0
