"""Regression sensors for the iter81 MDL best-split optimization.

`_mdl_bin_edges` used to recompute two `np.bincount` entropies per candidate split index (O(n^2) per feature). iter81 replaced that inner
loop with the single-pass `_best_mdl_split_kernel` (running prefix class counts). These tests pin (1) the kernel exists and is routed
through, and (2) the produced edges are bit-identical to the reference O(n^2) double-bincount logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer import mdl_binning_pairwise as M


def _entropy_multi_ref(y_subset, n_classes):
    if y_subset.size == 0:
        return 0.0
    counts = np.bincount(y_subset.astype(np.int32), minlength=n_classes).astype(np.float64)
    p = counts / counts.sum()
    e = 0.0
    for pi in p:
        if pi > 0:
            e -= pi * np.log2(pi)
    return e


def _mdl_bin_edges_ref(x, y_class, n_classes, max_bins=8, min_size=20):
    """Pure-Python replica of the pre-iter81 O(n^2) best-split logic (the baseline this optimization must match bit-for-bit)."""
    edges = []
    n = x.size
    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    y_sorted = y_class[order]
    H_S = _entropy_multi_ref(y_sorted, n_classes)
    if H_S < 1e-6 or n < 2 * min_size:
        return edges
    best_gain = -1.0
    best_idx = -1
    best_thresh = None
    for i in range(min_size, n - min_size):
        if i > 0 and x_sorted[i] == x_sorted[i - 1]:
            continue
        E_left = _entropy_multi_ref(y_sorted[:i], n_classes)
        E_right = _entropy_multi_ref(y_sorted[i:], n_classes)
        weighted = (i / n) * E_left + ((n - i) / n) * E_right
        gain = H_S - weighted
        if gain > best_gain:
            best_gain = gain
            best_idx = i
            best_thresh = (x_sorted[i - 1] + x_sorted[i]) / 2.0
    if best_idx < 0:
        return edges
    k = n_classes
    delta = np.log2(3**k - 2) - (
        k * H_S
        - 2 * _entropy_multi_ref(y_sorted[:best_idx], n_classes) * (best_idx / n)
        - (n - best_idx) / n * 2 * _entropy_multi_ref(y_sorted[best_idx:], n_classes)
    )
    if best_gain * n < np.log2(n - 1) + delta:
        return edges
    edges.append(float(best_thresh))
    return edges


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_mdl_bin_edges_bit_identical_to_reference(mode):
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.standard_normal(n).astype(np.float32)
    yc = (x * 0.8 + 0.4 * rng.standard_normal(n)).astype(np.float32)
    if mode == "binary":
        yclass = (yc > np.median(yc)).astype(np.int32)
        ncl = 2
    else:
        qs = np.quantile(yc, [0.2, 0.4, 0.6, 0.8])
        yclass = np.digitize(yc, qs).astype(np.int32)
        ncl = 5
    got = M._mdl_bin_edges(x, yclass, ncl, max_bins=8)
    ref = _mdl_bin_edges_ref(x, yclass, ncl, max_bins=8)
    assert np.array_equal(np.array(got), np.array(ref)), f"{got} != {ref}"


def test_mdl_bin_edges_with_ties_matches_reference():
    rng = np.random.default_rng(11)
    n = 2500
    # Heavy ties: a low-cardinality x so the equal-x skip path is exercised on both sides.
    x = rng.integers(0, 8, size=n).astype(np.float32)
    yc = (x + rng.standard_normal(n)).astype(np.float32)
    qs = np.quantile(yc, [0.2, 0.4, 0.6, 0.8])
    yclass = np.digitize(yc, qs).astype(np.int32)
    got = M._mdl_bin_edges(x, yclass, 5, max_bins=8)
    ref = _mdl_bin_edges_ref(x, yclass, 5, max_bins=8)
    assert np.array_equal(np.array(got), np.array(ref))


def test_mdl_bin_edges_routes_through_njit_kernel(monkeypatch):
    """The public path must call `_best_mdl_split_kernel` (absent on pre-iter81 HEAD -> AttributeError there)."""
    calls = {"n": 0}
    orig = M._best_mdl_split_kernel

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(M, "_best_mdl_split_kernel", spy)
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.standard_normal(n).astype(np.float32)
    yclass = (x > 0).astype(np.int32)
    M._mdl_bin_edges(x, yclass, 2, max_bins=8)
    assert calls["n"] == 1
