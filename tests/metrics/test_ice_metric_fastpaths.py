"""Regression sensors for iter68 fast-path optimizations in compute_probabilistic_multiclass_error.

Two changes, both bit-identical-by-construction:
  1. Binary 1-D y_score builds ``probs = [1 - y, y]`` directly instead of ``np.vstack([1-y, y]).T`` + re-slice.
  2. The non-0-indexed integer label auto-detect skips the O(n log n) ``np.unique`` when a cheap ``min==0 and max==K-1``
     pre-gate proves the remap branch cannot fire.

These tests pin the OUTPUT identity across the relevant input regimes so a future refactor that breaks either
fast path (e.g. mis-gating the unique skip on shifted labels, or dropping a probs column) is caught.
"""

import numpy as np
import pytest

from mlframe.metrics._ice_metric import compute_probabilistic_multiclass_error as ice


def _mk_multiclass(n, k, seed=0):
    rng = np.random.default_rng(seed)
    yt = rng.integers(0, k, n).astype(np.int64)
    yt[:k] = np.arange(k)  # guarantee all classes present
    sc = rng.random((n, k))
    sc /= sc.sum(1, keepdims=True)
    return yt, sc


def test_binary_1d_probs_equivalent_to_full_2col():
    """A 1-D y_score must yield the same ICE as the explicit 2-column proba matrix [1-y, y]."""
    rng = np.random.default_rng(1)
    yt = (rng.random(5000) < 0.3).astype(np.int64)
    p = np.clip(rng.random(5000) * 0.6 + yt * 0.3, 0, 1)
    v_1d = ice(yt, p)
    v_2col = ice(yt, np.column_stack([1 - p, p]))
    assert v_1d == v_2col


def test_label_remap_still_fires_on_shifted_labels():
    """Shifted integer labels (10,20,30) must still trigger the remap -> identical ICE to the 0..K-1 version.

    The min/max pre-gate must NOT skip the unique scan here (min=10 != 0), or the metric would mis-route and
    collapse to a no-skill value.
    """
    yt, sc = _mk_multiclass(4000, 3, seed=2)
    labelvals = np.array([10, 20, 30])
    yt_shift = labelvals[yt]
    v_base = ice(yt, sc)
    v_shift = ice(yt_shift, sc)
    assert v_shift == v_base


def test_gate_skip_is_bit_identical_to_unique_path():
    """On already-0..K-1 multiclass (gate skips unique) the result equals an explicit labels= call that forces the unique path."""
    for k in (3, 5):
        yt, sc = _mk_multiclass(4000, k, seed=3 + k)
        v_gate = ice(yt, sc)  # gate skips np.unique
        v_explicit = ice(yt, sc, labels=np.arange(k))  # labels given -> different branch, same numerics
        assert v_gate == v_explicit, f"k={k}"


def test_missing_class_with_full_range_no_remap():
    """When a middle class is absent but 0 and K-1 are present (min=0,max=K-1, size<K), no remap fires -- gate skip is valid."""
    rng = np.random.default_rng(7)
    k = 5
    yt = rng.integers(0, k, 4000).astype(np.int64)
    yt[yt == 2] = 3  # drop class 2
    yt[0], yt[1] = 0, k - 1  # ensure full range
    sc = rng.random((4000, k))
    sc /= sc.sum(1, keepdims=True)
    # Must not raise and must equal the labels=arange forced-unique path.
    assert ice(yt, sc) == ice(yt, sc, labels=np.arange(k))
