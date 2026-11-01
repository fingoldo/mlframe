"""The triple-MI kernel returns the triples in lexicographic order with the plug-in MI of each, whatever order its parallel loop visits the first indices in."""

import itertools

import numpy as np

from mlframe.feature_selection.filters._orthogonal_triplet_fe import _triple_mi_kernel


def test_triples_in_lexicographic_order_with_plug_in_mi():
    """Compare against a direct contingency-table computation."""
    rng = np.random.default_rng(1)
    p, n, base, n_y = 9, 400, 3, 2
    codes = rng.integers(0, base, (p, n)).astype(np.int64)
    y = rng.integers(0, n_y, n).astype(np.int64)
    oi, oj, ok, mi = _triple_mi_kernel(codes, y, base, n_y)
    want = list(itertools.combinations(range(p), 3))
    assert list(zip(oi, oj, ok)) == want
    for t, (i, j, k) in enumerate(want[:20]):
        cell = (codes[i] * base + codes[j]) * base + codes[k]
        joint = np.zeros((base**3, n_y))
        np.add.at(joint, (cell, y), 1)
        px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        nz = joint > 0
        ref = float(((joint / n)[nz] * np.log(joint[nz] * n / (px @ py)[nz])).sum())
        assert abs(mi[t] - ref) < 1e-12
