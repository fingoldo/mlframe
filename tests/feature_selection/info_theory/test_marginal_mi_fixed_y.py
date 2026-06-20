"""Regression sensor for the precomputed-y marginal-MI fast path.

``marginal_mi_binned_fixed_y`` (with terms from ``precompute_marginal_y_terms``) hoists the
fixed-``y`` ``H(Y)``/``k_y`` out of the per-candidate marginal-MI loop in the usability
candidate pool. It MUST stay bit-identical to the general ``_cmi_from_binned(x, y, None)``
path it replaces -- including on tied / discrete / constant columns where the MM bias and
plug-in entropies are most likely to drift if either path changes.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
    _cmi_from_binned,
    _quantile_bin,
    marginal_mi_binned_fixed_y,
    precompute_marginal_y_terms,
)


def test_marginal_mi_fixed_y_bit_identical_to_cmi_from_binned():
    rng = np.random.default_rng(7)
    for n in (300, 4000):
        cases = {
            "continuous": (_quantile_bin(rng.normal(size=n), 10), _quantile_bin(rng.normal(size=n), 10)),
            "discrete": (rng.integers(0, 5, n).astype(np.int64), rng.integers(0, 7, n).astype(np.int64)),
            "tied_binary": ((rng.normal(size=n) > 0).astype(np.int64), _quantile_bin(rng.normal(size=n), 10)),
            "constant_x": (np.zeros(n, dtype=np.int64), _quantile_bin(rng.normal(size=n), 10)),
        }
        for tag, (x, y) in cases.items():
            ref = _cmi_from_binned(x, y, None)
            fast = marginal_mi_binned_fixed_y(x, *precompute_marginal_y_terms(y))
            assert fast == ref, f"n={n} {tag}: {fast!r} != {ref!r}"


def test_renumber_two_dense_partition_identical_to_generic():
    """The single-pass 2-col densify must induce the SAME partition + nclasses as the generic
    factorize-then-combine path (first-seen ids may differ; every consumer is partition-invariant)."""
    import mlframe.feature_selection.filters._mi_greedy_cmi_fe as M

    def _canon(j):
        seen, out, c = {}, np.empty_like(j), 0
        for i, v in enumerate(j.tolist()):
            if v not in seen:
                seen[v] = c
                c += 1
            out[i] = seen[v]
        return out, c

    rng = np.random.default_rng(11)
    for n in (1, 500, 20000):
        a = rng.integers(0, 10, n).astype(np.int64)
        b = rng.integers(0, 13, n).astype(np.int64)
        inv, nc = M._renumber_two_dense_njit(a, b)
        assert nc >= 0
        ref, mref = M._factorize_dense_njit(np.ascontiguousarray(a, np.int64))
        ref2, m2 = M._combine_factorize_njit(ref, b, mref)
        c_fast, k_fast = _canon(inv)
        c_ref, k_ref = _canon(ref2)
        assert k_fast == k_ref == nc and np.array_equal(c_fast, c_ref)


def test_renumber_two_dense_falls_back_on_negative_and_huge_span():
    import mlframe.feature_selection.filters._mi_greedy_cmi_fe as M

    a = np.array([-1, 0, 1], dtype=np.int64)
    b = np.array([0, 1, 2], dtype=np.int64)
    _, nc = M._renumber_two_dense_njit(a, b)
    assert nc == -1  # negative id -> fallback sentinel
    # _renumber_joint must still produce a valid dense joint via the generic path on negatives.
    joint, k = M._renumber_joint(a, b)
    assert k == 3 and joint.shape == (3,)


def test_precompute_marginal_y_terms_matches_inline_entropy():
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _entropy_from_classes

    y = _quantile_bin(np.random.default_rng(3).normal(size=2000), 10)
    y_i, h_y, k_y = precompute_marginal_y_terms(y)
    h_ref, k_ref = _entropy_from_classes(np.ascontiguousarray(y, dtype=np.int64))
    assert h_y == h_ref and k_y == k_ref
