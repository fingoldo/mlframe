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


def test_precompute_marginal_y_terms_matches_inline_entropy():
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _entropy_from_classes

    y = _quantile_bin(np.random.default_rng(3).normal(size=2000), 10)
    y_i, h_y, k_y = precompute_marginal_y_terms(y)
    h_ref, k_ref = _entropy_from_classes(np.ascontiguousarray(y, dtype=np.int64))
    assert h_y == h_ref and k_y == k_ref
