"""Regression: ``group_aware_relevance`` was restructured to sort rows by group once
(contiguous per-query blocks) instead of per-(feature, group) boolean-mask indexing.
The result must be BIT-IDENTICAL to the old mask-based reference (``_binned_mi`` is
order-invariant and stable-sort keeps within-group order), and a feature that is
constant within every query must still score ~0 (the whole point of query-aware
relevance).
"""
from __future__ import annotations

import numpy as np

from mlframe.training.ranking._ranker_fs import _binned_mi, group_aware_relevance


def _old_reference(cols, arr, y, groups, bins=8):
    out = {}
    uniq = np.unique(groups)
    sizes = np.array([int(np.sum(groups == g)) for g in uniq], dtype=np.float64)
    masks = [groups == g for g in uniq]
    contributing_total = float(sizes[sizes >= 4].sum()) or 1.0
    for j, name in enumerate(cols):
        xj = arr[:, j]
        acc = 0.0
        for gi, m in enumerate(masks):
            if sizes[gi] >= 4:
                acc += sizes[gi] * _binned_mi(xj[m], y[m], bins=bins)
        out[name] = acc / contributing_total
    return out


def test_group_aware_relevance_bit_identical_to_mask_reference():
    rng = np.random.default_rng(7)
    n, nf, avg_g = 6000, 12, 25
    ngroups = n // avg_g
    groups = np.repeat(np.arange(ngroups), avg_g)[:n]
    rng.shuffle(groups)  # unsorted groups -> exercises the argsort block path
    arr = rng.standard_normal((n, nf))
    y = (rng.standard_normal(n) > 0).astype(np.float64) + rng.standard_normal(n) * 0.1
    cols = [f"f{j}" for j in range(nf)]
    got = group_aware_relevance(cols, arr, y, groups)
    ref = _old_reference(cols, arr, y, groups)
    assert set(got) == set(ref)
    for c in cols:
        assert got[c] == ref[c], f"{c}: {got[c]!r} != {ref[c]!r} (not bit-identical)"


def test_constant_within_group_feature_scores_low():
    rng = np.random.default_rng(3)
    n, avg_g = 4000, 20
    ngroups = n // avg_g
    groups = np.repeat(np.arange(ngroups), avg_g)[:n]
    y = rng.standard_normal(n)
    # f_query is constant within each query (encodes the query id) -> zero within-query
    # ranking power; f_signal correlates with y within groups.
    f_query = groups.astype(np.float64)
    f_signal = y + rng.standard_normal(n) * 0.1
    arr = np.column_stack([f_query, f_signal])
    out = group_aware_relevance(["f_query", "f_signal"], arr, y, groups)
    assert out["f_query"] < out["f_signal"]
    assert out["f_query"] < 1e-6
