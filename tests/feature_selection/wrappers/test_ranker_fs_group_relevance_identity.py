"""Regression: ``group_aware_relevance`` was restructured to sort rows by group once
(contiguous per-query blocks) instead of per-(feature, group) boolean-mask indexing.
The result must be BIT-IDENTICAL to the old mask-based reference (``_binned_mi`` is
order-invariant and stable-sort keeps within-group order), and a feature that is
constant within every query must still score ~0 (the whole point of query-aware
relevance).
"""
from __future__ import annotations

import numpy as np

from mlframe.training.ranking._ranker_fs import (
    _binned_mi,
    _mi_from_edges,
    group_aware_relevance,
)


def _mi_from_edges_numpy(x, y, xe, ye):
    """Numpy reference for the njit ``_mi_from_edges`` (the pre-njit body)."""
    xb = np.clip(np.searchsorted(xe[1:-1], x, side="right"), 0, xe.size - 2)
    yb = np.clip(np.searchsorted(ye[1:-1], y, side="right"), 0, ye.size - 2)
    joint = np.zeros((xe.size - 1, ye.size - 1), dtype=np.float64)
    np.add.at(joint, (xb, yb), 1.0)
    joint /= x.shape[0]
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))


def test_mi_from_edges_njit_matches_numpy_reference():
    # The njit kernel must reproduce the numpy searchsorted+add.at+entropy formula to entropy-sum
    # FP round-off (~1e-15); the binning (count of inner edges <= v) is exact.
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(3000):
        m = int(rng.integers(20, 40))
        x = rng.standard_normal(m)
        y = rng.standard_normal(m)
        xe = np.unique(np.quantile(x, np.linspace(0, 1, 9)))
        ye = np.unique(np.quantile(y, np.linspace(0, 1, 9)))
        if xe.size < 2 or ye.size < 2:
            continue
        worst = max(worst, abs(_mi_from_edges(x, y, xe, ye) - _mi_from_edges_numpy(x, y, xe, ye)))
    assert worst < 1e-12, f"njit _mi_from_edges diverges {worst:.2e} from numpy reference"


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


def test_group_aware_relevance_bit_identical_with_non_finite_fallback():
    # Non-finite values in some (feature, group) cells force the exact per-pair fallback
    # (the all-finite batched-quantile fast path cannot handle the JOINT per-(x, y) finite
    # mask). The whole-function result must still be bit-identical to the mask reference.
    rng = np.random.default_rng(11)
    n, nf, avg_g = 6000, 10, 25
    ngroups = n // avg_g
    groups = np.repeat(np.arange(ngroups), avg_g)[:n]
    rng.shuffle(groups)
    arr = rng.standard_normal((n, nf))
    arr[7, 2] = np.nan
    arr[avg_g + 3, 5] = np.inf
    arr[3 * avg_g : 3 * avg_g + 2, 0] = np.nan
    y = rng.standard_normal(n)
    y[50] = np.nan
    cols = [f"f{j}" for j in range(nf)]
    got = group_aware_relevance(cols, arr, y, groups)
    ref = _old_reference(cols, arr, y, groups)
    for c in cols:
        assert got[c] == ref[c], f"{c}: {got[c]!r} != {ref[c]!r} (fallback not bit-identical)"


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


def test_group_features_mi_njit_matches_per_column():
    # The fused per-group kernel must reproduce the per-column _mi_from_edges path bit-for-bit
    # (same batched-quantile edges deduped inline == np.unique, same joint-hist + entropy order).
    import numpy as np
    from mlframe.training.ranking._ranker_fs import _group_features_mi_njit, _mi_from_edges

    rng = np.random.default_rng(9)
    for _ in range(200):
        sz = int(rng.integers(20, 40))
        ncols = int(rng.integers(1, 8))
        block = np.ascontiguousarray(rng.standard_normal((sz, ncols)))
        y_g = rng.standard_normal(sz)
        probs = np.linspace(0, 1, 9)
        ye = np.unique(np.quantile(y_g, probs))
        if ye.size < 2:
            continue
        col_fin = np.isfinite(block).all(axis=0)
        qa = np.quantile(block, probs, axis=0)
        out = np.zeros(ncols)
        _group_features_mi_njit(block, y_g, qa, ye, col_fin, out)
        for j in range(ncols):
            xe = np.unique(qa[:, j])
            expected = 0.0 if (np.ptp(block[:, j]) == 0.0 or xe.size < 2) else _mi_from_edges(block[:, j], y_g, xe, ye)
            assert out[j] == expected, f"col {j}: {out[j]!r} != {expected!r}"


def test_group_aware_mrmr_incremental_redundancy_matches_mean_reference():
    # The greedy loop maintains red_sum incrementally instead of np.mean([red[i,s] for s in selected]) per
    # candidate. Assert the selected feature list is identical to a mean-based reference greedy on the same
    # relevance/redundancy (same summation order -> bit-identical redundancy -> identical argmax selection).
    import numpy as np

    def _ref_greedy(rel, red, eff_floor, cap, w=1.0):
        eligible = np.where(rel > eff_floor)[0]
        if eligible.size == 0:
            return []
        selected = [int(eligible[np.argmax(rel[eligible])])]
        remaining = [i for i in range(len(rel)) if i != selected[0]]
        while remaining and len(selected) < cap:
            best_i, best_score = None, -np.inf
            for i in remaining:
                if rel[i] <= eff_floor:
                    continue
                redundancy = float(np.mean([red[i, s] for s in selected]))
                score = rel[i] - w * redundancy
                if score > best_score:
                    best_score, best_i = score, i
            if best_i is None or best_score <= 0.0:
                break
            selected.append(best_i)
            remaining.remove(best_i)
        return selected

    def _inc_greedy(rel, red, eff_floor, cap, w=1.0):
        eligible = np.where(rel > eff_floor)[0]
        if eligible.size == 0:
            return []
        selected = [int(eligible[np.argmax(rel[eligible])])]
        remaining = [i for i in range(len(rel)) if i != selected[0]]
        red_sum = red[:, selected[0]].astype(np.float64, copy=True)
        while remaining and len(selected) < cap:
            ns = len(selected)
            best_i, best_score = None, -np.inf
            for i in remaining:
                if rel[i] <= eff_floor:
                    continue
                score = rel[i] - w * (red_sum[i] / ns)
                if score > best_score:
                    best_score, best_i = score, i
            if best_i is None or best_score <= 0.0:
                break
            selected.append(best_i)
            remaining.remove(best_i)
            red_sum += red[:, best_i]
        return selected

    rng = np.random.default_rng(3)
    for _ in range(200):
        nf = int(rng.integers(4, 40))
        rel = np.abs(rng.standard_normal(nf))
        red = np.abs(rng.standard_normal((nf, nf)))
        red = (red + red.T) / 2.0
        np.fill_diagonal(red, 0.0)
        eff_floor = 0.2 * float(rel.max())
        cap = int(rng.integers(2, nf))
        assert _ref_greedy(rel, red, eff_floor, cap) == _inc_greedy(rel, red, eff_floor, cap)
