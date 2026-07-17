"""Unit + biz_value tests for signed interaction-information (co-information) ranking & routing (backlog idea #8).

Covers:
  * MM correction of all three MI terms before the II difference (iron rule (a));
  * the signed sign contract -- synergy II>0, redundancy II<0, additive II~=0;
  * the permutation-null floor on positive II rejects chance-positive noise pairs (iron rule (b));
  * routing demotes the additive cross-mix speculative pair while keeping genuine synergy (the F2 mechanism);
  * SELF-GATING: floor==0 / disabled is a structural no-op (every pair kept).

biz_value: on a 3-pair pool {synergistic, additive cross-mix, redundant} the routing keeps ONLY the genuine
synergy and demotes the additive cross-mix surrogate the user's weak-F2 hit -- a strictly cleaner candidate set
than the ratio gate (which admits all three).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlframe.feature_selection.filters._interaction_information import (
    ROUTE_ADDITIVE,
    ROUTE_REDUNDANT,
    ROUTE_SYNERGY,
    _mm_bias,
    pair_interaction_information,
    pooled_pair_ii_null_floor,
    route_prospective_pairs,
)

NBINS = 10


def _discretize(x, nbins=NBINS):
    ranks = np.argsort(np.argsort(x))
    return (ranks * nbins // len(x)).astype(np.int64)


def _mi_plugin(xc, yc, kx, ky):
    n = len(xc)
    joint = np.zeros((kx, ky), dtype=np.float64)
    for i in range(n):
        joint[xc[i], yc[i]] += 1.0
    joint /= n
    px = joint.sum(1)
    py = joint.sum(0)
    mi = 0.0
    for i in range(kx):
        if px[i] <= 0:
            continue
        for j in range(ky):
            if joint[i, j] <= 0 or py[j] <= 0:
                continue
            mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))
    return mi


# ---------------------------------------------------------------------------
# MM correction
# ---------------------------------------------------------------------------
def test_mm_bias_formula_and_edges():
    # (Kx-1)(Ky-1)/2n
    assert _mm_bias(10, 5, 1000) == pytest.approx((9 * 4) / 2000.0)
    # degenerate cardinalities / n -> 0 (no negative bias)
    assert _mm_bias(1, 5, 1000) == 0.0
    assert _mm_bias(10, 1, 1000) == 0.0
    assert _mm_bias(10, 5, 0) == 0.0


def test_ii_mm_corrects_all_three_terms():
    """II_mm = (pair_mi - bias_joint) - (mi_a - bias_a) - (mi_b - bias_b). The joint term, with
    nbins_a*nbins_b bins, carries the largest correction; an un-corrected II is strictly larger."""
    mi_a, mi_b, pair_mi = 0.20, 0.18, 0.55
    n, ky = 2000, 10
    ii_raw = pair_interaction_information(mi_a, mi_b, pair_mi, NBINS, NBINS, ky, n, miller_madow=False)
    ii_mm = pair_interaction_information(mi_a, mi_b, pair_mi, NBINS, NBINS, ky, n, miller_madow=True)
    bias_a = _mm_bias(NBINS, ky, n)
    bias_b = _mm_bias(NBINS, ky, n)
    bias_joint = _mm_bias(NBINS * NBINS, ky, n)
    expected_mm = (pair_mi - bias_joint) - (mi_a - bias_a) - (mi_b - bias_b)
    assert ii_mm == pytest.approx(expected_mm)
    assert ii_raw == pytest.approx(pair_mi - mi_a - mi_b)
    # the joint bias dominates -> MM correction makes II SMALLER (debiases the joint-inflation synergy illusion)
    assert ii_mm < ii_raw


# ---------------------------------------------------------------------------
# Signed sign contract on real data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ii_sign_synergy_redundancy_additive(seed):
    n = 3000
    rng = np.random.default_rng(seed)

    # synergy: y = sign(a*b), a,b independent -> each ~0 marginal, joint high -> II >> 0
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y = (np.sign(a * b) + 1) / 2
    ac, bc, yc = _discretize(a), _discretize(b), _discretize(y)
    ky = int(yc.max()) + 1
    ii_syn = pair_interaction_information(
        _mi_plugin(ac, yc, NBINS, ky),
        _mi_plugin(bc, yc, NBINS, ky),
        _mi_plugin(ac * NBINS + bc, yc, NBINS * NBINS, ky),
        NBINS,
        NBINS,
        ky,
        n,
    )
    assert ii_syn > 0.1, f"synergy II should be strongly positive, got {ii_syn}"

    # redundancy: b ~= a, y ~ a -> a,b carry the SAME signal -> II < 0
    a = rng.standard_normal(n)
    b = a + 0.1 * rng.standard_normal(n)
    y = a + 0.3 * rng.standard_normal(n)
    ac, bc, yc = _discretize(a), _discretize(b), _discretize(y)
    ky = int(yc.max()) + 1
    ii_red = pair_interaction_information(
        _mi_plugin(ac, yc, NBINS, ky),
        _mi_plugin(bc, yc, NBINS, ky),
        _mi_plugin(ac * NBINS + bc, yc, NBINS * NBINS, ky),
        NBINS,
        NBINS,
        ky,
        n,
    )
    assert ii_red < -0.1, f"redundancy II should be strongly negative, got {ii_red}"

    # additive (noisy independent terms): y = a + c + 2*noise -> II ~= 0 (small, near the chance ceiling)
    a = rng.standard_normal(n)
    c = rng.standard_normal(n)
    y = a + c + 2.0 * rng.standard_normal(n)
    ac, cc, yc = _discretize(a), _discretize(c), _discretize(y)
    ky = int(yc.max()) + 1
    ii_add = pair_interaction_information(
        _mi_plugin(ac, yc, NBINS, ky),
        _mi_plugin(cc, yc, NBINS, ky),
        _mi_plugin(ac * NBINS + cc, yc, NBINS * NBINS, ky),
        NBINS,
        NBINS,
        ky,
        n,
    )
    # additive completion II is small and FAR below the synergy II.
    assert abs(ii_add) < 0.1, f"additive II should be ~0, got {ii_add}"
    assert ii_add < ii_syn


# ---------------------------------------------------------------------------
# Permutation-null floor
# ---------------------------------------------------------------------------
def test_null_floor_rejects_chance_positive_noise_ii():
    """On a pure-noise pool the actual max positive II sits AT/BELOW the q95 permutation-null floor."""
    n = 3000
    rng = np.random.default_rng(11)
    ncols = 12
    factors = np.zeros((n, ncols), dtype=np.int64)
    for k in range(ncols):
        factors[:, k] = _discretize(rng.standard_normal(n))
    yc = _discretize(rng.standard_normal(n))
    nbins = np.array([NBINS] * ncols, dtype=np.int64)
    freqs_y = np.bincount(yc).astype(np.float64) / n
    pairs = list(itertools.combinations(range(ncols), 2))
    pa = np.array([p[0] for p in pairs])
    pb = np.array([p[1] for p in pairs])
    floor = pooled_pair_ii_null_floor(
        factors_data=factors,
        nbins=nbins,
        pair_a=pa,
        pair_b=pb,
        marginal_mi_a=np.zeros(len(pairs)),
        marginal_mi_b=np.zeros(len(pairs)),
        classes_y=yc,
        freqs_y=freqs_y,
        n_permutations=50,
        quantile=0.95,
        random_seed=7,
    )
    ky = len(freqs_y)
    actual = max(
        pair_interaction_information(
            _mi_plugin(factors[:, i], yc, NBINS, ky),
            _mi_plugin(factors[:, j], yc, NBINS, ky),
            _mi_plugin(factors[:, i] * NBINS + factors[:, j], yc, NBINS * NBINS, ky),
            NBINS,
            NBINS,
            ky,
            n,
        )
        for i, j in pairs
    )
    assert floor > 0.0
    assert actual <= floor + 1e-6, f"noise max II {actual} should be <= null floor {floor}"


def test_null_floor_degenerate_returns_zero():
    n = 3000
    rng = np.random.default_rng(0)
    factors = np.zeros((n, 3), dtype=np.int64)
    for k in range(3):
        factors[:, k] = _discretize(rng.standard_normal(n))
    yc = _discretize(rng.standard_normal(n))
    nbins = np.array([NBINS, NBINS, NBINS])
    fy = np.bincount(yc).astype(float) / n
    # < 2 pairs -> 0; n_permutations 0 -> 0; single-class target -> 0
    assert pooled_pair_ii_null_floor(factors, nbins, np.array([0]), np.array([1]), np.zeros(1), np.zeros(1), yc, fy, n_permutations=10) == 0.0
    assert pooled_pair_ii_null_floor(factors, nbins, np.array([0, 1]), np.array([1, 2]), np.zeros(2), np.zeros(2), yc, fy, n_permutations=0) == 0.0
    assert (
        pooled_pair_ii_null_floor(factors, nbins, np.array([0, 1]), np.array([1, 2]), np.zeros(2), np.zeros(2), yc, np.array([1.0]), n_permutations=10) == 0.0
    )


# ---------------------------------------------------------------------------
# Routing -- the F2 cross-mix mechanism (biz_value)
# ---------------------------------------------------------------------------
def test_routing_demotes_additive_crossmix_keeps_synergy():
    """biz_value: pool = {synergy (a,b), additive cross-mix (a,c2), additive (c2,d2)}; the genuine signal is
    the synergy. Routing keeps ONLY the synergy and demotes the additive speculative cross-mix surrogate."""
    n = 3000
    rng = np.random.default_rng(5)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    ysyn = (np.sign(a * b) + 1) / 2  # genuine 2-way synergy is the target
    c2 = rng.standard_normal(n)
    d2 = rng.standard_normal(n)

    NCOL = 4
    fa = np.zeros((n, NCOL), dtype=np.int64)
    fa[:, 0] = _discretize(a)
    fa[:, 1] = _discretize(b)
    fa[:, 2] = _discretize(c2)
    fa[:, 3] = _discretize(d2)
    yc = _discretize(ysyn)
    ky = int(yc.max()) + 1
    nb = np.array([NBINS] * NCOL)
    fy = np.bincount(yc).astype(float) / n

    def jmi(i, j):
        return _mi_plugin(fa[:, i] * NBINS + fa[:, j], yc, NBINS * NBINS, ky)

    def mmi(i):
        return _mi_plugin(fa[:, i], yc, NBINS, ky)

    cached = {(i,): mmi(i) for i in range(NCOL)}
    pp = {
        ((0, 1), jmi(0, 1)): 1,  # synergy
        ((0, 2), jmi(0, 2)): 1,  # cross-mix additive (a + unrelated c2)
        ((2, 3), jmi(2, 3)): 1,  # additive (two unrelated noise cols)
    }
    pa = np.array([0, 0, 2])
    pb = np.array([1, 2, 3])
    floor = pooled_pair_ii_null_floor(
        factors_data=fa,
        nbins=nb,
        pair_a=pa,
        pair_b=pb,
        marginal_mi_a=np.zeros(3),
        marginal_mi_b=np.zeros(3),
        classes_y=yc,
        freqs_y=fy,
        n_permutations=50,
        quantile=0.95,
        random_seed=1,
    )
    kept, routes, _iis = route_prospective_pairs(
        pp,
        cached_MIs=cached,
        nbins=nb,
        nbins_y=ky,
        n=n,
        ii_floor=floor,
        synergy_added_idx={2, 3},  # c2, d2 are the speculative bootstrap operands
    )
    assert routes[(0, 1)] == ROUTE_SYNERGY
    assert routes[(0, 2)] == ROUTE_ADDITIVE
    assert routes[(2, 3)] == ROUTE_ADDITIVE
    kept_pairs = {k[0] for k in kept}
    assert (0, 1) in kept_pairs, "genuine synergy pair must be kept"
    assert (0, 2) not in kept_pairs, "additive cross-mix speculative pair must be demoted"
    assert (2, 3) not in kept_pairs, "additive speculative pair must be demoted"


def test_routing_keeps_redundant_and_selected_pairs():
    """Negative-II (redundant) pairs and selected-selected (non-speculative) additive pairs are KEPT
    (redundant tagged for cluster-aggregate; selected pairs preserve the create/keep/drop contract)."""
    n = 3000
    rng = np.random.default_rng(2)
    a = rng.standard_normal(n)
    b = a + 0.1 * rng.standard_normal(n)  # b ~= a
    y = a + 0.3 * rng.standard_normal(n)
    c2 = rng.standard_normal(n)

    fa = np.zeros((n, 3), dtype=np.int64)
    fa[:, 0] = _discretize(a)
    fa[:, 1] = _discretize(b)
    fa[:, 2] = _discretize(c2)
    yc = _discretize(y)
    ky = int(yc.max()) + 1
    nb = np.array([NBINS] * 3)
    fy = np.bincount(yc).astype(float) / n

    def jmi(i, j):
        return _mi_plugin(fa[:, i] * NBINS + fa[:, j], yc, NBINS * NBINS, ky)

    def mmi(i):
        return _mi_plugin(fa[:, i], yc, NBINS, ky)

    cached = {(i,): mmi(i) for i in range(3)}
    pp = {
        ((0, 1), jmi(0, 1)): 1,  # redundant (b ~= a)
        ((0, 2), jmi(0, 2)): 1,  # additive cross-mix but operand 2 NOT speculative here
    }
    pa = np.array([0, 0])
    pb = np.array([1, 2])
    floor = pooled_pair_ii_null_floor(
        factors_data=fa,
        nbins=nb,
        pair_a=pa,
        pair_b=pb,
        marginal_mi_a=np.zeros(2),
        marginal_mi_b=np.zeros(2),
        classes_y=yc,
        freqs_y=fy,
        n_permutations=40,
        quantile=0.95,
        random_seed=3,
    )
    kept, routes, _iis = route_prospective_pairs(
        pp,
        cached_MIs=cached,
        nbins=nb,
        nbins_y=ky,
        n=n,
        ii_floor=floor,
        synergy_added_idx=set(),  # nothing speculative -> nothing demoted
    )
    assert routes[(0, 1)] == ROUTE_REDUNDANT
    kept_pairs = {k[0] for k in kept}
    assert (0, 1) in kept_pairs and (0, 2) in kept_pairs, "no speculative operands -> all pairs kept"


def test_routing_no_op_when_floor_zero():
    """ii_floor <= 0 (degenerate / disabled null) keeps EVERY pair regardless of route -> byte-stable."""
    n = 2000
    rng = np.random.default_rng(0)
    fa = np.zeros((n, 3), dtype=np.int64)
    for k in range(3):
        fa[:, k] = _discretize(rng.standard_normal(n))
    yc = _discretize(rng.standard_normal(n))
    nb = np.array([NBINS] * 3)
    ky = int(yc.max()) + 1
    cached = {(i,): 0.01 for i in range(3)}
    pp = {((0, 1), 0.05): 1, ((0, 2), 0.04): 1}
    kept, _routes, _iis = route_prospective_pairs(
        pp,
        cached_MIs=cached,
        nbins=nb,
        nbins_y=ky,
        n=n,
        ii_floor=0.0,
        synergy_added_idx={1, 2},
    )
    assert len(kept) == len(pp), "floor==0 must keep every pair (no-op)"
