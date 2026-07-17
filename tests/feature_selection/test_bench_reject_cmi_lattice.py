"""bench-reject evidence: backlog #10 "Conditional-MI complementarity growth (Apriori lattice)".

#10 proposed growing triples from the order-2 SURVIVOR frontier by testing only third
columns ``c`` maximising the conditional-MI uplift ``I((a,b,c);y) - I((a,b);y)`` (reusing
the shipped #7 order-3 maxT floor as the rail), to catch a 3rd operand that matters ONLY
given ``(a,b)`` -- pure higher-order synergy the univariate triplet seeder misses.

It was prototyped + benchmarked (D:/Temp/cmi_lattice_proto{,2,3,4,5}.py, cmi_lattice_cost.py)
and REJECTED. This test LOCKS IN the three measured findings so the mechanism is not
re-attempted as the backlog specifies (see the bench-rejected note in ``_mrmr_fe_step.py``):

  1. ANTI-MONOTONE WALL (the binding reason). The backlog fixture ``y = sign(x1*x2)*x3 > 0``
     is mathematically a PURE 3-way sign XOR (verified 100% agreement with the 3-sign XOR),
     NOT a "detectable 2-way (x1,x2) + conditional x3". For a pure k-way interaction EVERY
     (k-1)-way sub-tuple has joint MI buried in the noise cloud (no distinguished signal), so
     the needle PAIR never enters the top-N order-2 frontier -> the needle TRIPLE is never
     grown. Apriori "grow from surviving (k-1)-tuples" structurally cannot reach pure
     higher-order synergy.

  2. The GROWTH KERNEL works but is moot: GIVEN a base pair holding 2 of the 3 needle legs,
     the CMI uplift ranks the true 3rd operand RANK-0 -- but that base pair is exactly what
     the frontier never contains (finding 1).

  3. REDUNDANT with the shipped #6 GBM seeder: on the SAME pure-3-way-XOR fixture the GBM
     split-co-occurrence proposer recovers the needle as its top triple (it conditions a
     zero-marginal operand on its co-splitter WITHOUT a detectable (k-1)-tuple, so it is
     immune to the anti-monotone wall). #10's purpose is already delivered by #6.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from mlframe.feature_selection.filters.info_theory import (
    batch_pair_mi_prange,
    batch_triple_mi_prange,
)


def _discretize(X, nb=5):
    """Helper that discretize."""
    n, p = X.shape
    D = np.zeros((n, p), dtype=np.int32)
    nbins = np.zeros(p, dtype=np.int64)
    for j in range(p):
        col = X[:, j]
        qs = np.quantile(col, np.linspace(0, 1, nb + 1)[1:-1])
        codes = np.searchsorted(qs, col, side="right")
        uniq = np.unique(codes)
        remap = {u: i for i, u in enumerate(uniq)}
        D[:, j] = np.array([remap[c] for c in codes], dtype=np.int32)
        nbins[j] = len(uniq)
    return D, nbins


def _ctx(D, y, nbins):
    """Helper that ctx."""
    n = D.shape[0]
    yc = np.ascontiguousarray(y.astype(np.int64))
    ky = int(np.unique(yc).size)
    fy = np.bincount(yc, minlength=ky).astype(np.float64) / n
    return yc, np.ascontiguousarray(fy)


def _pair_mi(D, nbins, a, b, yc, fy):
    """Pair mi."""
    return float(batch_pair_mi_prange(D, np.array([a]), np.array([b]), nbins, yc, fy)[0])


def _all_pair_mis(D, nbins, p, yc, fy):
    """All pair mis."""
    pa, pb = zip(*combinations(range(p), 2))
    pa = np.asarray(pa, np.int64)
    pb = np.asarray(pb, np.int64)
    return pa, pb, batch_pair_mi_prange(D, pa, pb, nbins, yc, fy)


def _grow_from_pair(D, nbins, a, b, p, yc, fy):
    """Return (true-3rd-operand CMI-uplift rank, best-c). Reproduces #10's growth step."""
    pm = _pair_mi(D, nbins, a, b, yc, fy)
    tc = np.array([c for c in range(p) if c != a and c != b], np.int64)
    ta = np.full(len(tc), a, np.int64)
    tb = np.full(len(tc), b, np.int64)
    uplift = batch_triple_mi_prange(D, ta, tb, tc, nbins, yc, fy) - pm
    rank = np.argsort(uplift)[::-1]
    return tc, uplift, rank


# =============================================================================
# FINDING 1 (binding wall): the pure-3-way-XOR needle's sub-pairs are at the
# noise floor, so the needle pair never enters a top-N order-2 frontier.
# =============================================================================
def test_pure_3way_xor_subpairs_below_frontier():
    """Pure 3way xor subpairs below frontier."""
    rng = np.random.default_rng(0)
    n, p = 3000, 40
    X = rng.standard_normal((n, p))
    y = (np.sign(X[:, 0] * X[:, 1]) * X[:, 2] > 0).astype(np.int64)
    # it really is a pure 3-way sign XOR
    sign_xor = ((np.sign(X[:, 0]) * np.sign(X[:, 1]) * np.sign(X[:, 2])) > 0).astype(np.int64)
    assert np.mean(y == sign_xor) == 1.0

    D, nbins = _discretize(X, nb=5)
    yc, fy = _ctx(D, y, nbins)
    pa, pb, pmi = _all_pair_mis(D, nbins, p, yc, fy)
    order = np.argsort(pmi)[::-1]

    # ranks of the three needle sub-pairs among all C(40,2) pairs
    def _rank_of(s):
        """Rank of."""
        for r, i in enumerate(order):
            if {int(pa[i]), int(pb[i])} == s:
                return r
        raise AssertionError("pair not found")

    r01 = _rank_of({0, 1})
    r02 = _rank_of({0, 2})
    r12 = _rank_of({1, 2})
    # The load-bearing claim (binning-robust): NONE of the three needle sub-pairs is
    # in the realistic top-N order-2 frontier the lattice would grow from -- their
    # joint MI sits in the noise cloud (finite-sample binning gives them a small,
    # non-distinguished jMI, ranking them deep, NOT among the top handful). So
    # Apriori growth never seeds a base pair carrying 2 needle legs.
    FRONTIER_N = 12
    assert min(r01, r02, r12) >= FRONTIER_N, (r01, r02, r12)
    # equivalently, the top frontier pairs hold NO 2-leg needle pair
    top_frontier = [{int(pa[i]), int(pb[i])} for i in order[:FRONTIER_N]]
    assert not any(len(s & {0, 1, 2}) == 2 for s in top_frontier)


# =============================================================================
# FINDING 2: the growth kernel itself is sound -- GIVEN a base pair with 2 of
# the 3 needle legs, CMI uplift ranks the true 3rd operand RANK-0. (Moot, since
# finding 1 shows that base pair never reaches the frontier.)
# =============================================================================
def test_cmi_uplift_ranks_true_third_operand_first_given_base():
    """Cmi uplift ranks true third operand first given base."""
    rng = np.random.default_rng(0)
    n, p = 3000, 40
    X = rng.standard_normal((n, p))
    y = (np.sign(X[:, 0] * X[:, 1]) * X[:, 2] > 0).astype(np.int64)
    D, nbins = _discretize(X, nb=4)
    yc, fy = _ctx(D, y, nbins)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        third = ({0, 1, 2} - {a, b}).pop()
        tc, uplift, rank = _grow_from_pair(D, nbins, a, b, p, yc, fy)
        assert int(tc[rank[0]]) == third, (a, b, third, int(tc[rank[0]]))
        # the true third's uplift is far above the next-best (a clear margin)
        sorted_up = np.sort(uplift)[::-1]
        assert sorted_up[0] > 5 * max(sorted_up[1], 1e-9)


# =============================================================================
# FINDING 1 end-to-end: running the full lattice (frontier -> grow -> order-3
# floor) does NOT recover the pure-3-way-XOR needle (it is never grown).
# =============================================================================
def test_lattice_misses_pure_3way_needle_end_to_end():
    """Lattice misses pure 3way needle end to end."""
    rng = np.random.default_rng(0)
    n, p = 3000, 40
    X = rng.standard_normal((n, p))
    y = (np.sign(X[:, 0] * X[:, 1]) * X[:, 2] > 0).astype(np.int64)
    D, nbins = _discretize(X, nb=4)
    yc, fy = _ctx(D, y, nbins)
    pa, pb, pmi = _all_pair_mis(D, nbins, p, yc, fy)
    order = np.argsort(pmi)[::-1][:8]

    grown = {}
    for i in order:
        a, b, pm = int(pa[i]), int(pb[i]), float(pmi[i])
        tc = np.array([c for c in range(p) if c != a and c != b], np.int64)
        ta = np.full(len(tc), a, np.int64)
        tb = np.full(len(tc), b, np.int64)
        up = batch_triple_mi_prange(D, ta, tb, tc, nbins, yc, fy) - pm
        for ci in np.argsort(up)[::-1][:3]:
            grown[tuple(sorted((a, b, int(tc[ci]))))] = 1
    # the needle triple is NEVER among the grown candidates (anti-monotone wall)
    assert (0, 1, 2) not in grown


# =============================================================================
# FINDING 3 (redundancy): the shipped #6 GBM seeder recovers the SAME needle
# on the SAME pure-3-way-XOR fixture, so #10 closes no gap.
# =============================================================================
def test_gbm_seeder_already_recovers_what_lattice_targets():
    """Gbm seeder already recovers what lattice targets."""
    import pytest

    pytest.importorskip("lightgbm")
    from mlframe.feature_selection.filters._surrogate_interaction_seeder import (
        surrogate_gbm_interaction_seeds,
    )

    rng = np.random.default_rng(0)
    n, p = 3000, 40
    X = rng.standard_normal((n, p))
    y = (np.sign(X[:, 0] * X[:, 1]) * X[:, 2] > 0).astype(np.int64)
    D, nbins = _discretize(X, nb=5)
    yc, _ = _ctx(D, y, nbins)
    _, triples, _ = surrogate_gbm_interaction_seeds(
        D,
        yc,
        list(range(p)),
        is_classification=True,
        top_k_pairs=12,
        top_k_triples=8,
        n_estimators=300,
        max_depth=4,
        self_gate_reps=3,
        self_gate_min_z=2.0,
        random_seed=0,
    )
    # #6 seeds the pure-3-way-XOR needle (immune to the anti-monotone wall that blocks #10)
    assert any(set(t) == {0, 1, 2} for t in triples), triples
