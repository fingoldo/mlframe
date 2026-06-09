"""unit + biz_value: surrogate-GBM split-co-occurrence interaction seeder (backlog #6)
and its mandatory order-3 Westfall-Young maxT permutation-null floor rail (#7).

The univariate-MI ``seed_count`` that the prospective-pair sweep / triplet FE pick source
columns by is BLIND to pure synergy: a zero-marginal interaction operand
(``y = sign(x_a*x_b*x_c) + noise`` -- every marginal MI ~= 0) is never ranked top-N, so the
pair is never enumerated and the triple never seeded -> the needle is MISSED. The surrogate
seeder fits one shallow LightGBM, walks root-to-leaf paths, and tallies depth-discounted
split-gain co-occurrence to reach those zero-marginal operands; the order-3 maxT floor is the
load-bearing safety rail that keeps best-of-pool chance-max noise triples out.

Gates pinned here:
  #6 UNIT       -- ``batch_triple_mi_prange`` is bit-consistent with the merge_vars MI path.
  #6 BIZ_VALUE  -- needle RECALL: the 3-way + 2-way zero-marginal needles the univariate
                   seed_count completely misses are recovered as the top co-occurrence
                   seeds; the GBM self-gate emits NOTHING on pure noise.
  #7 BIZ_VALUE  -- order-3 floor: a pure-noise triple pool admits 0 triples; a genuine
                   3-way needle embedded in noise clears the floor alone.
  + self-gating / disabled-knob no-ops.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    batch_triple_mi_prange, merge_vars, compute_mi_from_classes,
)
from mlframe.feature_selection.filters._permutation_null import (
    pooled_triple_permutation_null_joint_mi_floor,
)

lgb = pytest.importorskip("lightgbm")
from mlframe.feature_selection.filters._surrogate_interaction_seeder import (  # noqa: E402
    surrogate_gbm_interaction_seeds,
)


def _discretize(X, nb=10):
    """Quantile-bin each column to <=nb dense ordinal bins -> (int matrix, per-col nbins)."""
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


def _univariate_topN(D, y, nbins, N):
    """What seed_count does: rank cols by univariate MI(x_j; y), keep top-N indices."""
    n, p = D.shape
    yc = y.astype(np.int64)
    ky = int(np.unique(yc).size)
    fy = np.bincount(yc, minlength=ky).astype(np.float64) / n
    mis = np.zeros(p)
    for j in range(p):
        cl, fx, _ = merge_vars(D, np.array([j]), None, nbins, dtype=np.int64)
        mis[j] = compute_mi_from_classes(cl.astype(np.int64), fx, yc, fy)
    order = np.argsort(mis)[::-1]
    return set(order[:N].tolist()), mis


# =============================================================================
# #6 UNIT: batch_triple_mi_prange bit-consistency with the merge_vars MI path.
# =============================================================================


class TestBatchTripleMIKernel:
    def test_matches_merge_vars_reference(self):
        rng = np.random.default_rng(0)
        n, p = 1200, 6
        X = rng.standard_normal((n, p))
        y = (np.sign(X[:, 0] * X[:, 1] * X[:, 2]) > 0).astype(np.int32)
        D, nbins = _discretize(X, nb=8)
        yc = y.astype(np.int32)
        fy = np.bincount(yc, minlength=2).astype(np.float64) / n

        triples = [(0, 1, 2), (0, 3, 4), (1, 2, 5), (3, 4, 5)]
        ta = np.array([t[0] for t in triples], dtype=np.int64)
        tb = np.array([t[1] for t in triples], dtype=np.int64)
        tc = np.array([t[2] for t in triples], dtype=np.int64)
        got = batch_triple_mi_prange(D, ta, tb, tc, nbins, yc, fy)

        ref = []
        for t in triples:
            cl, fx, _ = merge_vars(D, np.array(t), None, nbins, dtype=np.int64)
            ref.append(compute_mi_from_classes(cl.astype(np.int64), fx, yc.astype(np.int64), fy))
        ref = np.array(ref)
        assert np.max(np.abs(got - ref)) < 1e-9, f"kernel mismatch: {got} vs {ref}"
        # the genuine 3-way product joint MI dominates the noise triples.
        assert got[0] > 1.5 * max(got[1:]), f"needle joint MI should dominate: {got}"

    def test_empty_matrix_returns_zeros(self):
        D = np.zeros((0, 4), dtype=np.int32)
        nbins = np.array([3, 3, 3, 3], dtype=np.int64)
        out = batch_triple_mi_prange(
            D, np.array([0]), np.array([1]), np.array([2]), nbins,
            np.zeros(0, dtype=np.int32), np.array([0.5, 0.5]),
        )
        assert out.shape == (1,) and out[0] == 0.0


# =============================================================================
# #7 BIZ_VALUE: order-3 maxT floor -- noise FP control + genuine-needle clearance.
# =============================================================================


class TestOrder3MaxTFloor:
    def test_pure_noise_pool_admits_nothing(self):
        from itertools import combinations
        rng = np.random.default_rng(1)
        n, p = 2000, 12
        X = rng.standard_normal((n, p))
        y = rng.integers(0, 2, n).astype(np.int32)
        D, nbins = _discretize(X, nb=8)
        fy = np.bincount(y, minlength=2).astype(np.float64) / n
        allt = list(combinations(range(p), 3))
        ta = np.array([t[0] for t in allt], dtype=np.int64)
        tb = np.array([t[1] for t in allt], dtype=np.int64)
        tc = np.array([t[2] for t in allt], dtype=np.int64)
        obs = batch_triple_mi_prange(D, ta, tb, tc, nbins, y, fy)
        floor = pooled_triple_permutation_null_joint_mi_floor(
            D, nbins, ta, tb, tc, y, fy, n_permutations=25, quantile=0.95, random_seed=1)
        n_survive = int(np.sum(obs >= floor))
        # HARD gate: on pure noise, NO triple clears the best-of-pool chance ceiling.
        assert n_survive == 0, f"order-3 floor admitted {n_survive} pure-noise triples (floor={floor})"

    def test_genuine_needle_clears_floor_alone(self):
        from itertools import combinations
        rng = np.random.default_rng(2)
        n, p = 2000, 12
        X = rng.standard_normal((n, p))
        y = (np.sign(X[:, 0] * X[:, 1] * X[:, 2]) > 0).astype(np.int32)
        D, nbins = _discretize(X, nb=8)
        fy = np.bincount(y, minlength=2).astype(np.float64) / n
        allt = list(combinations(range(p), 3))
        ta = np.array([t[0] for t in allt], dtype=np.int64)
        tb = np.array([t[1] for t in allt], dtype=np.int64)
        tc = np.array([t[2] for t in allt], dtype=np.int64)
        obs = batch_triple_mi_prange(D, ta, tb, tc, nbins, y, fy)
        floor = pooled_triple_permutation_null_joint_mi_floor(
            D, nbins, ta, tb, tc, y, fy, n_permutations=25, quantile=0.95, random_seed=2)
        survivors = [allt[i] for i in range(len(allt)) if obs[i] >= floor]
        assert (0, 1, 2) in survivors, f"genuine 3-way needle did not clear the floor (={floor})"
        # The needle should be (nearly) the ONLY survivor -- noise stays out.
        assert len(survivors) <= 2, f"too many triples cleared the floor: {survivors}"

    def test_disabled_floor_is_zero_noop(self):
        rng = np.random.default_rng(3)
        n, p = 500, 6
        X = rng.standard_normal((n, p))
        y = rng.integers(0, 2, n).astype(np.int32)
        D, nbins = _discretize(X, nb=6)
        fy = np.bincount(y, minlength=2).astype(np.float64) / n
        from itertools import combinations
        allt = list(combinations(range(p), 3))
        ta = np.array([t[0] for t in allt], dtype=np.int64)
        tb = np.array([t[1] for t in allt], dtype=np.int64)
        tc = np.array([t[2] for t in allt], dtype=np.int64)
        floor = pooled_triple_permutation_null_joint_mi_floor(
            D, nbins, ta, tb, tc, y, fy, n_permutations=0)
        assert floor == 0.0


# =============================================================================
# #6 BIZ_VALUE: needle recall -- GBM seeder recovers zero-marginal needles the
# univariate seed_count misses, with NO noise admission (self-gate).
# =============================================================================


class TestGBMSeederNeedleRecall:
    def _frame(self, kind, seed=42, n=4000, p=200):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        if kind == "3way":
            y = (np.sign(X[:, 7] * X[:, 42] * X[:, 113]) > 0).astype(np.int64)
            needle = (7, 42, 113)
        elif kind == "2way":
            y = (np.sign(X[:, 5] * X[:, 30]) > 0).astype(np.int64)
            needle = (5, 30)
        else:  # noise
            y = rng.integers(0, 2, n).astype(np.int64)
            needle = None
        return X, y, needle

    def test_3way_needle_recovered_univariate_misses(self):
        X, y, needle = self._frame("3way")
        D, nbins = _discretize(X, nb=10)
        # CURRENT: univariate seed_count top-8 misses every needle operand.
        uni_top, mis = _univariate_topN(D, y, nbins, N=8)
        assert len(set(needle) & uni_top) == 0, (
            f"fixture invalid: univariate seed already sees needle {set(needle) & uni_top}"
        )
        # NEW: GBM seeder.
        pairs, triples, info = surrogate_gbm_interaction_seeds(
            D, y, list(range(D.shape[1])), is_classification=True,
            top_k_pairs=15, top_k_triples=10, random_seed=0)
        assert info["gated"], f"self-gate should pass on a genuine 3-way signal: {info}"
        ops = set()
        for pr in pairs: ops.update(pr)
        for tr in triples: ops.update(tr)
        # all three zero-marginal needle operands recovered.
        assert set(needle) <= ops, f"needle operands not all recovered: {set(needle) - ops}"
        # exact 3-way tuple proposed.
        assert tuple(sorted(needle)) in [tuple(sorted(t)) for t in triples], (
            f"exact 3-way needle not in seeded triples: {triples}"
        )

    def test_2way_needle_recovered_univariate_misses(self):
        X, y, needle = self._frame("2way")
        D, nbins = _discretize(X, nb=10)
        uni_top, mis = _univariate_topN(D, y, nbins, N=8)
        assert len(set(needle) & uni_top) == 0, "fixture invalid: univariate already sees needle"
        pairs, triples, info = surrogate_gbm_interaction_seeds(
            D, y, list(range(D.shape[1])), is_classification=True,
            top_k_pairs=15, top_k_triples=10, random_seed=0)
        assert info["gated"], f"self-gate should pass on a genuine 2-way signal: {info}"
        assert tuple(sorted(needle)) in [tuple(sorted(p_)) for p_ in pairs], (
            f"exact 2-way needle not in seeded pairs: {pairs}"
        )

    def test_pure_noise_self_gate_emits_nothing(self):
        X, y, _ = self._frame("noise")
        D, nbins = _discretize(X, nb=10)
        pairs, triples, info = surrogate_gbm_interaction_seeds(
            D, y, list(range(D.shape[1])), is_classification=True,
            top_k_pairs=15, top_k_triples=10, random_seed=0)
        # HARD gate: the permuted-y self-gate must reject pure noise -> no seeds.
        assert not info["gated"], f"self-gate must fail on pure noise: {info}"
        assert pairs == [] and triples == [], (
            f"seeder emitted {len(pairs)} pairs / {len(triples)} triples on pure noise"
        )

    def test_degenerate_pool_no_seeds(self):
        # < 2 candidates -> empty, no LightGBM fit.
        D = np.zeros((100, 3), dtype=np.int32)
        pairs, triples, info = surrogate_gbm_interaction_seeds(
            D, np.zeros(100, dtype=np.int64), [0], is_classification=True)
        assert pairs == [] and triples == [] and not info["gated"]
