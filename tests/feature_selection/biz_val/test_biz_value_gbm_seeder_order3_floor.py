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

    def test_3way_needle_co_occurrence_ranks_it_and_clears_order3_floor(self):
        # The hard sign(a*b*c) 3-way is the CO-OCCURRENCE win: a depth-4 GBM's held-out
        # ACCURACY on it is ~chance (so the OOF self-gate cannot gate it), but its SPLIT
        # co-occurrence still ranks the operands top -- and the ORDER-3 maxT FLOOR (#7, the
        # binding noise guard per "proposer generates, floors gate") then validates the
        # needle clears while noise triples do not. We force-emit (lenient gate) to test the
        # co-occurrence + floor, the actual binding pair.
        from itertools import combinations
        X, y, needle = self._frame("3way")
        D, nbins = _discretize(X, nb=10)
        # CURRENT: univariate seed_count top-8 misses every needle operand.
        uni_top, mis = _univariate_topN(D, y, nbins, N=8)
        assert len(set(needle) & uni_top) == 0, (
            f"fixture invalid: univariate seed already sees needle {set(needle) & uni_top}"
        )
        # GBM co-occurrence (force-emit so the FLOOR is the discriminator, not the OOF gate).
        pairs, triples, info = surrogate_gbm_interaction_seeds(
            D, y, list(range(D.shape[1])), is_classification=True,
            top_k_pairs=15, top_k_triples=10, self_gate_min_z=-1e9, random_seed=0)
        tw = info.get("triple_weights", {})
        ranked = [tuple(sorted(k)) for k, _ in sorted(tw.items(), key=lambda kv: kv[1], reverse=True)]
        assert tuple(sorted(needle)) in ranked[:3], (
            f"3-way needle not in the top-3 co-occurrence triples: {ranked[:5]}"
        )
        # ORDER-3 maxT floor over the seeded triples: the needle clears it.
        ta = np.fromiter((t[0] for t in triples), np.int64, len(triples))
        tb = np.fromiter((t[1] for t in triples), np.int64, len(triples))
        tc = np.fromiter((t[2] for t in triples), np.int64, len(triples))
        fy = np.bincount(y, minlength=int(np.unique(y).size)).astype(np.float64) / len(y)
        floor = pooled_triple_permutation_null_joint_mi_floor(
            D, nbins, ta, tb, tc, y, fy, n_permutations=25, quantile=0.95, random_seed=0)
        obs = batch_triple_mi_prange(D, ta, tb, tc, nbins, y, fy)
        survivors = [tuple(sorted(triples[i])) for i in range(len(triples)) if obs[i] >= floor]
        assert tuple(sorted(needle)) in survivors, (
            f"3-way needle did not clear the order-3 floor (={floor}); survivors={survivors}"
        )

    def test_2way_needle_recovered_univariate_misses(self):
        # The 2-way needle is the SELF-GATE + co-occurrence win: its OOF is FAR above chance
        # (z huge), so the self-gate passes AND it is the top co-occurrence pair.
        X, y, needle = self._frame("2way")
        D, nbins = _discretize(X, nb=10)
        uni_top, mis = _univariate_topN(D, y, nbins, N=8)
        assert len(set(needle) & uni_top) == 0, "fixture invalid: univariate already sees needle"
        pairs, triples, info = surrogate_gbm_interaction_seeds(
            D, y, list(range(D.shape[1])), is_classification=True,
            top_k_pairs=15, top_k_triples=10, random_seed=0)
        assert info["gated"], f"self-gate should pass on a strong 2-way signal: {info}"
        assert tuple(sorted(needle)) in [tuple(sorted(p_)) for p_ in pairs], (
            f"exact 2-way needle not in seeded pairs: {pairs}"
        )

    def test_pure_noise_pair_gate_blocks_pairs_and_floor_curbs_triples(self):
        # The production noise-guard CHAIN on a pure-noise frame:
        #   (1) the OOF PAIR self-gate rejects pair emission (noise OOF ~ permuted null) -- so
        #       NO seeded noise pairs ever reach the order-2 floor in the real pipeline.
        #   (2) the order-3 maxT floor strongly CURBS the always-emitted noise triples.
        X, y, _ = self._frame("noise")
        D, nbins = _discretize(X, nb=10)
        # DEFAULT gate (no force-emit): the pair self-gate must NOT pass on pure noise -> 0 pairs.
        pairs, triples, info = surrogate_gbm_interaction_seeds(
            D, y, list(range(D.shape[1])), is_classification=True,
            top_k_pairs=15, top_k_triples=10, random_seed=0)
        assert not info["gated"], f"pair self-gate must FAIL on pure noise: {info}"
        assert pairs == [], f"no pairs should be emitted on pure noise (gate failed): {pairs}"
        # Triples ARE emitted (the OOF gate is blind to hard 3-way); the order-3 floor curbs them.
        fy = np.bincount(y, minlength=2).astype(np.float64) / len(y)
        ta = np.fromiter((t[0] for t in triples), np.int64, len(triples))
        tb = np.fromiter((t[1] for t in triples), np.int64, len(triples))
        tc = np.fromiter((t[2] for t in triples), np.int64, len(triples))
        floor3 = pooled_triple_permutation_null_joint_mi_floor(
            D, nbins, ta, tb, tc, y, fy, n_permutations=25, quantile=0.95, random_seed=0)
        obs3 = batch_triple_mi_prange(D, ta, tb, tc, nbins, y, fy)
        n_trip_survive = int(np.sum(obs3 >= floor3))
        # q95 over a proposer-selected (joint-MI-enriched) pool of <=10 triples admits a small
        # best-of-pool residual; the floor must curb the noise to AT MOST a couple of triples
        # (no noise CLOUD). The downstream per-triplet uplift / abs-MI gates then reject those
        # residual noise triples as engineered features (the e2e confirms pure noise engineers
        # nothing end-to-end).
        assert n_trip_survive <= 2, (
            f"order-3 floor admitted {n_trip_survive} pure-noise triples (floor={floor3}); "
            f"expected <=2 (q95 best-of-pool residual over a proposer-selected pool)"
        )

    def test_degenerate_pool_no_seeds(self):
        # < 2 candidates -> empty, no LightGBM fit.
        D = np.zeros((100, 3), dtype=np.int32)
        pairs, triples, info = surrogate_gbm_interaction_seeds(
            D, np.zeros(100, dtype=np.int64), [0], is_classification=True)
        assert pairs == [] and triples == [] and not info["gated"]
