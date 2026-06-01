"""Layer 96 biz_value: O(p^2) -> O(p) + O(p^2)-cheap II pre-filter speedup.

Layer 89 scores ALL C(p, 2) categorical pairs by interaction information
``II(a, b; y) = I(a, b; y) - I(a; y) - I(b; y)``. The naive scorer recomputes
each parent's dense codes, its cardinality, and its marginal ``I(cat; y)`` for
EVERY pair it appears in -- O(p^2) work in the per-column quantities. Layer 96
HOISTS that per-column work out of the pair loop and caches it: each cat's
codes, Horner radix (``card + 1``), and marginal MI are computed exactly once
(O(p)); only the joint ``I(a, b; y)`` stays per-pair (O(p^2) but irreducible).

Since ``II = I(a,b;y) - I(a;y) - I(b;y)``, caching the two marginals removes
2/3 of the redundant MI estimator calls. The speedup comes ENTIRELY from
caching shared computation -- NO candidate pair is dropped. That distinction is
load-bearing: the pure categorical XOR ``y = a XOR b`` has BOTH marginals ~0
yet a large II, so any "prune by marginal MI" filter would silently discard the
single synergy the whole module exists to surface. Contract 3 pins that the
hoisting does NOT prune the XOR pair.

Contracts pinned (real numbers, never xfail):

* Speedup: cached pair scorer >= 1.5x faster than naive per-pair recompute at
  p=30, n=5000.
* Bit-equivalence: cached II identical to naive II (rtol 1e-9, in fact 0.0).
* XOR preserved: marginal hoisting does NOT prune the pure-XOR pair (both
  marginals ~0, II large, pair still scored and top-ranked).
* L89 contracts intact: XOR synergy still recovered, redundancy still negative.
* L94 triple speedup: cached triple scorer (mi_cache) >= 1.5x faster than a
  naive per-term recompute on the same beam.

2026-06-01 Layer 96.
"""
from __future__ import annotations

import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Naive (BEFORE) reference scorers -- recompute per-column quantities per pair
# ---------------------------------------------------------------------------


def _naive_pair_ii(X, y, cat_cols, n_bins: int = 10):
    """BEFORE: recompute each parent's codes / cardinality / marginal MI inside
    the per-pair loop (no cross-pair cache). The shape Layer 96 replaces."""
    from mlframe.feature_selection.filters._cat_pair_fe import _bin_target
    from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str
    from mlframe.feature_selection.filters._adaptive_nbins import _plug_in_mi

    pairs = list(combinations(cat_cols, 2))
    y_bin = _bin_target(y, n_bins=n_bins)
    out: dict[tuple, float] = {}
    for a, b in pairs:
        _, ca = np.unique(_column_to_str(X[a]), return_inverse=True)
        ca = ca.astype(np.int64)
        _, cb = np.unique(_column_to_str(X[b]), return_inverse=True)
        cb = cb.astype(np.int64)
        mi_i = float(_plug_in_mi(ca, y_bin))
        mi_j = float(_plug_in_mi(cb, y_bin))
        jk = ca * (int(cb.max()) + 1) + cb
        _, jc = np.unique(jk, return_inverse=True)
        mi_joint = float(_plug_in_mi(jc.astype(np.int64), y_bin))
        out[(a, b)] = mi_joint - mi_i - mi_j
    return out


def _build_wide_cats(seed: int, p: int = 30, n: int = 5000):
    """p mixed-cardinality categorical columns, n rows; weak/binary target."""
    rng = np.random.default_rng(int(seed))
    cols = {}
    for i in range(p):
        card = int(rng.integers(2, 12))
        cols[f"c{i}"] = rng.integers(0, card, n).astype(str)
    X = pd.DataFrame(cols)
    y = rng.integers(0, 2, n).astype(int)
    return X, y


def _build_cat_xor(seed: int, n: int = 6000):
    """y = cat_a XOR cat_b: both marginals ~0, joint fully predictive."""
    rng = np.random.default_rng(int(seed))
    cat_a = rng.integers(0, 2, n)
    cat_b = rng.integers(0, 2, n)
    flip = rng.random(n) < 0.03
    y = (cat_a ^ cat_b) ^ flip.astype(int)
    X = pd.DataFrame({
        "cat_a": cat_a.astype(str),
        "cat_b": cat_b.astype(str),
        "decoy_0": rng.integers(0, 2, n).astype(str),
        "decoy_1": rng.integers(0, 3, n).astype(str),
    })
    return X, y.astype(int)


def _warm_numba():
    from mlframe.feature_selection.filters._adaptive_nbins import _plug_in_mi
    _ = _plug_in_mi(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]))


# ---------------------------------------------------------------------------
# Contract 1: cached pair scorer >= 1.5x faster than naive recompute
# ---------------------------------------------------------------------------


class TestPairScoringSpeedup:
    def test_cached_at_least_1p5x_faster_at_p30(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            score_cat_pairs_by_interaction_information,
        )
        X, y = _build_wide_cats(0, p=30, n=5000)
        cat_cols = list(X.columns)
        _warm_numba()

        n_iter = 3
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _naive_pair_ii(X, y, cat_cols)
        t_naive = (time.perf_counter() - t0) / n_iter

        t0 = time.perf_counter()
        for _ in range(n_iter):
            score_cat_pairs_by_interaction_information(X, y, cat_cols)
        t_cached = (time.perf_counter() - t0) / n_iter

        speedup = t_naive / max(t_cached, 1e-9)
        assert speedup >= 1.5, (
            f"cached II pair scorer only {speedup:.2f}x faster than naive "
            f"per-pair recompute (naive {t_naive*1000:.1f} ms, cached "
            f"{t_cached*1000:.1f} ms at p=30 n=5000); expected >= 1.5x. The "
            f"O(p^2)->O(p) marginal/code hoisting is not delivering the win."
        )


# ---------------------------------------------------------------------------
# Contract 2: cached II bit-identical to naive II (rtol 1e-9)
# ---------------------------------------------------------------------------


class TestBitEquivalence:
    def test_cached_ii_equals_naive_ii(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            score_cat_pairs_by_interaction_information,
        )
        for seed in (0, 7, 42):
            X, y = _build_wide_cats(seed, p=20, n=4000)
            cat_cols = list(X.columns)
            naive = _naive_pair_ii(X, y, cat_cols)
            sc = score_cat_pairs_by_interaction_information(X, y, cat_cols)
            max_diff = 0.0
            for _, r in sc.iterrows():
                key = (r["cat_i"], r["cat_j"])
                max_diff = max(max_diff, abs(naive[key] - float(r["ii"])))
            assert max_diff < 1e-9, (
                f"seed={seed}: cached II differs from naive II by "
                f"{max_diff:.3e} > 1e-9; the caching changed the numbers. "
                f"Caching must be a pure speed transform, bit-identical."
            )


# ---------------------------------------------------------------------------
# Contract 3: XOR preserved -- marginal hoisting does NOT prune the XOR pair
# ---------------------------------------------------------------------------


class TestXorPreserved:
    def test_pure_xor_pair_not_pruned_by_marginal_hoist(self):
        """The KEY subtlety: pure XOR has BOTH marginals ~0 yet II large. The
        Layer 96 caching must NOT prune a pair by its marginal MI -- it only
        hoists the marginal COMPUTATION out of the loop, it never gates on it.
        Pin that the XOR pair is (a) still scored, (b) has near-zero marginals,
        (c) has a large positive II, and (d) ranks top-1."""
        from mlframe.feature_selection.filters._cat_pair_fe import (
            score_cat_pairs_by_interaction_information,
            engineered_name_cat_pair_cross,
        )
        for seed in (1, 7, 13, 42, 101):
            X, y = _build_cat_xor(seed)
            cat_cols = ["cat_a", "cat_b", "decoy_0", "decoy_1"]
            sc = score_cat_pairs_by_interaction_information(X, y, cat_cols)
            xor_name = engineered_name_cat_pair_cross("cat_a", "cat_b")
            xor_rows = sc[sc["engineered_col"] == xor_name]
            assert len(xor_rows) == 1, (
                f"seed={seed}: the XOR pair was DROPPED from scoring "
                f"({len(xor_rows)} rows). Marginal hoisting must never prune."
            )
            row = xor_rows.iloc[0]
            # Both marginals near-zero -- a naive "needs marginal signal" filter
            # would have killed this pair.
            assert abs(float(row["mi_i"])) < 0.02 and abs(float(row["mi_j"])) < 0.02, (
                f"seed={seed}: XOR parent marginals are not ~0 "
                f"(mi_i={row['mi_i']:.4f}, mi_j={row['mi_j']:.4f}); fixture "
                f"is not a pure-synergy XOR."
            )
            # ...yet the interaction information is large and the pair tops the
            # ranking -- proving the synergy survived the hoist.
            assert float(row["ii"]) > 0.2, (
                f"seed={seed}: XOR pair II={row['ii']:.4f} <= 0.2; the synergy "
                f"was lost by the optimization."
            )
            assert str(sc.iloc[0]["engineered_col"]) == xor_name, (
                f"seed={seed}: XOR pair not top-1 after caching "
                f"(top={sc.iloc[0]['engineered_col']})."
            )


# ---------------------------------------------------------------------------
# Contract 4: L89 contracts intact -- synergy recovered, redundancy negative
# ---------------------------------------------------------------------------


class TestL89ContractsIntact:
    def test_xor_synergy_recovery_still_holds(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            score_cat_pairs_by_interaction_information,
            engineered_name_cat_pair_cross,
        )
        wins = 0
        for s in (1, 7, 13, 42, 101):
            X, y = _build_cat_xor(s)
            sc = score_cat_pairs_by_interaction_information(
                X, y, ["cat_a", "cat_b", "decoy_0", "decoy_1"],
            )
            xor_name = engineered_name_cat_pair_cross("cat_a", "cat_b")
            top3 = list(sc.head(3)["engineered_col"])
            xor_row = sc[sc["engineered_col"] == xor_name].iloc[0]
            if float(xor_row["ii"]) > 0.2 and xor_name in top3:
                wins += 1
        assert wins >= 4, (
            f"post-Layer-96 XOR synergy recovered in top-3 on only {wins}/5 "
            f"seeds; the caching regressed the L89 recovery contract."
        )

    def test_redundant_copy_still_negative(self):
        from mlframe.feature_selection.filters._cat_pair_fe import (
            score_cat_pairs_by_interaction_information,
        )
        for s in (1, 7, 13, 42, 101):
            rng = np.random.default_rng(int(s))
            n = 6000
            cat_a = rng.integers(0, 4, n)
            cat_b = cat_a.copy()
            y = (cat_a >= 2).astype(int) ^ (rng.random(n) < 0.03).astype(int)
            X = pd.DataFrame({"cat_a": cat_a.astype(str), "cat_b": cat_b.astype(str)})
            sc = score_cat_pairs_by_interaction_information(
                X, y.astype(int), ["cat_a", "cat_b"],
            )
            assert float(sc["ii"].iloc[0]) < 0.0, (
                f"seed={s}: redundant copy-cat II {sc['ii'].iloc[0]:.4f} not < 0 "
                f"after caching; redundancy detection regressed."
            )


# ---------------------------------------------------------------------------
# Contract 5: L94 triple scorer caching speedup
# ---------------------------------------------------------------------------


def _naive_triple_beam_ii3(X, y, cat_cols, evaluated_triples, n_bins: int = 10):
    """BEFORE: recompute EVERY MI term (7 per triple: 3 singles, 3 pairs, 1
    triple) for each evaluated triple, with no frozenset mi_cache. The shape
    the Layer 94 mi_cache replaces."""
    from mlframe.feature_selection.filters._cat_pair_fe import _bin_target
    from mlframe.feature_selection.filters._cat_triple_fe import (
        _dense_codes, _join_codes,
    )
    from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str
    from mlframe.feature_selection.filters._adaptive_nbins import _plug_in_mi

    y_bin = _bin_target(y, n_bins=n_bins)

    def codes(col):
        return _dense_codes(_column_to_str(X[col]))

    def mi(cols):
        joint = _join_codes(*(codes(c) for c in cols))
        return float(_plug_in_mi(joint, y_bin))

    out = {}
    for (a, b, c) in evaluated_triples:
        ii3 = (
            mi((a, b, c))
            - (mi((a, b)) + mi((a, c)) + mi((b, c)))
            + (mi((a,)) + mi((b,)) + mi((c,)))
        )
        out[frozenset((a, b, c))] = ii3
    return out


class TestTripleScoringSpeedup:
    def test_cached_triple_at_least_1p5x_faster(self):
        from mlframe.feature_selection.filters._cat_triple_fe import (
            score_cat_triples_by_interaction_information,
        )
        # Wide cat set so the beam evaluates many triples sharing sub-joints
        # (where the frozenset mi_cache pays off).
        rng = np.random.default_rng(7)
        n, p = 5000, 10
        a = rng.integers(0, 2, n)
        b = rng.integers(0, 2, n)
        c = rng.integers(0, 2, n)
        y = (a ^ b ^ c) ^ (rng.random(n) < 0.02).astype(int)
        cols = {"cat_a": a.astype(str), "cat_b": b.astype(str), "cat_c": c.astype(str)}
        for d in range(p - 3):
            cols[f"decoy_{d}"] = rng.integers(0, 2 + d, n).astype(str)
        X = pd.DataFrame(cols)
        y = y.astype(int)
        cat_cols = list(X.columns)
        _warm_numba()

        # Run the cached beam once to capture which triples it evaluates, so the
        # naive baseline scores the SAME candidate set (apples to apples).
        sc = score_cat_triples_by_interaction_information(
            X, y, cat_cols, beam_width=3, top_k_pairs=3,
        )
        evaluated = [
            (str(r["cat_a"]), str(r["cat_b"]), str(r["cat_c"]))
            for _, r in sc.iterrows()
        ]

        n_iter = 3
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _naive_triple_beam_ii3(X, y, cat_cols, evaluated)
        t_naive = (time.perf_counter() - t0) / n_iter

        t0 = time.perf_counter()
        for _ in range(n_iter):
            score_cat_triples_by_interaction_information(
                X, y, cat_cols, beam_width=3, top_k_pairs=3,
            )
        t_cached = (time.perf_counter() - t0) / n_iter

        speedup = t_naive / max(t_cached, 1e-9)
        assert speedup >= 1.5, (
            f"cached triple II3 scorer only {speedup:.2f}x faster than naive "
            f"per-term recompute (naive {t_naive*1000:.1f} ms, cached "
            f"{t_cached*1000:.1f} ms); expected >= 1.5x. The frozenset mi_cache "
            f"is not delivering the sub-joint reuse win."
        )
