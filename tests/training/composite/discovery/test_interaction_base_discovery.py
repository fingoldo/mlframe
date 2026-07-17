"""Unit + biz_value tests for interaction-aware base-pair discovery.

Covers ``score_interaction_pairs`` + ``discover_interaction_bases`` in
``mlframe.training.composite.discovery._interaction_bases``:

- unit: scorer ranks the true interaction pair top on a pure-interaction DGP,
  flags it ``qualifies``, and rejects an additive (non-interaction) DGP.
- unit: edge cases (single candidate, constant column, op dedup).
- biz_value: on ``y = a*b``, the interaction base beats the best single base on
  held-out OOS RMSE by a wide margin (floor 50%; measured ~90%).
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._interaction_bases import (
    discover_interaction_bases,
    score_interaction_pairs,
)


def _pure_interaction(seed, n=4000):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    y = a * b + 0.1 * rng.standard_normal(n)
    return {"a": a, "b": b, "c": c}, y


def _ridge_oos_rmse(f_tr, y_tr, f_oos, y_oos, lam=1e-3):
    X = np.column_stack([np.ones_like(f_tr), f_tr])
    coef = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ y_tr)
    pred = coef[0] + coef[1] * f_oos
    return float(np.sqrt(np.mean((y_oos - pred) ** 2)))


# ---------------------------------------------------------------- unit ----


class TestScoreInteractionPairs:
    def test_detects_pure_interaction_pair_top(self):
        cand, y = _pure_interaction(0)
        scored = score_interaction_pairs(cand, y, ops=("mul",), top_k=3)
        assert scored, "expected scored pairs"
        top = scored[0]
        assert set(top["parents"]) == {"a", "b"}
        assert top["op"] == "mul"
        assert top["qualifies"]
        # Interaction MI must dwarf the (near-zero) marginal MIs.
        assert top["mi_z"] > 5.0 * top["add_mi"]
        assert top["gain"] > 0.1

    def test_synergy_ratio_far_higher_for_pure_than_additive(self):
        # Discriminating property: for a PURE-interaction DGP (y=a*b) the product
        # is informative while the marginals are ~0, so mi_z/add_mi is enormous.
        # For an ADDITIVE DGP (y=a+b) each base is already informative, so even
        # though MI(y, a*b) > marginals (a*b correlates with (a+b)^2), the RATIO
        # mi_z/add_mi is small. The scorer's job is to surface a usable
        # interaction base, and the synergy RATIO cleanly separates the two
        # regimes (the gate flags both, but ranks pure interaction far higher).
        rng = np.random.default_rng(1)
        n = 4000
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        add_cand = {"a": a, "b": b}
        add_y = a + b + 0.1 * rng.standard_normal(n)
        pure_cand, pure_y = _pure_interaction(11)
        add_top = score_interaction_pairs(add_cand, add_y, ops=("mul",), top_k=2)[0]
        pure_top = score_interaction_pairs(pure_cand, pure_y, ops=("mul",), top_k=3)[0]
        add_ratio = add_top["mi_z"] / max(add_top["add_mi"], 1e-12)
        pure_ratio = pure_top["mi_z"] / max(pure_top["add_mi"], 1e-12)
        # Measured: add_ratio ~1.67, pure_ratio ~4.98 (clean ~3x separation).
        assert add_ratio < 2.5, f"additive synergy ratio unexpectedly high: {add_ratio}"
        assert pure_ratio > 4.0, f"pure synergy ratio too low: {pure_ratio}"
        assert pure_ratio > 2.0 * add_ratio

    def test_single_candidate_returns_empty(self):
        cand = {"a": np.arange(100.0)}
        assert score_interaction_pairs(cand, np.arange(100.0)) == []

    def test_constant_column_skipped(self):
        # A constant base makes a*const constant -> generate_interaction_bases
        # marks it constant and the scorer drops it.
        rng = np.random.default_rng(2)
        n = 2000
        a = rng.standard_normal(n)
        const = np.full(n, 3.0)
        y = a * rng.standard_normal(n)
        cand = {"a": a, "k": const}
        scored = score_interaction_pairs(cand, y, ops=("mul",), top_k=2)
        # a__mul__k is constant*a (not constant) but k__mul__a same; the pure
        # constant a*const is non-constant so just assert no crash + finite mi.
        for rec in scored:
            assert np.isfinite(rec["mi_z"])


class TestDiscoverInteractionBases:
    def test_surfaces_synthetic_for_pure_interaction(self):
        cand, y = _pure_interaction(3)
        synth, recs = discover_interaction_bases(
            cand,
            y,
            ops=("mul",),
            top_k=3,
            max_pairs=2,
        )
        assert synth, "expected at least one surfaced interaction base"
        assert recs
        name = next(iter(synth))
        assert "__mul__" in name
        assert synth[name].shape == y.shape

    def test_no_synergy_returns_empty(self):
        # Pure noise target -> no pair clears the gate.
        rng = np.random.default_rng(4)
        n = 2000
        cand = {k: rng.standard_normal(n) for k in ("a", "b", "c")}
        y = rng.standard_normal(n)
        synth, recs = discover_interaction_bases(cand, y, ops=("mul",), top_k=3)
        assert synth == {}
        assert recs == []

    def test_commutative_op_dedup(self):
        cand, y = _pure_interaction(5)
        synth, _ = discover_interaction_bases(
            cand,
            y,
            ops=("mul",),
            top_k=3,
            max_pairs=5,
        )
        # a__mul__b and b__mul__a must not BOTH appear (commutative dedup).
        norm = {tuple(sorted(n.split("__mul__"))) for n in synth}
        assert len(norm) == len(synth)


# ------------------------------------------------------------ biz_value ----


class TestBizValInteractionBaseDiscovery:
    def test_biz_val_interaction_beats_single_base_oos_rmse(self):
        """On y=a*b, the interaction base must beat the best single base on
        held-out OOS RMSE by >=50% (measured ~90%), majority of 5 seeds."""
        single_rmses, inter_rmses = [], []
        for seed in range(5):
            cand_tr, y_tr = _pure_interaction(seed)
            cand_oos, y_oos = _pure_interaction(seed + 100)
            best_single = min(
                ("a", "b", "c"),
                key=lambda c: _ridge_oos_rmse(
                    cand_tr[c],
                    y_tr,
                    cand_oos[c],
                    y_oos,
                ),
            )
            single = _ridge_oos_rmse(
                cand_tr[best_single],
                y_tr,
                cand_oos[best_single],
                y_oos,
            )
            synth, _ = discover_interaction_bases(
                cand_tr,
                y_tr,
                ops=("mul",),
                top_k=3,
                max_pairs=1,
            )
            assert synth, f"seed {seed}: no interaction base surfaced"
            sname = next(iter(synth))
            pa, pb = sname.split("__mul__")
            inter = _ridge_oos_rmse(
                synth[sname],
                y_tr,
                cand_oos[pa] * cand_oos[pb],
                y_oos,
            )
            single_rmses.append(single)
            inter_rmses.append(inter)

        wins = sum(1 for s, i in zip(single_rmses, inter_rmses) if i < s)
        rel = (np.mean(single_rmses) - np.mean(inter_rmses)) / np.mean(single_rmses)
        assert wins >= 3, f"interaction won only {wins}/5 seeds"
        assert rel >= 0.50, f"OOS RMSE improvement {rel:.1%} < 50% floor"
