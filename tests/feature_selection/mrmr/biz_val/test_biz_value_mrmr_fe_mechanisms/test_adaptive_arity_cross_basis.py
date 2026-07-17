"""Layer 78 biz_value: ADAPTIVE-ARITY cross-basis ORTHOGONAL-POLY FE.

Consolidated verbatim from test_biz_value_mrmr_layer78.py (per audit finding test_code_quality-16).

Validates ``hybrid_orth_mi_adaptive_arity_fe`` introduced 2026-06-01
(sibling module ``_orthogonal_adaptive_arity_fe``). Layers 22 (pair) /
56 (triplet) / 77 (quadruplet) each fix arity at construction time;
Layer 78 tries arities 2/3/4 per seed tuple and emits ONLY the winning
arity per maximal signal set.

Contracts pinned
----------------

* TestArityDiscovery2Way:    y = sign(x1*x2)     -> adaptive picks 2-way
* TestArityDiscovery3Way:    y = sign(x1*x2*x3)  -> adaptive picks 3-way
* TestArityDiscovery4Way:    y = sign(x1*x2*x3*x4) -> adaptive picks 4-way
* TestCombinatorialBound:    candidate count <= documented cap
* TestDefaultDisabledByteIdentical: master switch OFF leaves
  ``support_`` / ``feature_names_in_`` clear of adaptive-arity columns
* TestPickleAndClone:         clone + pickle round-trips ctor params

NEVER xfail.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_aa_fe():
    """Lazily import the adaptive-arity cross-basis orthogonal-polynomial FE functions."""
    from mlframe.feature_selection.filters._orthogonal_adaptive_arity_fe import (
        generate_adaptive_arity_cross_basis,
        score_adaptive_arity_cross_basis,
        hybrid_orth_mi_adaptive_arity_fe,
    )

    return (
        generate_adaptive_arity_cross_basis,
        score_adaptive_arity_cross_basis,
        hybrid_orth_mi_adaptive_arity_fe,
    )


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr

# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_xor2(seed: int, n: int = 3000):
    """``y = sign(x1*x2)`` -- pure 2-way XOR. Adding x3/x4 to the seed
    pool must NOT cause a 3-way / 4-way cell to be emitted, because the
    triplet / quadruplet MI does not exceed the pair MI.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
        }
    )
    y = ((x1 * x2) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_xor3(seed: int, n: int = 4000):
    """``y = sign(x1*x2*x3)`` -- pure 3-way XOR. The 2-way cells over
    any pair carry zero marginal MI; only the (x1,x2,x3) triplet wins.
    The (x1,x2,x3,x4) quadruplet's MI equals the triplet's (x4 is
    random and integrates to a constant), so the higher-arity cell does
    NOT strictly beat the triplet -- adaptive emits the triplet.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
        }
    )
    y = ((x1 * x2 * x3) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_xor4(seed: int, n: int = 5000):
    """``y = sign(x1*x2*x3*x4)`` -- pure 4-way XOR. Every triplet
    marginal MI is zero; only the 4-way cell carries signal.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
        }
    )
    y = ((x1 * x2 * x3 * x4) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1500):
    """Plain linear-additive signal. Used for the default-disabled
    byte-identical contract.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
        }
    )
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract: 2-way XOR -> adaptive picks 2-way for the winning tuple
# ---------------------------------------------------------------------------


class TestArityDiscovery2Way:
    """``y = sign(x1*x2)`` (2-way XOR) makes adaptive-arity pick the 2-way pair, never a higher-arity superset."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_adaptive_picks_arity_2(self, seed):
        """The {x1,x2} pair is the top-uplift winner at arity 2; no higher-arity superset of it is emitted."""
        gen_aa, _, _ = _import_aa_fe()
        X, y = _build_xor2(seed)
        _eng_X, scores = gen_aa(
            X,
            y.values,
            source_cols=["x1", "x2", "x3", "x4"],
            max_arity=4,
            max_degree=1,
            basis="hermite",
        )
        assert not scores.empty, f"seed={seed}: adaptive-arity must emit at least the 2-way winner (x1,x2)"
        # The (x1, x2) pair must be the top-uplift winner at arity 2.
        # No 3-way / 4-way cell that includes {x1, x2} should be emitted:
        # its MI does not beat the pair MI on a pure 2-way XOR target.
        top = scores.iloc[0]
        assert top["arity"] == 2, f"seed={seed}: top winner arity should be 2, got {top['arity']}; scores:\n{scores}"
        assert set(top["source_cols"]) == {"x1", "x2"}, f"seed={seed}: top winner tuple should be {{x1,x2}}, got {top['source_cols']}; scores:\n{scores}"
        # No emitted cell should be a strict SUPERSET of {x1, x2}:
        # adaptive eclipses higher-arity dilutions.
        offenders = [row for _, row in scores.iterrows() if row["arity"] > 2 and {"x1", "x2"}.issubset(set(row["source_cols"]))]
        assert not offenders, (
            f"seed={seed}: adaptive-arity emitted a higher-arity superset of {{x1,x2}} on a pure 2-way XOR signal: {offenders}; scores:\n{scores}"
        )


# ---------------------------------------------------------------------------
# Contract: 3-way XOR -> adaptive picks 3-way
# ---------------------------------------------------------------------------


class TestArityDiscovery3Way:
    """``y = sign(x1*x2*x3)`` (3-way XOR) makes adaptive-arity pick the (x1,x2,x3) triplet, never a 4-way superset."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_adaptive_picks_arity_3(self, seed):
        """The (x1,x2,x3) triplet wins at arity 3; no 4-way superset of it is emitted."""
        gen_aa, _, _ = _import_aa_fe()
        X, y = _build_xor3(seed)
        _eng_X, scores = gen_aa(
            X,
            y.values,
            source_cols=["x1", "x2", "x3", "x4"],
            max_arity=4,
            max_degree=1,
            basis="hermite",
        )
        # Find the row whose source_cols == {x1, x2, x3} (the true signal).
        signal_rows = [row for _, row in scores.iterrows() if set(row["source_cols"]) == {"x1", "x2", "x3"}]
        assert signal_rows, f"seed={seed}: adaptive-arity must emit the (x1,x2,x3) triplet on a 3-way XOR target; scores:\n{scores}"
        row = signal_rows[0]
        assert row["arity"] == 3, f"seed={seed}: (x1,x2,x3) should win at arity 3, got {row['arity']}; scores:\n{scores}"
        # No emitted superset {x1, x2, x3, x4} -- the 4-way dilution does
        # NOT strictly beat the triplet (x4 randomises and integrates out).
        offenders = [row for _, row in scores.iterrows() if row["arity"] == 4 and {"x1", "x2", "x3"}.issubset(set(row["source_cols"]))]
        assert not offenders, f"seed={seed}: adaptive-arity emitted a 4-way superset of (x1,x2,x3) on a pure 3-way XOR signal: {offenders}; scores:\n{scores}"


# ---------------------------------------------------------------------------
# Contract: 4-way XOR -> adaptive picks 4-way
# ---------------------------------------------------------------------------


class TestArityDiscovery4Way:
    """``y = sign(x1*x2*x3*x4)`` (4-way XOR) makes adaptive-arity pick the full 4-way quadruplet as the top MI winner."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_adaptive_picks_arity_4(self, seed):
        """The (x1,x2,x3,x4) quadruplet is the top engineered-MI winner, with substantial MI."""
        gen_aa, _, _ = _import_aa_fe()
        X, y = _build_xor4(seed)
        _eng_X, scores = gen_aa(
            X,
            y.values,
            source_cols=["x1", "x2", "x3", "x4"],
            max_arity=4,
            max_degree=1,
            basis="hermite",
        )
        # The top engineered_mi winner should be the quadruplet (x1..x4).
        top = scores.sort_values("engineered_mi", ascending=False).iloc[0]
        assert set(top["source_cols"]) == {"x1", "x2", "x3", "x4"}, (
            f"seed={seed}: top MI cell on 4-way XOR should be the quadruplet, got {top['source_cols']}; scores:\n{scores}"
        )
        assert top["arity"] == 4, f"seed={seed}: arity should be 4, got {top['arity']}; scores:\n{scores}"
        assert top["engineered_mi"] >= 0.2, f"seed={seed}: 4-way XOR engineered_mi {top['engineered_mi']:.3f} should clear 0.2 at n=5000"


# ---------------------------------------------------------------------------
# Contract: combinatorial bound
# ---------------------------------------------------------------------------


class TestCombinatorialBound:
    """The emitted cell count matches the combinatorial bound: sum of C(seed_k, arity) for arity in [2, max_arity]."""

    def test_seed_k_4_max_arity_3_at_most_10_tuples(self):
        """4 sources at max_arity=3 (C(4,2)+C(4,3)=10 candidates) emits at most 10 cells."""
        gen_aa, _, _ = _import_aa_fe()
        # 4 sources, max_arity=3 -> C(4,2)+C(4,3) = 6+4 = 10 candidate
        # tuples evaluated. With max_degree=1 -> exactly 10 cells before
        # the adaptive prune; after prune the kept count is <= 10.
        rng = np.random.default_rng(0)
        n = 1000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        # random y -- the MI gates will keep at most all 10, never more.
        y = (rng.standard_normal(n) > 0).astype(int)
        eng_X, _scores = gen_aa(
            X,
            y,
            source_cols=list(X.columns),
            max_arity=3,
            max_degree=1,
            basis="hermite",
        )
        assert eng_X.shape[1] <= 10, f"seed_k=4 + max_arity=3 should produce <= 10 emitted cells, got {eng_X.shape[1]}"

    def test_seed_k_4_max_arity_4_at_most_11_tuples(self):
        """4 sources at max_arity=4 (C(4,2)+C(4,3)+C(4,4)=11 candidates) emits at most 11 cells."""
        gen_aa, _, _ = _import_aa_fe()
        # C(4,2)+C(4,3)+C(4,4) = 6+4+1 = 11.
        rng = np.random.default_rng(0)
        n = 1000
        X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(1, 5)})
        y = (rng.standard_normal(n) > 0).astype(int)
        eng_X, _ = gen_aa(
            X,
            y,
            source_cols=list(X.columns),
            max_arity=4,
            max_degree=1,
            basis="hermite",
        )
        assert eng_X.shape[1] <= 11, f"seed_k=4 + max_arity=4 should produce <= 11 emitted cells, got {eng_X.shape[1]}"


# ---------------------------------------------------------------------------
# Contract: default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_adaptive_arity_enable=False (the default) leaves feature_names_in_ / hybrid_orth_features_ clear."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_adaptive_arity_columns(self, seed):
        """Default fe_hybrid_orth_adaptive_arity_enable=False injects no cross columns."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        # No cross columns (>=1 '*') surfaced in feature_names_in_.
        names = list(m.feature_names_in_)
        cross_names = [n for n in names if "*" in str(n)]
        assert not cross_names, (
            f"seed={seed}: default fe_hybrid_orth_adaptive_arity_enable=False should NOT inject adaptive-arity cross columns; got {cross_names}"
        )
        assert list(getattr(m, "hybrid_orth_features_", []) or []) == [], f"seed={seed}: hybrid_orth_features_ should be empty when all hybrid flags are off"

    @pytest.mark.parametrize("seed", (1, 7, 13))
    def test_enable_adaptive_arity_appends_engineered(self, seed):
        """fe_hybrid_orth_adaptive_arity_enable=True appends at least one cross column to hybrid_orth_features_."""
        X, y = _build_xor3(seed, n=3000)
        m = _make_mrmr(
            fe_hybrid_orth_adaptive_arity_enable=True,
            fe_hybrid_orth_adaptive_arity_max_arity=3,
            fe_hybrid_orth_adaptive_arity_max_degree=1,
            fe_hybrid_orth_adaptive_arity_seed_k=4,
            fe_hybrid_orth_adaptive_arity_top_count=3,
        ).fit(X, y)
        added = [n for n in (getattr(m, "hybrid_orth_features_", None) or []) if "*" in str(n)]
        assert added, (
            f"seed={seed}: adaptive-arity flag ON should append at least one cross column to hybrid_orth_features_; got {list(m.hybrid_orth_features_ or [])}"
        )


# ---------------------------------------------------------------------------
# Contract: pickle / clone preserve adaptive-arity ctor params + recipes
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """clone() preserves the adaptive-arity ctor params; pickle round-trips a fitted MRMR with adaptive-arity recipes."""

    def test_clone_preserves_adaptive_arity_params(self):
        """clone() copies all fe_hybrid_orth_adaptive_arity_* params without carrying over fitted state."""
        m = _make_mrmr(
            fe_hybrid_orth_adaptive_arity_enable=True,
            fe_hybrid_orth_adaptive_arity_max_arity=4,
            fe_hybrid_orth_adaptive_arity_max_degree=2,
            fe_hybrid_orth_adaptive_arity_seed_k=5,
            fe_hybrid_orth_adaptive_arity_top_count=4,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_adaptive_arity_enable", True),
            ("fe_hybrid_orth_adaptive_arity_max_arity", 4),
            ("fe_hybrid_orth_adaptive_arity_max_degree", 2),
            ("fe_hybrid_orth_adaptive_arity_seed_k", 5),
            ("fe_hybrid_orth_adaptive_arity_top_count", 4),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got {getattr(m2, name)}"

    def test_pickle_roundtrip_fitted_with_adaptive_arity(self):
        """pickle.dumps/loads preserves feature_names_in_ and hybrid_orth_features_ for an adaptive-arity-enabled fit."""
        X, y = _build_xor3(seed=42, n=2500)
        m = _make_mrmr(
            fe_hybrid_orth_adaptive_arity_enable=True,
            fe_hybrid_orth_adaptive_arity_max_arity=3,
            fe_hybrid_orth_adaptive_arity_max_degree=1,
            fe_hybrid_orth_adaptive_arity_seed_k=4,
            fe_hybrid_orth_adaptive_arity_top_count=3,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        assert list(getattr(m2, "hybrid_orth_features_", []) or []) == list(getattr(m, "hybrid_orth_features_", []) or []), (
            "pickle changed hybrid_orth_features_"
        )
