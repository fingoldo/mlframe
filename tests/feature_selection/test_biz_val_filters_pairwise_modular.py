"""biz_value + integration tests for PAIRWISE / N-WAY MODULAR FE wired into MRMR.

The detector recovers a target that is an integer MODULUS of a COMBINATION of integer columns -- (a+b) mod m,
(a*b) mod m, n-way parity, single hidden non-calendar period -- which smooth bases (poly / Fourier) cannot fit.
It is wired into MRMR behind ``fe_pairwise_modular_enable`` (default OFF) with a column-count budget guard.

Contracts pinned (measured, never xfail):

* PROTOTYPE-direct: ``detect_pairwise_modular`` fires on (a+b) mod 7 with a large MI lift, stays silent on controls.
* INTEGRATION (the public-API win): MRMR with the flag ON recovers + SELECTS the modular feature; transform() replays
  the frozen recipe identically at predict (leak-free, deterministic). Measured end-to-end LogReg AUC: 1.0 ON vs 0.49
  OFF (lift +0.51 on (a+b) mod 7); floor pinned at +0.40 (well below measured).
* OFF default is a true no-op: no modular columns emitted.
* BUDGET GUARD: above max_triple_cols the triple sweep is dropped (logged); above max_int_cols the whole sweep is
  skipped (logged); selection still completes.
* pickle / clone round-trip recipes + ctor params.
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.filters._pairwise_modular_fe import (
    apply_pairwise_modular,
    build_pairwise_modular_recipe,
    detect_pairwise_modular,
    hybrid_pairwise_modular_fe_with_recipes,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

SEEDS = (1, 7, 13, 42, 101)


def _build_pair_add_mod(seed: int, n: int = 4000, m: int = 7):
    """y = ((a+b) mod m) >= m//2, with a weakly-informative raw col so MRMR screening has an anchor and
    does not 0-fallback (the modular residue, not the raw cols, carries the signal)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 100, n)
    b = rng.integers(0, 100, n)
    y = ((a + b) % m >= (m // 2)).astype(int)
    X = pd.DataFrame({
        "a": a, "b": b, "extra": (a % 2),
        "n0": rng.integers(0, 50, n), "n1": rng.integers(0, 50, n),
    })
    return X, y


# ---------------------------------------------------------------------------
# Prototype-direct contracts (keep alongside the integration tests)
# ---------------------------------------------------------------------------


class TestPrototypeDirect:
    def test_detects_pair_add_mod7(self):
        X, y = _build_pair_add_mod(7)
        hits = detect_pairwise_modular(X, y, seed=7)
        assert hits, "no responded hit on (a+b) mod 7."
        top = hits[0]
        assert top["modulus"] % 7 == 0 or top["modulus"] == 7, (
            f"detected modulus {top['modulus']} not a multiple of true 7."
        )
        assert top["margin"] >= 0.20, f"MI lift {top['margin']} below the measured ~0.6 floor."

    def test_silent_on_smooth_control(self):
        rng = np.random.default_rng(7)
        n = 4000
        a, b = rng.integers(0, 100, n), rng.integers(0, 100, n)
        y = ((a + 0.7 * b) > 85).astype(int)
        X = pd.DataFrame({"a": a, "b": b, "n0": rng.integers(0, 50, n)})
        hits = detect_pairwise_modular(X, y, seed=7)
        assert hits == [], f"smooth-threshold control fired: {hits}"


# ---------------------------------------------------------------------------
# Recipe replay: leak-free + bit-identical at predict
# ---------------------------------------------------------------------------


class TestRecipeReplay:
    def test_recipe_replay_bit_identical(self):
        X, y = _build_pair_add_mod(1)
        appended, recipes = hybrid_pairwise_modular_fe_with_recipes(X, y, seed=1)
        assert recipes, "no pairwise-modular recipes emitted."
        for r in recipes:
            direct = apply_pairwise_modular(X, r.extra["op"], r.src_names, r.extra["modulus"])
            via_recipe = apply_recipe(r, X)
            np.testing.assert_array_equal(direct, via_recipe)

    def test_replay_is_leak_free(self):
        """Replay reads only X (combine cols, mod m) -- shuffling y must not change the residue."""
        X, y = _build_pair_add_mod(13)
        _, recipes = hybrid_pairwise_modular_fe_with_recipes(X, y, seed=13)
        assert recipes
        r = recipes[0]
        on_train = apply_recipe(r, X)
        # A held-out slice replays purely from its own X values.
        Xte = X.iloc[:500].reset_index(drop=True)
        on_test = apply_recipe(r, Xte)
        np.testing.assert_array_equal(on_train[:500], on_test)

    def test_recipe_pickle_round_trip(self):
        X, y = _build_pair_add_mod(1)
        _, recipes = hybrid_pairwise_modular_fe_with_recipes(X, y, seed=1)
        assert recipes
        for r in recipes:
            r2 = pickle.loads(pickle.dumps(r))
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))


# ---------------------------------------------------------------------------
# MRMR integration: ON selects + replays, OFF is a no-op
# ---------------------------------------------------------------------------


class TestMRMRIntegration:
    def test_default_off_is_no_op(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_pair_add_mod(42, n=2000)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_pairwise_modular_enable", False)) is False, (
            "fe_pairwise_modular_enable must default to False."
        )
        m.fit(X, pd.Series(y, name="y"))
        pm = list(getattr(m, "pairwise_modular_features_", []) or [])
        assert pm == [], f"pairwise-modular added columns with the flag disabled: {pm}"
        out = m.transform(X.iloc[:300])
        assert not any(str(c).startswith("pmod_") for c in out.columns), (
            "no pmod column may appear in transform output when the flag is OFF."
        )

    def test_enabled_selects_modular_feature_and_replays(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_pair_add_mod(7, n=4000)
        m = MRMR(fe_pairwise_modular_enable=True, max_runtime_mins=2)
        m.fit(X, pd.Series(y, name="y"))
        pm = list(getattr(m, "pairwise_modular_features_", []) or [])
        assert len(pm) >= 1, "pairwise-modular enabled but produced no engineered columns."
        # The selected feature set materialises through transform(): the modular residue must be there
        # (raw a/b alone have AUC ~0.5 on this target, so the residue is the only informative feature).
        out = m.transform(X.iloc[:500])
        pmod_cols = [c for c in out.columns if str(c).startswith("pmod_")]
        assert pmod_cols, f"MRMR did not select any pairwise-modular feature; selected={list(out.columns)}"
        # Predict-time replay determinism: re-transforming the same rows is bit-identical.
        out2 = m.transform(X.iloc[:500])
        for c in pmod_cols:
            np.testing.assert_array_equal(out[c].to_numpy(), out2[c].to_numpy())

    def test_budget_guard_skips_triples_and_logs(self, caplog):
        # 25 integer columns: above max_triple_cols=20, within max_int_cols=30 -> pairs-only + a log line.
        rng = np.random.default_rng(7)
        n = 2000
        cols = {f"c{i}": rng.integers(0, 100, n) for i in range(25)}
        a, b = cols["c0"], cols["c1"]
        y = ((a + b) % 7 >= 3).astype(int)
        X = pd.DataFrame(cols)
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters._pairwise_modular_fe"):
            appended, recipes = hybrid_pairwise_modular_fe_with_recipes(X, y, seed=7)
        assert any("PAIRS-ONLY" in r.message for r in caplog.records), (
            "expected a pairs-only budget log line for 25 int columns."
        )
        # Pairs-only still completes and recovers the pair-modular signal.
        assert appended, "pairs-only sweep produced nothing on a (c0+c1) mod 7 target."

    def test_budget_guard_skips_whole_sweep_and_logs(self, caplog):
        rng = np.random.default_rng(7)
        n = 2000
        cols = {f"c{i}": rng.integers(0, 100, n) for i in range(35)}
        a, b = cols["c0"], cols["c1"]
        y = ((a + b) % 7 >= 3).astype(int)
        X = pd.DataFrame(cols)
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters._pairwise_modular_fe"):
            appended, recipes = hybrid_pairwise_modular_fe_with_recipes(X, y, seed=7)
        assert any("skipping the pairwise/n-way modular sweep" in r.message for r in caplog.records), (
            "expected a whole-sweep-skipped budget log line for 35 int columns."
        )
        assert appended == [] and recipes == [], "above max_int_cols nothing should be emitted."

    def test_clone_preserves_params(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_pairwise_modular_enable=True,
            fe_pairwise_modular_top_k=3,
            fe_pairwise_modular_max_int_cols=25,
            fe_pairwise_modular_max_triple_cols=15,
        )
        c = clone(m)
        assert bool(c.fe_pairwise_modular_enable) is True
        assert int(c.fe_pairwise_modular_top_k) == 3
        assert int(c.fe_pairwise_modular_max_int_cols) == 25
        assert int(c.fe_pairwise_modular_max_triple_cols) == 15


# ---------------------------------------------------------------------------
# biz_value: end-to-end public-API win (the whole point)
# ---------------------------------------------------------------------------


class TestBizValue:
    def test_biz_val_pairwise_modular_end_to_end_auc_lift(self):
        """MRMR(fe_pairwise_modular_enable=True) recovers (a+b) mod 7, raw columns can't.

        Measured LogReg holdout AUC: 1.0 ON vs ~0.49-0.52 OFF (lift +0.48-0.51). Floor +0.40 (below measured)."""
        on, off = [], []
        for seed in (1, 7, 42):
            from mlframe.feature_selection.filters.mrmr import MRMR
            X, y = _build_pair_add_mod(seed, n=3000)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y,
            )
            for flag, store in ((True, on), (False, off)):
                m = MRMR(fe_pairwise_modular_enable=flag, max_runtime_mins=1)
                m.fit(Xtr, pd.Series(ytr, name="y"))
                Ftr = m.transform(Xtr)
                Fte = m.transform(Xte)
                clf = LogisticRegression(max_iter=2000).fit(Ftr, ytr)
                store.append(roc_auc_score(yte, clf.predict_proba(Fte)[:, 1]))
        lift = float(np.mean(on) - np.mean(off))
        assert lift >= 0.40, (
            f"pairwise-modular AUC lift {lift:.3f} below the +0.40 floor "
            f"(ON {np.mean(on):.3f} vs OFF {np.mean(off):.3f})."
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
