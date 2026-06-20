"""biz_value + integration tests for PAIRWISE INTEGER-LATTICE FE wired into MRMR.

The detector recovers a target that is a function of a hidden COMMON DIVISOR (gcd(a,b) -- shared factor), its dual
lcm(a,b), or a bit-level co-occurrence (a & b) of integer columns -- structure smooth/arithmetic/modular ops cannot
express (gcd is number-theoretic, non-smooth, non-monotone). XOR is excluded as redundant with the modular operator.
Wired into MRMR behind ``fe_integer_lattice_enable`` (default ON, wide-frame validated) with a column-count budget guard.

Contracts pinned (measured, never xfail):

* PROTOTYPE-direct: ``detect_integer_lattice`` fires on a gcd-shared-factor target with a large MI lift, stays silent on controls.
* INTEGRATION (the public-API win): MRMR with the flag ON recovers + SELECTS the gcd feature; transform() replays the
  frozen recipe identically at predict (leak-free, deterministic).
* ON is the default (wide-frame validated: zero FP at p=30, signal caught amid noise); opt-out (=False) is a true no-op.
* BUDGET GUARD: above max_int_cols the whole sweep is skipped (logged); selection still completes.
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

from mlframe.feature_selection.filters._integer_lattice_fe import (
    apply_integer_lattice,
    detect_integer_lattice,
    hybrid_integer_lattice_fe_with_recipes,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _build_gcd_target(seed: int, n: int = 4000):
    """y = gcd(a,b) >= 3, with a weakly-informative raw col so MRMR screening has an anchor and does not 0-fallback
    (the gcd column, not the raw cols, carries the signal)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 60, n)
    b = rng.integers(1, 60, n)
    y = (np.gcd(a, b) >= 3).astype(int)
    X = pd.DataFrame({
        "a": a, "b": b, "extra": (a % 2),
        "n0": rng.integers(0, 50, n), "n1": rng.integers(0, 50, n),
    })
    return X, y


class TestPrototypeDirect:
    def test_detects_gcd_shared_factor(self):
        X, y = _build_gcd_target(7)
        hits = detect_integer_lattice(X, y, seed=7)
        assert hits, "no responded hit on the gcd-shared-factor target."
        assert any(h["op"] == "gcd" for h in hits), "gcd op must respond on a shared-factor target."
        assert max(h["margin"] for h in hits) >= 0.10, "MI lift below the measured floor."

    def test_silent_on_smooth_control(self):
        rng = np.random.default_rng(7)
        n = 4000
        a, b = rng.integers(0, 100, n), rng.integers(0, 100, n)
        y = ((a + 0.7 * b) > 85).astype(int)
        X = pd.DataFrame({"a": a, "b": b, "n0": rng.integers(0, 50, n)})
        hits = detect_integer_lattice(X, y, seed=7)
        assert hits == [], f"smooth-threshold control fired: {hits}"


class TestRecipeReplay:
    def test_recipe_replay_bit_identical(self):
        X, y = _build_gcd_target(1)
        appended, recipes = hybrid_integer_lattice_fe_with_recipes(X, y, seed=1)
        assert recipes, "no integer-lattice recipes emitted."
        for r in recipes:
            direct = apply_integer_lattice(X, r.extra["op"], r.src_names)
            via_recipe = apply_recipe(r, X)
            np.testing.assert_array_equal(direct, via_recipe)

    def test_replay_is_leak_free(self):
        """Replay reads only X (cast ints, gcd/lcm/and) -- a held-out slice replays purely from its own X values."""
        X, y = _build_gcd_target(13)
        _, recipes = hybrid_integer_lattice_fe_with_recipes(X, y, seed=13)
        assert recipes
        r = recipes[0]
        on_train = apply_recipe(r, X)
        Xte = X.iloc[:500].reset_index(drop=True)
        on_test = apply_recipe(r, Xte)
        np.testing.assert_array_equal(on_train[:500], on_test)

    def test_recipe_pickle_round_trip(self):
        X, y = _build_gcd_target(1)
        _, recipes = hybrid_integer_lattice_fe_with_recipes(X, y, seed=1)
        assert recipes
        for r in recipes:
            r2 = pickle.loads(pickle.dumps(r))
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))


class TestMRMRIntegration:
    def test_opt_out_is_no_op(self):
        """The opt-out (fe_integer_lattice_enable=False) must stay a byte-identical no-op for legacy/replay."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_gcd_target(42, n=2000)
        m = MRMR(fe_integer_lattice_enable=False, max_runtime_mins=0.5)
        m.fit(X, pd.Series(y, name="y"))
        il = list(getattr(m, "integer_lattice_features_", []) or [])
        assert il == [], f"integer-lattice added columns with the flag disabled: {il}"
        out = m.transform(X.iloc[:300])
        assert not any(str(c).startswith("il_") for c in out.columns), (
            "no il column may appear in transform output when the flag is OFF."
        )

    def test_default_on_detects_gcd_signal(self):
        """The DEFAULT (no flag passed) is now ON: a fresh MRMR recovers the gcd feature with no explicit flag."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_gcd_target(7, n=4000)
        m = MRMR(max_runtime_mins=2)
        assert bool(getattr(m, "fe_integer_lattice_enable", False)) is True, (
            "fe_integer_lattice_enable must default to True (the validated ON default)."
        )
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:500])
        il_cols = [c for c in out.columns if str(c).startswith("il_")]
        assert il_cols, f"default-ON MRMR did not select an integer-lattice feature; selected={list(out.columns)}"

    def test_enabled_selects_gcd_feature_and_replays(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_gcd_target(7, n=4000)
        m = MRMR(fe_integer_lattice_enable=True, max_runtime_mins=2)
        m.fit(X, pd.Series(y, name="y"))
        il = list(getattr(m, "integer_lattice_features_", []) or [])
        assert len(il) >= 1, "integer-lattice enabled but produced no engineered columns."
        out = m.transform(X.iloc[:500])
        il_cols = [c for c in out.columns if str(c).startswith("il_")]
        assert il_cols, f"MRMR did not select any integer-lattice feature; selected={list(out.columns)}"
        out2 = m.transform(X.iloc[:500])
        for c in il_cols:
            np.testing.assert_array_equal(out[c].to_numpy(), out2[c].to_numpy())

    def test_both_operators_on_no_nested_engineered_recipe_replays_clean(self):
        """With pairwise-modular AND integer-lattice both default ON, the lattice operand pool must stay raw-only.
        Combining gcd onto an engineered pmod_ column would build a recipe whose engineered source is unresolved at
        replay time -- transform() would emit a NaN column and silently drop the feature. Regression for that bug."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_gcd_target(1, n=4000)
        m = MRMR(fe_integer_lattice_enable=True, fe_pairwise_modular_enable=True, max_runtime_mins=2)
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:500])
        il_cols = [c for c in out.columns if str(c).startswith("il_")]
        assert il_cols, "integer-lattice produced no replayable column with both operators on."
        nested = [c for c in il_cols if "pmod" in str(c) or "__il_" in str(c)]
        assert not nested, f"integer-lattice built a recipe on an engineered source (unresolvable at replay): {nested}"
        nan_cols = [c for c in out.columns if out[c].isna().any()]
        assert not nan_cols, f"replay emitted NaN columns (nested-engineered recipe): {nan_cols}"

    def test_budget_guard_skips_whole_sweep_and_logs(self, caplog):
        rng = np.random.default_rng(7)
        n = 2000
        cols = {f"c{i}": rng.integers(1, 60, n) for i in range(35)}
        a, b = cols["c0"], cols["c1"]
        y = (np.gcd(a, b) >= 3).astype(int)
        X = pd.DataFrame(cols)
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters._integer_lattice_fe"):
            appended, recipes = hybrid_integer_lattice_fe_with_recipes(X, y, seed=7)
        assert any("skipping the pairwise integer-lattice sweep" in r.message for r in caplog.records), (
            "expected a whole-sweep-skipped budget log line for 35 int columns."
        )
        assert appended == [] and recipes == [], "above max_int_cols nothing should be emitted."

    def test_clone_preserves_params(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_integer_lattice_enable=True,
            fe_integer_lattice_top_k=3,
            fe_integer_lattice_max_int_cols=25,
        )
        c = clone(m)
        assert bool(c.fe_integer_lattice_enable) is True
        assert int(c.fe_integer_lattice_top_k) == 3
        assert int(c.fe_integer_lattice_max_int_cols) == 25


class TestBizValue:
    def test_biz_val_integer_lattice_end_to_end_auc_lift(self):
        """MRMR(fe_integer_lattice_enable=True) recovers gcd(a,b)>=3, raw columns can't.

        gcd is non-monotone in either argument so raw a/b carry near-zero MI for this target; the gcd column is the only
        informative feature. Floor +0.20 AUC lift (well below the measured ON~0.9 / OFF~0.5 separation)."""
        on, off = [], []
        for seed in (1, 7, 42):
            from mlframe.feature_selection.filters.mrmr import MRMR
            X, y = _build_gcd_target(seed, n=3000)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y,
            )
            for flag, store in ((True, on), (False, off)):
                m = MRMR(fe_integer_lattice_enable=flag, fe_pairwise_modular_enable=False, max_runtime_mins=1)
                m.fit(Xtr, pd.Series(ytr, name="y"))
                Ftr = m.transform(Xtr)
                Fte = m.transform(Xte)
                clf = LogisticRegression(max_iter=2000).fit(Ftr, ytr)
                store.append(roc_auc_score(yte, clf.predict_proba(Fte)[:, 1]))
        lift = float(np.mean(on) - np.mean(off))
        assert lift >= 0.20, (
            f"integer-lattice AUC lift {lift:.3f} below the +0.20 floor "
            f"(ON {np.mean(on):.3f} vs OFF {np.mean(off):.3f})."
        )


class TestIntegerLatticeTargetTypeRobustness:
    """Target-type contract for the default-ON gcd/lcm/AND operator. A CONTINUOUS 1D y is now ELIGIBLE -- quantile-binned once into
    ``quantization_nbins`` before class-MI scoring -- so a genuine gcd regression target DETECTS while a smooth/noise continuous target stays
    SPECIFIC (0 emission). A 2D y stays SKIPPED (label-matrix binning out of scope). No hang/crash on any target type (permanent safety contract)."""

    def _xy(self, kind, n=600, seed=0):
        rng = np.random.default_rng(seed)
        xi = rng.integers(0, 20, size=(n, 4)).astype(float)
        xf = rng.normal(size=(n, 4))
        df = pd.DataFrame(np.column_stack([xi, xf]), columns=[f"x{i}" for i in range(8)])
        score = xf[:, 0] + xf[:, 1]
        if kind == "regression":
            y = pd.Series(score + 0.1 * rng.normal(size=n), name="y")
        elif kind == "quantile":
            y = pd.Series(score * 1000 + rng.normal(0, 500, size=n), name="y")
        elif kind == "count":
            y = pd.Series(rng.poisson(np.exp(0.5 * xf[:, 0] + 0.5 * xf[:, 1])), name="y")
        elif kind == "multilabel":
            y = pd.DataFrame(np.column_stack([(xf[:, 0] > 0).astype(int), (xf[:, 1] > 0).astype(int)]), columns=["l0", "l1"])
        elif kind == "multitarget":
            y = pd.DataFrame(np.column_stack([xf[:, 0], xf[:, 1]]), columns=["o0", "o1"])
        else:
            raise ValueError(kind)
        return df, y

    def _fit(self, df, y):
        import time
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0, dcd_enable=False,
                 cluster_aggregate_enable=False, build_friend_graph=False, cat_fe_config=None,
                 quantization_nbins=10, random_seed=0,
                 fe_pairwise_modular_enable=False, fe_integer_lattice_enable=True,
                 fe_row_argmax_enable=False, fe_conditional_gate_enable=False)
        assert bool(m.fe_integer_lattice_enable) is True
        t0 = time.time()
        m.fit(df, y)
        assert time.time() - t0 < 30.0, "integer-lattice fit exceeded 30s wall (hang-class bug)"
        return m

    @pytest.mark.parametrize("kind", ["regression", "quantile", "count"])
    def test_integer_lattice_specific_on_smooth_continuous_target_no_crash_or_hang(self, kind):
        df, y = self._xy(kind)
        m = self._fit(df, y)
        assert list(getattr(m, "integer_lattice_features_", []) or []) == [], (
            f"integer-lattice FE must emit nothing on a SMOOTH continuous {kind} target (specificity on binned y)"
        )

    @pytest.mark.parametrize("kind", ["multilabel", "multitarget"])
    def test_integer_lattice_skipped_on_2d_target_no_crash_or_hang(self, kind):
        df, y = self._xy(kind)
        m = self._fit(df, y)
        assert list(getattr(m, "integer_lattice_features_", []) or []) == [], (
            f"integer-lattice FE must clean-skip on 2D {kind} y (class-MI floor undefined on a label matrix)"
        )

    def test_integer_lattice_detects_on_gcd_regression_target(self):
        """Continuous-1D y with TRUE gcd structure (y = gcd(a, b) + noise) DETECTS + emits the gcd feature on the binned y."""
        n, seed = 600, 0
        rng = np.random.default_rng(seed)
        a = rng.integers(1, 40, n) * 2
        b = rng.integers(1, 40, n) * 2
        df = pd.DataFrame({"a": a, "b": b, "f": rng.normal(0, 1, n)})
        y = pd.Series(np.gcd(a, b).astype(float) + rng.normal(0, 0.1, n), name="y")
        m = self._fit(df, y)
        feats = list(getattr(m, "integer_lattice_features_", []) or [])
        assert any("gcd" in f for f in feats), f"expected a gcd feature on the regression gcd target; got {feats}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
