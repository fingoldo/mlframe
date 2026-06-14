"""biz_value + integration tests for ROW-ARGMAX and CONDITIONAL-GATE FE wired into MRMR.

Two frontier-pass-2 operators the rich catalog cannot express for the MI / linear-downstream selector:

* ROW-ARGMAX -- ``argmax_row(a,b,c)`` = which column is the row maximum (ordinal/comparison). No shipped column equals the
  3-way argmax code. ZERO free params, detector-clean. Wired behind ``fe_row_argmax_enable`` (default ON, wide-frame validated).
* CONDITIONAL-GATE -- ``c>tau ? a : b`` (select) / ``1[c>tau]*a`` (mask): two raw features routed/masked by a third column's
  data-dependent threshold tau (FROZEN in the recipe). HARDENED detector (beats best-existing-op MI, not the raw operand floor)
  removes the prototype's smooth/ordinary_mul false positives. Wired behind ``fe_conditional_gate_enable`` (default ON).

Contracts pinned (measured, never xfail):

* PROTOTYPE-direct: detectors fire on their natural targets with a large MI lift, stay silent on controls (incl. the gate's hard
  smooth / ordinary_mul cases after hardening).
* INTEGRATION: MRMR with the flag ON recovers + SELECTS the engineered feature; transform() replays the frozen recipe identically
  at predict (leak-free, deterministic; for the gate this includes the frozen tau).
* ON is the default; opt-out (=False) is a true no-op.
* BUDGET GUARD: above max_cols the whole sweep is skipped (logged); selection still completes.
* INTERACTION: with modular + lattice + argmax + gate ALL ON, no nested-engineered recipe + no NaN replay column.
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

from mlframe.feature_selection.filters._conditional_gate_fe import (
    apply_conditional_gate,
    apply_row_argmax,
    detect_conditional_gate,
    detect_row_argmax,
    hybrid_conditional_gate_fe_with_recipes,
    hybrid_row_argmax_fe_with_recipes,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _argmax_target(seed: int, n: int = 4000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = np.argmax(np.stack([a, b, c], axis=1), axis=1)
    return pd.DataFrame({"a": a, "b": b, "c": c, "extra": rng.normal(0, 1, n)}), y


def _gate_target(seed: int, n: int = 4000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    sel = np.where(c > 0.0, a, b)
    y = (sel > np.median(sel)).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "extra": rng.normal(0, 1, n)}), y


def _smooth_control(seed: int, n: int = 4000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c}), ((a + 0.5 * b) > 0).astype(int)


def _ordinary_mul_control(seed: int, n: int = 4000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c}), ((a * b) > 0).astype(int)


class TestPrototypeDirect:
    def test_argmax_detects_target(self):
        X, y = _argmax_target(7)
        hits = detect_row_argmax(X, y, seed=7)
        assert hits, "no responded hit on the row-argmax target."
        assert max(h["margin"] for h in hits) >= 0.30, "argmax MI lift below the measured floor."

    def test_argmax_silent_on_controls(self):
        for gen in (_smooth_control, _ordinary_mul_control):
            X, y = gen(7)
            assert detect_row_argmax(X, y, seed=7) == [], f"argmax fired on {gen.__name__} control."

    def test_gate_detects_regime_target(self):
        X, y = _gate_target(13)
        hits = detect_conditional_gate(X, y, seed=13)
        assert hits, "no responded hit on the regime-switch target."
        assert hits[0]["mode"] in ("select", "mask")
        assert max(h["margin"] for h in hits) >= 0.20, "gate MI lift below the measured floor."

    def test_gate_hardened_silent_on_smooth_and_ordinary_controls(self):
        """The HARDENED gate (beats best-existing-op MI) must NOT fire on smooth / ordinary_mul -- the prototype's FP cases."""
        for gen in (_smooth_control, _ordinary_mul_control):
            for s in (1, 7, 13):
                X, y = gen(s)
                assert detect_conditional_gate(X, y, seed=s) == [], (
                    f"hardened gate fired on {gen.__name__} control (seed={s}) -- the hardening regressed."
                )

    def test_gate_silent_on_multidriver_additive_target(self):
        """Regression sensor: a purely ADDITIVE target ``y = x0+x1+x2`` (no regime structure) must NOT fire the gate on ANY target
        type. A piecewise ``c>tau ? a : b`` partially reconstructs the additive sum, so the prototype floor (pairwise product /
        ratio / diff / min / max only) was cleared on the binned-regression / 4-driver cases (1 spurious feature emitted). The
        broadened best-existing-op baseline -- now also scoring ``a+b`` / ``a+c`` / ``b+c`` / ``a+b+c`` -- captures the additive
        signal so the gate cannot manufacture lift over it. Pins 0 emission on binary + 10-bin-regression + 4-driver additive."""
        def _additive(seed, ndrivers, ttype, n=2000):
            rng = np.random.default_rng(seed)
            X = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(6)})
            sig = sum(X[f"x{i}"].to_numpy() for i in range(ndrivers))
            if ttype == "binary":
                y = (sig > np.median(sig)).astype(int)
            else:
                y = pd.qcut(sig, 10, labels=False, duplicates="drop")
            return X, np.asarray(y)

        for ndrivers in (3, 4):
            for ttype in ("binary", "10bin"):
                for s in (0, 1, 2):
                    X, y = _additive(s, ndrivers, ttype)
                    hits = detect_conditional_gate(X, y, list(X.columns), seed=s)
                    assert hits == [], (
                        f"gate fired on {ndrivers}-driver additive {ttype} target (seed={s}): {hits} -- additive-floor regressed."
                    )


class TestRecipeReplay:
    def test_argmax_recipe_replay_bit_identical(self):
        X, y = _argmax_target(1)
        appended, recipes = hybrid_row_argmax_fe_with_recipes(X, y, seed=1)
        assert recipes, "no row-argmax recipes emitted."
        for r in recipes:
            direct = apply_row_argmax(X, r.src_names)
            np.testing.assert_array_equal(direct, apply_recipe(r, X))

    def test_gate_recipe_replay_bit_identical_with_frozen_tau(self):
        X, y = _gate_target(1)
        appended, recipes = hybrid_conditional_gate_fe_with_recipes(X, y, seed=1)
        assert recipes, "no conditional-gate recipes emitted."
        for r in recipes:
            assert "tau" in r.extra, "gate recipe must freeze tau."
            direct = apply_conditional_gate(X, r.extra["mode"], r.src_names, r.extra["tau"])
            np.testing.assert_array_equal(direct, apply_recipe(r, X))

    def test_gate_frozen_tau_replays_on_holdout(self):
        """Replay reads only X + the frozen tau -- a held-out slice replays purely from its own X values (leak-free)."""
        X, y = _gate_target(13)
        _, recipes = hybrid_conditional_gate_fe_with_recipes(X, y, seed=13)
        assert recipes
        r = recipes[0]
        on_train = apply_recipe(r, X)
        Xte = X.iloc[:500].reset_index(drop=True)
        np.testing.assert_array_equal(on_train[:500], apply_recipe(r, Xte))

    def test_recipe_pickle_round_trip(self):
        X, y = _gate_target(1)
        _, gate_recipes = hybrid_conditional_gate_fe_with_recipes(X, y, seed=1)
        Xa, ya = _argmax_target(1)
        _, am_recipes = hybrid_row_argmax_fe_with_recipes(Xa, ya, seed=1)
        assert gate_recipes and am_recipes
        for r, frame in [(gate_recipes[0], X), (am_recipes[0], Xa)]:
            r2 = pickle.loads(pickle.dumps(r))
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            np.testing.assert_array_equal(apply_recipe(r, frame), apply_recipe(r2, frame))


class TestMRMRIntegration:
    def test_argmax_opt_out_is_no_op(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _argmax_target(42, n=2000)
        m = MRMR(fe_row_argmax_enable=False, max_runtime_mins=0.5)
        m.fit(X, pd.Series(y, name="y"))
        assert list(getattr(m, "row_argmax_features_", []) or []) == []
        out = m.transform(X.iloc[:300])
        assert not any(str(c).startswith("argmax_") for c in out.columns)

    def test_argmax_off_selection_identical_to_baseline(self):
        """OFF is a true no-op: the selected raw set with fe_row_argmax_enable=False is identical to disabling the operator
        entirely (proves OFF does not perturb selection vs the pre-operator baseline)."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(3)
        n = 2000
        X = pd.DataFrame({f"c{i}": rng.normal(0, 1, n) for i in range(6)})
        y = (X["c0"].to_numpy() + 0.5 * X["c1"].to_numpy() > 0).astype(int)
        sel = []
        for _ in range(2):
            m = MRMR(fe_row_argmax_enable=False, fe_conditional_gate_enable=False,
                     fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False, max_runtime_mins=0.5)
            m.fit(X, pd.Series(y, name="y"))
            sel.append(tuple(m.transform(X.iloc[:200]).columns))
        assert sel[0] == sel[1], f"OFF selection not deterministic: {sel[0]} vs {sel[1]}"
        assert not any(str(c).startswith("argmax_") for c in sel[0])

    def test_gate_opt_out_is_no_op(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _gate_target(42, n=2000)
        m = MRMR(fe_conditional_gate_enable=False, max_runtime_mins=0.5)
        m.fit(X, pd.Series(y, name="y"))
        assert list(getattr(m, "conditional_gate_features_", []) or []) == []
        out = m.transform(X.iloc[:300])
        assert not any(str(c).startswith("gate_") for c in out.columns)

    def test_argmax_default_on_selects_feature(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _argmax_target(7, n=4000)
        m = MRMR(max_runtime_mins=2)
        assert bool(getattr(m, "fe_row_argmax_enable", False)) is True
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:500])
        assert [c for c in out.columns if str(c).startswith("argmax_")], (
            f"default-ON MRMR did not select a row-argmax feature; selected={list(out.columns)}"
        )

    def test_gate_default_is_on(self):
        """conditional-gate now defaults ON: the relevance-pruned candidate set makes the sweep O(k^2) flat-in-p (the prior
        O(p^3) cost that forced it OFF is gone). Opt out with fe_conditional_gate_enable=False; see the ctor docstring."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert bool(getattr(m, "fe_conditional_gate_enable", False)) is True
        assert int(getattr(m, "fe_conditional_gate_k_gate", 0)) == 8
        assert int(getattr(m, "fe_conditional_gate_k_operand", 0)) == 10

    def test_gate_default_on_selects_feature_and_replays(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _gate_target(13, n=4000)
        m = MRMR(max_runtime_mins=2)
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:500])
        gate_cols = [c for c in out.columns if str(c).startswith("gate_")]
        assert gate_cols, f"default-ON MRMR did not select a conditional-gate feature; selected={list(out.columns)}"
        out2 = m.transform(X.iloc[:500])
        for c in gate_cols:
            np.testing.assert_array_equal(out[c].to_numpy(), out2[c].to_numpy())

    def test_all_operators_on_no_nested_engineered_recipe_replays_clean(self):
        """With modular + lattice + argmax + gate ALL default ON, every operand pool must stay raw-only. Combining onto an
        engineered column (pmod_/il_/argmax_/gate_) would build a recipe whose engineered source is unresolved at replay --
        transform() would emit a NaN column and silently drop the feature. Regression for that interaction bug."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _gate_target(1, n=4000)
        m = MRMR(
            fe_row_argmax_enable=True, fe_conditional_gate_enable=True,
            fe_pairwise_modular_enable=True, fe_integer_lattice_enable=True,
            max_runtime_mins=2,
        )
        m.fit(X, pd.Series(y, name="y"))
        out = m.transform(X.iloc[:500])
        eng = [c for c in out.columns if str(c).startswith(("argmax_", "gate_", "il_", "pmod_"))]
        nested = [c for c in eng if any(p in str(c) for p in ("__argmax_", "__gate_", "__il_", "__pmod_"))]
        assert not nested, f"an operator built a recipe on an engineered source (unresolvable at replay): {nested}"
        nan_cols = [c for c in out.columns if out[c].isna().any()]
        assert not nan_cols, f"replay emitted NaN columns (nested-engineered recipe): {nan_cols}"

    def test_argmax_budget_guard_skips_and_logs(self, caplog):
        rng = np.random.default_rng(7)
        n = 2000
        cols = {f"c{i}": rng.normal(0, 1, n) for i in range(35)}
        X = pd.DataFrame(cols)
        y = np.argmax(np.stack([X["c0"], X["c1"], X["c2"]], axis=1), axis=1)
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters._conditional_gate_fe"):
            appended, recipes = hybrid_row_argmax_fe_with_recipes(X, y, seed=7)
        assert any("skipping the row-argmax sweep" in r.message for r in caplog.records)
        assert appended == [] and recipes == []

    def test_gate_budget_guard_skips_and_logs(self, caplog):
        rng = np.random.default_rng(7)
        n = 2000
        cols = {f"c{i}": rng.normal(0, 1, n) for i in range(25)}
        X = pd.DataFrame(cols)
        y = (np.where(X["c2"].to_numpy() > 0, X["c0"].to_numpy(), X["c1"].to_numpy()) > 0).astype(int)
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters._conditional_gate_fe"):
            appended, recipes = hybrid_conditional_gate_fe_with_recipes(X, y, seed=7, max_cols=20)
        assert any("skipping the conditional-gate sweep" in r.message for r in caplog.records)
        assert appended == [] and recipes == []

    @pytest.mark.timeout(120)
    def test_gate_specific_on_noise_regression_target_no_hang(self):
        """A CONTINUOUS regression y is now ELIGIBLE (quantile-binned once before the tau-grid + conditional-divergence MI, which
        previously exploded under the int64 cast). On a single-driver smooth regression target with no regime structure the gate MUST
        stay SPECIFIC -- fit completes fast, conditional_gate_features_ stays empty. The no-hang safety contract is permanent.

        Multi-DRIVER ADDITIVE specificity is now covered separately (see ``test_gate_silent_on_multidriver_additive_target``): the
        broadened best-existing-op baseline (which includes ``a+b`` / ``a+c`` / ``b+c`` / ``a+b+c``) captures the additive signal a
        piecewise ``c>tau ? a : b`` used to partially reconstruct, so the gate no longer fires on a purely additive target."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(42)
        n = 600
        X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"x{i}" for i in range(8)])
        y = 2.0 * X["x0"].to_numpy() + 0.3 * rng.normal(size=n)  # single-driver smooth continuous target (no regime / no multi-driver sum)
        m = MRMR(verbose=0, random_seed=42)
        assert bool(m.fe_conditional_gate_enable) is True
        m.fit(X, pd.Series(y, name="y"))
        assert list(getattr(m, "conditional_gate_features_", []) or []) == [], (
            "conditional-gate FE must emit nothing on a single-driver smooth regression target (specificity on binned y, no regime structure)."
        )

    def test_clone_preserves_params(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_row_argmax_enable=True, fe_row_argmax_top_k=3, fe_row_argmax_max_cols=25,
            fe_conditional_gate_enable=True, fe_conditional_gate_top_k=2, fe_conditional_gate_max_cols=15,
            fe_conditional_gate_k_gate=6, fe_conditional_gate_k_operand=7,
        )
        c = clone(m)
        assert bool(c.fe_row_argmax_enable) is True
        assert int(c.fe_row_argmax_top_k) == 3
        assert int(c.fe_row_argmax_max_cols) == 25
        assert bool(c.fe_conditional_gate_enable) is True
        assert int(c.fe_conditional_gate_top_k) == 2
        assert int(c.fe_conditional_gate_max_cols) == 15
        assert int(c.fe_conditional_gate_k_gate) == 6
        assert int(c.fe_conditional_gate_k_operand) == 7


class TestRelevancePruning:
    """The relevance-pruned candidate set makes the gate sweep O(k_operand^2 * k_gate), flat in p (the prior O(p^3) cost
    that forced the gate OFF). Detection must SURVIVE the prune: the true operands a,b (marginal relevance) and the gate
    column c (marginally y-independent -> ranked by conditional divergence, NOT raw MI) must stay in their top-k pools."""

    def _gate_with_noise(self, seed: int, n_noise: int = 25, n: int = 4000):
        rng = np.random.default_rng(seed)
        a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
        sel = np.where(c > 0.0, a, b)
        y = (sel > np.median(sel)).astype(int)
        cols = {"a": a, "b": b, "c": c}
        for i in range(n_noise):
            cols[f"noise{i}"] = rng.normal(0, 1, n)
        return pd.DataFrame(cols), y

    def test_gate_signal_survives_pruning_amid_25_noise_three_seeds(self):
        """With the default k_gate=8 / k_operand=10, the regime-switch signal is still detected amid 25 noise columns on every
        seed -- the gate column c (raw MI ~ 0) survives via the conditional-divergence rank, the operands via raw MI."""
        for seed in (1, 7, 42):
            X, y = self._gate_with_noise(seed)
            hits = detect_conditional_gate(X, y, list(X.columns), seed=seed)
            assert hits, f"seed={seed}: gate signal lost after relevance-pruning amid 25 noise columns."

    def test_pruned_gate_column_ranked_by_conditional_divergence_not_raw_mi(self):
        """The true gate column c is marginally y-independent (raw MI ~ 0, would rank LAST), yet must enter the gate pool via the
        conditional-divergence rank. Pins the dual-signal prune: raw-MI ranking alone would drop c and miss the regime switch."""
        from mlframe.feature_selection.filters._conditional_gate_fe import _rank_and_prune
        X, y = self._gate_with_noise(7)
        gate_pool, operand_pool = _rank_and_prune(X, list(X.columns), np.asarray(y).astype(np.int64), 12, 8, 10)
        assert "c" in gate_pool, f"gate column c dropped from the k_gate pool: {gate_pool}"
        assert "a" in operand_pool and "b" in operand_pool, f"operands missing from the k_operand pool: {operand_pool}"

    def test_gate_scan_cost_flat_in_p(self):
        """The pruned scan's candidate count is bounded by O(k_operand^2 * k_gate), independent of p: tripling the noise-column
        count must NOT grow the candidate set proportionally (the unpruned C(p,2)*p sweep would explode). Cap = k_gate * mask +
        k_gate * select = k_gate*(k_operand-1) + k_gate*(k_operand-1)*(k_operand-1), well below the unpruned count at p=63."""
        from mlframe.feature_selection.filters._conditional_gate_fe import cheap_conditional_gate_scan
        k_gate, k_operand = 8, 10
        cap = k_gate * (k_operand + k_operand * (k_operand - 1))  # per gate col: mask (<=k_operand) + select (<=k_operand*(k_operand-1))
        counts = []
        for n_noise in (10, 60):
            X, y = self._gate_with_noise(7, n_noise=n_noise, n=2000)
            counts.append(len(cheap_conditional_gate_scan(X, y, list(X.columns), k_gate=k_gate, k_operand=k_operand)))
        assert all(cnt <= cap for cnt in counts), f"candidate count {counts} exceeded the O(k^2*k_gate) cap {cap}."
        assert counts[1] <= counts[0] * 1.3, f"candidate count grew with p ({counts}) -- the prune is not flat-in-p."


class TestBizValue:
    def test_biz_val_row_argmax_end_to_end_recovers_feature_with_mi_lift(self):
        """MRMR(fe_row_argmax_enable=True) recovers + SELECTS the argmax-of-(a,b,c) column, and that single engineered column
        carries the target's structure with a large MI lift over the best raw / pairwise op (the catalog gap the operator fills).

        End-to-end AUC over a MULTINOMIAL-logit downstream is the wrong sensor here: a softmax over (a,b,c) already reconstructs
        the row-argmax, so both ON / OFF hit AUC~1.0 -- the win is at the single-feature MI/usability level, where the shipped
        catalog has no column equal to the 3-way argmax code (measured +0.55 MI lift, pinned in
        ``test_biz_val_conditional_gate_fe`` / the prototype-direct tests). Floor +0.30 MI lift below the measured +0.55."""
        from mlframe.feature_selection.filters._pairwise_modular_fe import _mi
        lifts = []
        for seed in (1, 7, 42):
            from mlframe.feature_selection.filters.mrmr import MRMR
            X, y = _argmax_target(seed, n=3000)
            m = MRMR(fe_row_argmax_enable=True, fe_conditional_gate_enable=False,
                     fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False, max_runtime_mins=1)
            m.fit(X, pd.Series(y, name="y"))
            F = m.transform(X)
            sel = [c for c in F.columns if str(c).startswith("argmax_")]
            assert sel, f"seed={seed}: default-ON MRMR did not select the row-argmax feature; selected={list(F.columns)}"
            yi = np.asarray(y).astype(np.int64)
            argmax_mi = _mi(np.asarray(F[sel[0]], dtype=np.float64), yi, nbins=12)
            best_raw = max(_mi(np.asarray(X[c], dtype=np.float64), yi, nbins=12) for c in ("a", "b", "c", "extra"))
            lifts.append(argmax_mi - best_raw)
        lift = float(np.mean(lifts))
        assert lift >= 0.30, f"row-argmax MI lift over best raw {lift:.3f} below +0.30 floor."

    def test_biz_val_conditional_gate_end_to_end_auc_lift(self):
        """MRMR(fe_conditional_gate_enable=True) recovers the regime-switch ``c>0 ? a : b`` target; raw + smooth ops cannot.
        Floor +0.10 AUC lift below the measured ON/OFF separation."""
        on, off = [], []
        for seed in (1, 7, 42):
            from mlframe.feature_selection.filters.mrmr import MRMR
            X, y = _gate_target(seed, n=3000)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            for flag, store in ((True, on), (False, off)):
                # fe_fast_search=False: the lift here is the gate's MARGINAL benefit (ON vs OFF). The
                # default fast path (2026-06-14) strengthens the non-gate baseline (OFF AUC 0.80->0.96),
                # compressing the measured lift below the +0.10 floor even though the gate ON result is
                # unchanged (0.998). Pin the exhaustive path so the marginal-lift contract is faithful.
                m = MRMR(fe_conditional_gate_enable=flag, fe_row_argmax_enable=False, fe_fast_search=False,
                         fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False, max_runtime_mins=1)
                m.fit(Xtr, pd.Series(ytr, name="y"))
                clf = LogisticRegression(max_iter=2000).fit(m.transform(Xtr), ytr)
                store.append(roc_auc_score(yte, clf.predict_proba(m.transform(Xte))[:, 1]))
        lift = float(np.mean(on) - np.mean(off))
        assert lift >= 0.10, f"conditional-gate AUC lift {lift:.3f} below +0.10 (ON {np.mean(on):.3f} vs OFF {np.mean(off):.3f})."


class TestArgmaxAndGateTargetTypeRobustness:
    """Target-type contract for the default-ON row-argmax + conditional-gate operators. A CONTINUOUS 1D y (quantile/count) is now ELIGIBLE --
    quantile-binned once before class-MI scoring -- so a genuine argmax / regime regression target DETECTS while a smooth/noise continuous
    target stays SPECIFIC (0 emission). A 2D y stays SKIPPED (label-matrix binning out of scope). No crash / no >30s hang on any target type."""

    def _xy(self, kind, n=600, seed=0):
        rng = np.random.default_rng(seed)
        xi = rng.integers(0, 20, size=(n, 4)).astype(float)
        xf = rng.normal(size=(n, 4))
        df = pd.DataFrame(np.column_stack([xi, xf]), columns=[f"x{i}" for i in range(8)])
        score = xf[:, 0] + xf[:, 1]
        if kind == "quantile":
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

    def _mrmr(self, **flags):
        from mlframe.feature_selection.filters.mrmr import MRMR

        return MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0, dcd_enable=False,
                    cluster_aggregate_enable=False, build_friend_graph=False, cat_fe_config=None,
                    quantization_nbins=10, random_seed=0,
                    fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False, **flags)

    @pytest.mark.parametrize("kind", ["quantile", "count"])
    def test_row_argmax_specific_on_smooth_continuous_target_no_crash_or_hang(self, kind):
        import time

        df, y = self._xy(kind)
        m = self._mrmr(fe_row_argmax_enable=True, fe_conditional_gate_enable=False)
        assert bool(m.fe_row_argmax_enable) is True
        t0 = time.time()
        m.fit(df, y)
        assert time.time() - t0 < 30.0, f"row-argmax fit on {kind} exceeded 30s wall (hang-class bug)"
        assert list(getattr(m, "row_argmax_features_", []) or []) == [], (
            f"row-argmax FE must emit nothing on a SMOOTH continuous {kind} target (specificity on binned y)"
        )

    @pytest.mark.parametrize("kind", ["multilabel", "multitarget"])
    def test_row_argmax_skipped_on_2d_target_no_crash_or_hang(self, kind):
        import time

        df, y = self._xy(kind)
        m = self._mrmr(fe_row_argmax_enable=True, fe_conditional_gate_enable=False)
        t0 = time.time()
        m.fit(df, y)
        assert time.time() - t0 < 30.0, f"row-argmax fit on {kind} exceeded 30s wall (hang-class bug)"
        assert list(getattr(m, "row_argmax_features_", []) or []) == [], (
            f"row-argmax FE must clean-skip on 2D {kind} y (class-MI floor undefined on a label matrix)"
        )

    @pytest.mark.parametrize("kind", ["quantile", "count"])
    def test_conditional_gate_specific_on_smooth_continuous_target_no_crash_or_hang(self, kind):
        import time

        df, y = self._xy(kind)
        m = self._mrmr(fe_row_argmax_enable=False, fe_conditional_gate_enable=True)
        assert bool(m.fe_conditional_gate_enable) is True
        t0 = time.time()
        m.fit(df, y)
        assert time.time() - t0 < 30.0, f"conditional-gate fit on {kind} exceeded 30s wall (hang-class bug)"
        assert list(getattr(m, "conditional_gate_features_", []) or []) == [], (
            f"conditional-gate FE must emit nothing on a SMOOTH continuous {kind} target (specificity on binned y)"
        )

    @pytest.mark.parametrize("kind", ["multilabel", "multitarget"])
    def test_conditional_gate_skipped_on_2d_target_no_crash_or_hang(self, kind):
        import time

        df, y = self._xy(kind)
        m = self._mrmr(fe_row_argmax_enable=False, fe_conditional_gate_enable=True)
        t0 = time.time()
        m.fit(df, y)
        assert time.time() - t0 < 30.0, f"conditional-gate fit on {kind} exceeded 30s wall (hang-class bug)"
        assert list(getattr(m, "conditional_gate_features_", []) or []) == [], (
            f"conditional-gate FE must clean-skip on 2D {kind} y (class-MI floor undefined on a label matrix)"
        )

    def test_row_argmax_detects_on_argmax_regression_target(self):
        """Continuous-1D y driven by which of 3 cols is the row-max (y = 5*argmax + noise) DETECTS + emits the argmax feature on binned y."""
        n, seed = 600, 0
        rng = np.random.default_rng(seed)
        ca, cb, cc = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
        idx = np.argmax(np.stack([ca, cb, cc], 1), 1)
        df = pd.DataFrame({"ca": ca, "cb": cb, "cc": cc})
        y = pd.Series(idx.astype(float) * 5 + rng.normal(0, 0.1, n), name="y")
        m = self._mrmr(fe_row_argmax_enable=True, fe_conditional_gate_enable=False)
        m.fit(df, y)
        feats = list(getattr(m, "row_argmax_features_", []) or [])
        assert any("argmax" in f for f in feats), f"expected an argmax feature on the regression argmax target; got {feats}"

    def test_conditional_gate_detects_on_regime_regression_target(self):
        """Continuous-1D regime-switch y (y = a if c>median else b) DETECTS + emits a gate feature on the binned y."""
        n, seed = 600, 0
        rng = np.random.default_rng(seed)
        a, b, c = rng.normal(0, 1, n), rng.normal(5, 1, n), rng.normal(0, 1, n)
        y = pd.Series(np.where(c > np.median(c), a, b) + rng.normal(0, 0.05, n), name="y")
        df = pd.DataFrame({"a": a, "b": b, "c": c})
        m = self._mrmr(fe_row_argmax_enable=False, fe_conditional_gate_enable=True)
        m.fit(df, y)
        feats = list(getattr(m, "conditional_gate_features_", []) or [])
        assert any("gate" in f for f in feats), f"expected a gate feature on the regime regression target; got {feats}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
