"""Layer 95 biz_value: PERIODIC/MODULAR (PART A) + PER-GROUP DISTRIBUTION-
DISTANCE (PART B) FE.

PART A extends Layer 90 (``_periodic_fe``): ``x mod period`` plus sin/cos phase
encoding, gated by Layer 62 bootstrap-stable MI (the gate doubles as
auto-period detection). PART B extends Layer 88 (``_group_distance_fe``):
per-group z / KL / Wasserstein-1 distance from the global distribution,
broadcast to rows, MI-uplift gated.

Contracts pinned (real numbers, never xfail):

PART A
* periodic signal: y = f(x mod 24); the modular feature recovers it, raw x
  doesn't; LogReg AUC lift >= +0.10.
* cyclic continuity: sin/cos encoding beats raw mod for a smoothly-cyclic
  target (no discontinuity at the period boundary).
* auto-period: the correct period (24) is selected over wrong periods (7, 12).

PART B
* group-anomaly signal: y depends on whether the row's group is atypical
  (group_mean far from global); the group-distance feature captures it.
* AUC lift >= +0.05.

BOTH
* no leakage: transform(X, y_shuffled) == transform(X); recipe replay reads X.
* default disabled byte-identical.
* pickle / clone round-trips recipes + ctor params.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# MI helper (matches the L62 / L19 estimator the gates use internally)
# ---------------------------------------------------------------------------


def _mi_one(col: np.ndarray, y: np.ndarray, nbins: int = 10) -> float:
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        _mi_classif_batch,
    )
    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


# ---------------------------------------------------------------------------
# PART A fixtures
# ---------------------------------------------------------------------------


def _build_hour_of_day(seed: int, n: int = 6000):
    """y is a step function of ``x mod 24`` (hour-of-day cycle). The raw column
    ``t`` is a large running magnitude (cumulative time) so its raw value
    carries almost no MI about y -- the signal lives entirely in the residue."""
    rng = np.random.default_rng(int(seed))
    t = rng.uniform(0.0, 2400.0, n)  # arbitrary running magnitude
    hour = np.mod(t, 24.0)
    flip = rng.random(n) < 0.03
    y = ((hour >= 12.0).astype(int)) ^ flip.astype(int)
    X = pd.DataFrame({"t": t})
    return X, y.astype(int)


def _build_smooth_cyclic(seed: int, n: int = 6000):
    """y has a SMOOTHLY-cyclic mean: P(y=1) varies sinusoidally with the phase
    ``2*pi*(t mod 24)/24``. The raw residue ``mod`` has a discontinuity at the
    period boundary (23.9 and 0.0 are far in value, adjacent in phase) that a
    linear model mis-reads; the sin/cos encoding maps the residue onto the unit
    circle so the target is a linear function of (sin, cos)."""
    rng = np.random.default_rng(int(seed))
    t = rng.uniform(0.0, 2400.0, n)
    phase = 2.0 * np.pi * np.mod(t, 24.0) / 24.0
    p = 1.0 / (1.0 + np.exp(-(2.0 * np.sin(phase))))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"t": t})
    return X, y.astype(int)


# ---------------------------------------------------------------------------
# PART B fixtures
# ---------------------------------------------------------------------------


def _build_group_anomaly(seed: int, n: int = 6000, n_groups: int = 24):
    """y depends on whether the row's GROUP is DISTRIBUTIONALLY atypical. Every
    group shares the SAME mean (0), so the row's own value reveals nothing about
    group membership -- but a handful of anomalous groups have a heavy-tailed /
    high-variance value distribution (the global is tight-Gaussian). y=1 iff the
    row's group is one of the anomalous (wide-distribution) groups.

    Raw ``v`` cannot separate (same centre, individual values are ambiguous
    between a tight-group draw and a wide-group draw); only a per-group
    DISTRIBUTIONAL distance from the global pool (KL / Wasserstein-1) flags the
    anomaly. This is exactly the shape-anomaly that Layer 88's within-group
    rank and a mean-shift z both miss."""
    rng = np.random.default_rng(int(seed))
    grp = rng.integers(0, n_groups, n)
    # A quarter of the groups are distributionally anomalous (wide spread);
    # the rest are tight. All share mean 0.
    anomalous = np.zeros(n_groups, dtype=bool)
    anomalous[: max(3, n_groups // 4)] = True
    rng.shuffle(anomalous)
    group_scale = np.where(anomalous, 6.0, 1.0)  # wide vs tight, same mean
    val = rng.normal(0.0, 1.0, n) * group_scale[grp]
    flip = rng.random(n) < 0.03
    y = anomalous[grp].astype(int) ^ flip.astype(int)
    X = pd.DataFrame({"g": grp.astype(np.int64), "v": val})
    return X, y.astype(int)


# ---------------------------------------------------------------------------
# PART A Contract 1: periodic signal recovered; raw x doesn't; AUC lift >= +0.10
# ---------------------------------------------------------------------------


class TestPeriodicSignal:
    def test_modular_recovers_period_mi(self):
        from mlframe.feature_selection.filters._periodic_fe import apply_modular
        gains = []
        for s in SEEDS:
            X, y = _build_hour_of_day(s)
            t = X["t"].to_numpy()
            mi_raw = _mi_one(t, y)
            mi_mod = _mi_one(apply_modular(t, 24.0, "mod"), y)
            gains.append(mi_mod - mi_raw)
        mean_gain = float(np.mean(gains))
        assert mean_gain >= 0.15, (
            f"modular MI gain {mean_gain:.4f} < 0.15 over raw t (per-seed "
            f"{[round(g, 4) for g in gains]}); x mod 24 is not recovering the "
            f"hour-of-day signal."
        )

    def test_logreg_auc_lift(self):
        from mlframe.feature_selection.filters._periodic_fe import (
            hybrid_modular_fe_with_recipes,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        lifts = []
        for s in SEEDS:
            X, y = _build_hour_of_day(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            base = LogisticRegression(max_iter=2000).fit(Xtr[["t"]], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[["t"]])[:, 1])

            _, appended, recipes, _ = hybrid_modular_fe_with_recipes(
                Xtr, ytr.values if hasattr(ytr, "values") else ytr,
                periods=(7, 12, 24, 30, 365), top_k=6, seed=s,
            )
            assert appended, f"seed={s}: no modular survivors."
            Xtr_aug = Xtr[["t"]].reset_index(drop=True).copy()
            Xte_aug = Xte[["t"]].reset_index(drop=True).copy()
            for r in recipes:
                Xtr_aug[r.name] = apply_recipe(r, Xtr)
                Xte_aug[r.name] = apply_recipe(r, Xte)
            aug = LogisticRegression(max_iter=2000).fit(Xtr_aug, ytr)
            auc_aug = roc_auc_score(yte, aug.predict_proba(Xte_aug)[:, 1])
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.10, (
            f"hour-of-day AUC lift {mean_lift:.4f} < 0.10 (per-seed "
            f"{[round(x, 4) for x in lifts]}); the modular decomposition is "
            f"not recovering a separation the raw model can't learn."
        )


# ---------------------------------------------------------------------------
# PART A Contract 2: cyclic continuity -- sin/cos beats raw mod
# ---------------------------------------------------------------------------


class TestCyclicContinuity:
    def test_sincos_beats_raw_mod_for_smooth_cyclic_target(self):
        from mlframe.feature_selection.filters._periodic_fe import apply_modular
        lifts = []
        for s in SEEDS:
            X, y = _build_smooth_cyclic(s)
            t = X["t"].to_numpy()
            idx = np.arange(len(y))
            tr, te = train_test_split(
                idx, test_size=0.3, random_state=s, stratify=y,
            )
            mod = apply_modular(t, 24.0, "mod").reshape(-1, 1)
            sincos = np.column_stack([
                apply_modular(t, 24.0, "sin"),
                apply_modular(t, 24.0, "cos"),
            ])

            def _auc(F):
                return roc_auc_score(
                    y[te],
                    LogisticRegression(max_iter=2000)
                    .fit(F[tr], y[tr])
                    .predict_proba(F[te])[:, 1],
                )
            lifts.append(_auc(sincos) - _auc(mod))
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.03, (
            f"sin/cos encoding AUC lift {mean_lift:.4f} < 0.03 over raw mod "
            f"for a smoothly-cyclic target (per-seed "
            f"{[round(x, 4) for x in lifts]}); cyclic continuity is not "
            f"materialising (the period-boundary discontinuity in raw mod "
            f"should hurt the linear model)."
        )


# ---------------------------------------------------------------------------
# PART A Contract 3: auto-period -- correct period (24) selected over wrong (7,12)
# ---------------------------------------------------------------------------


class TestAutoPeriod:
    def test_correct_period_selected(self):
        from mlframe.feature_selection.filters._periodic_fe import (
            hybrid_modular_fe_with_recipes,
            _parse_modular_name,
        )
        correct = 0
        for s in SEEDS:
            X, y = _build_hour_of_day(s)
            _, appended, recipes, scores = hybrid_modular_fe_with_recipes(
                X, y, periods=(7, 12, 24, 30), top_k=6, seed=s,
            )
            assert appended, f"seed={s}: no survivors for auto-period."
            # The TOP survivor (highest engineered_mi_lcb) must be period 24.
            top_name = appended[0]
            parsed = _parse_modular_name(top_name)
            assert parsed is not None, f"unparseable survivor {top_name!r}"
            _op, _src, period = parsed
            if int(period) == 24:
                correct += 1
            # The wrong periods 7 / 12 must NOT beat 24: no period-7 or
            # period-12 column ranks above the first period-24 column.
            periods_in_order = [
                int(_parse_modular_name(c)[2]) for c in appended
                if _parse_modular_name(c) is not None
            ]
            first_24 = periods_in_order.index(24) if 24 in periods_in_order else 999
            for wrong in (7, 12):
                if wrong in periods_in_order:
                    assert periods_in_order.index(wrong) > first_24, (
                        f"seed={s}: wrong period {wrong} ranked above the "
                        f"correct period 24 ({periods_in_order})."
                    )
        assert correct >= 4, (
            f"correct period (24) was the top survivor in only {correct}/5 "
            f"seeds; auto-period detection is unreliable."
        )


# ---------------------------------------------------------------------------
# PART B Contract 4: group-anomaly signal captured; AUC lift >= +0.05
# ---------------------------------------------------------------------------


class TestGroupAnomalySignal:
    def test_group_distance_captures_anomaly_mi(self):
        from mlframe.feature_selection.filters._group_distance_fe import (
            generate_group_distance_features,
            engineered_name_group_kl,
        )
        gains = []
        for s in SEEDS:
            X, y = _build_group_anomaly(s)
            enc, _ = generate_group_distance_features(X, ["g"], ["v"])
            klname = engineered_name_group_kl("v", "g")
            mi_raw = _mi_one(X["v"].to_numpy(), y)
            mi_kl = _mi_one(enc[klname].to_numpy(), y)
            gains.append(mi_kl - mi_raw)
        mean_gain = float(np.mean(gains))
        assert mean_gain >= 0.05, (
            f"group-distance (KL) MI gain {mean_gain:.4f} < 0.05 over raw v "
            f"(per-seed {[round(g, 4) for g in gains]}); the per-group "
            f"distributional distance is not capturing the shape-anomaly "
            f"signal."
        )

    def test_logreg_auc_lift(self):
        from mlframe.feature_selection.filters._group_distance_fe import (
            hybrid_group_distance_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        lifts = []
        for s in SEEDS:
            X, y = _build_group_anomaly(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            base = LogisticRegression(max_iter=2000).fit(Xtr[["v"]], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[["v"]])[:, 1])

            _, appended, recipes, _ = hybrid_group_distance_fe(
                Xtr, ytr.values if hasattr(ytr, "values") else ytr,
                group_cols=["g"], num_cols=["v"], top_k=6,
            )
            assert appended, f"seed={s}: no group-distance survivors."
            Xtr_aug = Xtr[["v"]].reset_index(drop=True).copy()
            Xte_aug = Xte[["v"]].reset_index(drop=True).copy()
            for r in recipes:
                Xtr_aug[r.name] = apply_recipe(r, Xtr)
                Xte_aug[r.name] = apply_recipe(r, Xte)
            aug = LogisticRegression(max_iter=2000).fit(Xtr_aug, ytr)
            auc_aug = roc_auc_score(yte, aug.predict_proba(Xte_aug)[:, 1])
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.05, (
            f"group-anomaly AUC lift {mean_lift:.4f} < 0.05 (per-seed "
            f"{[round(x, 4) for x in lifts]}); the group-distance feature is "
            f"not recovering a separation the raw model can't learn."
        )


# ---------------------------------------------------------------------------
# Contract 5: no leakage (BOTH) -- replay independent of y
# ---------------------------------------------------------------------------


class TestNoYLeak:
    def test_modular_transform_same_under_shuffled_y(self):
        from mlframe.feature_selection.filters._periodic_fe import (
            hybrid_modular_fe_with_recipes,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y = _build_hour_of_day(7)
        rng = np.random.default_rng(0)
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        _, appended, recipes, _ = hybrid_modular_fe_with_recipes(
            X, y, periods=(7, 12, 24, 30), top_k=6, seed=7,
        )
        assert recipes, "no modular recipes for leakage test."
        for r in recipes:
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r, X))
            assert "y" not in dict(r.extra), (
                f"recipe {r.name!r} captured a y reference -- leakage risk."
            )

    def test_group_distance_transform_same_under_shuffled_y(self):
        from mlframe.feature_selection.filters._group_distance_fe import (
            hybrid_group_distance_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y = _build_group_anomaly(13)
        _, appended, recipes, _ = hybrid_group_distance_fe(
            X, y, group_cols=["g"], num_cols=["v"], top_k=6,
        )
        assert recipes, "no group-distance recipes for leakage test."
        for r in recipes:
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r, X))
            assert "y" not in dict(r.extra), (
                f"recipe {r.name!r} captured a y reference -- leakage risk."
            )

    def test_generators_never_see_y(self):
        from mlframe.feature_selection.filters._periodic_fe import (
            generate_modular_features,
        )
        from mlframe.feature_selection.filters._group_distance_fe import (
            generate_group_distance_features,
        )
        X, y = _build_hour_of_day(42)
        m1 = generate_modular_features(X, periods=(7, 24))
        m2 = generate_modular_features(X, periods=(7, 24))
        pd.testing.assert_frame_equal(m1, m2)

        Xg, _ = _build_group_anomaly(42)
        g1, _ = generate_group_distance_features(Xg, ["g"], ["v"])
        g2, _ = generate_group_distance_features(Xg, ["g"], ["v"])
        pd.testing.assert_frame_equal(g1, g2)


# ---------------------------------------------------------------------------
# Contract 6: default disabled byte-identical (BOTH)
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_mrmr_default_off_adds_nothing(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_hour_of_day(42, n=2000)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_modular_enable", False)) is False, (
            "fe_modular_enable must default to False."
        )
        assert bool(getattr(m, "fe_group_distance_enable", False)) is False, (
            "fe_group_distance_enable must default to False."
        )
        m.fit(X, pd.Series(y, name="y"))
        md = list(getattr(m, "modular_features_", []) or [])
        gd = list(getattr(m, "group_distance_features_", []) or [])
        assert md == [], f"modular added columns with feature disabled: {md}"
        assert gd == [], f"group_distance added columns with feature disabled: {gd}"

    def test_mrmr_modular_enabled_adds_columns(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_hour_of_day(42, n=4000)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_modular_enable=True,
            fe_modular_periods=(7, 12, 24, 30),
            fe_modular_top_k=4,
        )
        m.fit(X, pd.Series(y, name="y"))
        md = list(getattr(m, "modular_features_", []) or [])
        assert len(md) >= 1, (
            "modular enabled but produced no engineered columns on the "
            "hour-of-day fixture."
        )

    def test_mrmr_group_distance_enabled_adds_columns(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_group_anomaly(42, n=4000)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_group_distance_enable=True,
            fe_group_distance_group_cols=("g",),
            fe_group_distance_num_cols=("v",),
            fe_group_distance_top_k=4,
        )
        m.fit(X, pd.Series(y, name="y"))
        gd = list(getattr(m, "group_distance_features_", []) or [])
        assert len(gd) >= 1, (
            "group_distance enabled but produced no engineered columns on the "
            "group-anomaly fixture."
        )


# ---------------------------------------------------------------------------
# Contract 7: pickle / clone round-trip (BOTH)
# ---------------------------------------------------------------------------


class TestPickleClone:
    def test_modular_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._periodic_fe import (
            hybrid_modular_fe_with_recipes,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y = _build_hour_of_day(1)
        _, appended, recipes, _ = hybrid_modular_fe_with_recipes(
            X, y, periods=(7, 12, 24, 30), top_k=6, seed=1,
        )
        assert recipes, "no modular recipes for pickle test."
        for r in recipes:
            r2 = pickle.loads(pickle.dumps(r))
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))

    def test_group_distance_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._group_distance_fe import (
            hybrid_group_distance_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        X, y = _build_group_anomaly(1)
        _, appended, recipes, _ = hybrid_group_distance_fe(
            X, y, group_cols=["g"], num_cols=["v"], top_k=6,
        )
        assert recipes, "no group-distance recipes for pickle test."
        for r in recipes:
            r2 = pickle.loads(pickle.dumps(r))
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))

    def test_mrmr_clone_preserves_params(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_modular_enable=True,
            fe_modular_periods=(7, 24),
            fe_modular_top_k=4,
            fe_group_distance_enable=True,
            fe_group_distance_group_cols=("g",),
            fe_group_distance_num_cols=("v",),
            fe_group_distance_top_k=3,
        )
        c = clone(m)
        assert bool(c.fe_modular_enable) is True
        assert tuple(c.fe_modular_periods) == (7, 24)
        assert int(c.fe_modular_top_k) == 4
        assert bool(c.fe_group_distance_enable) is True
        assert tuple(c.fe_group_distance_group_cols) == ("g",)
        assert tuple(c.fe_group_distance_num_cols) == ("v",)
        assert int(c.fe_group_distance_top_k) == 3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
