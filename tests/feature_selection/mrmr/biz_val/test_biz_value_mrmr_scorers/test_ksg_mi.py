"""Layer 65 biz_value: KSG / k-NN MI ranking for hybrid orth-poly FE.

Validates the new ``score_features_by_ksg_mi_uplift`` /
``hybrid_orth_mi_ksg_fe`` introduced 2026-05-31 (sibling module
``_orthogonal_ksg_mi_fe``). Layer 21 ranks engineered columns via the
plug-in quantile-binned MI estimator (``_mi_classif_batch``); Layer 65
swaps in the Kraskov-Stoegbauer-Grassberger k-NN MI estimator via
sklearn's ``mutual_info_classif`` (Ross 2014 mixed-KSG for discrete y) /
``mutual_info_regression`` (classical KSG for continuous y). KSG is
asymptotically unbiased on continuous data and recovers smooth signal
that binning erases.

What the contract classes pin
-----------------------------

* ``TestKsgVsPlugInContinuousY``: on a continuous target
  ``y = He_3(x1) + sigma * eps`` the KSG estimate of ``MI(x1__He3; y)``
  is HIGHER than the plug-in estimate across every seed -- the headline
  accuracy claim (binning destroys smooth-y structure that KSG sees).

* ``TestSmoothHe3WinsUnderKsg``: on a continuous-y smooth-He_3 fixture
  the KSG ranking pushes ``x1__He3`` materially higher than the
  plug-in's ``x1__He3`` uplift relative to a noise baseline -- KSG
  preserves the smooth-signal win that binning erases.

* ``TestKsgAugmentedAucLift``: end-to-end downstream biz_value. On a
  classification fixture built from a logistic-of-He_3 target, the
  KSG-augmented LogReg AUC beats RAW LogReg AUC on every seed.

* ``TestDefaultDisabledByteIdentical``: default
  ``fe_hybrid_orth_ksg_enable=False`` leaves ``hybrid_orth_features_``
  empty -- legacy behaviour preserved.

* ``TestEnableAppendsEngineered``: turning the flag on appends at least
  one ``x1__*`` engineered column on a clean signal frame.

* ``TestPickleAndClone``: sklearn ``clone`` preserves the 2 ctor params;
  ``pickle`` preserves the appended ``hybrid_orth_features_`` AND the
  ``orth_univariate`` recipes round-trip.

* ``TestRecipeReplay``: each appended column's recipe replays bit-equal
  to its fit-time value.

NEVER xfail. NEVER mask bugs via runtime workarounds.

Consolidated verbatim from test_biz_value_mrmr_layer65.py (per audit finding test_code_quality-16).
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
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


def _import_ksg_fe():
    """Lazily import the Layer-65 KSG k-NN MI scoring/FE functions."""
    from mlframe.feature_selection.filters._orthogonal_ksg_mi_fe import (
        score_features_by_ksg_mi_uplift,
        hybrid_orth_mi_ksg_fe,
        hybrid_orth_mi_ksg_fe_with_recipes,
    )

    return (
        score_features_by_ksg_mi_uplift,
        hybrid_orth_mi_ksg_fe,
        hybrid_orth_mi_ksg_fe_with_recipes,
    )


def _import_plug_in_fe():
    """Lazily import the Layer-21 plug-in marginal-MI univariate FE functions."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
        score_features_by_mi_uplift,
        hybrid_orth_mi_fe,
    )

    return (
        generate_univariate_basis_features,
        score_features_by_mi_uplift,
        hybrid_orth_mi_fe,
    )


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_continuous_he3(seed: int, n: int = 600, n_noise: int = 5):
    """y = He_3(x1) + small noise (continuous target). The cubic ripple is
    SMOOTH and binning y to 10 quantile bins (as the plug-in must) averages
    it out; the KSG k-NN estimator sees the full continuous structure.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    cols: dict = {"x1": x1}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    he3 = x1**3 - 3.0 * x1
    y = he3 + 0.4 * rng.standard_normal(n)
    return X, pd.Series(y, name="y")


def _build_classification_he3(seed: int, n: int = 2000, amp: float = 0.7, n_noise: int = 4):
    """Classification target: y ~ Bernoulli(sigma(amp * He_3(x1))).
    Strong-enough He_3 signal that augmenting with x1__He3 lifts LogReg
    AUC over the linear-only baseline robustly.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    cols: dict = {"x1": x1}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    he3 = x1**3 - 3.0 * x1
    p = 1.0 / (1.0 + np.exp(-float(amp) * he3))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, pd.Series(y, name="y")


from tests.feature_selection._biz_val_synth import _build_linear, _build_quadratic_classif

# ---------------------------------------------------------------------------
# Contract 1: KSG MI > plug-in MI on continuous y
# ---------------------------------------------------------------------------


class TestKsgVsPlugInContinuousY:
    """The headline accuracy claim: KSG estimates a HIGHER MI than the
    plug-in on a smooth continuous target because binning destroys the
    sub-bin-resolution cubic structure that the k-NN estimator sees.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_ksg_mi_greater_than_plugin_on_he3(self, seed):
        """KSG MI on x1__He3 is strictly above the plug-in MI on the smooth continuous-y fixture."""
        score_ksg, _, _ = _import_ksg_fe()
        gen, score_pi, _ = _import_plug_in_fe()
        X, y = _build_continuous_he3(seed)
        eng = gen(X, degrees=(2, 3), basis="hermite")
        # Plug-in must qcut continuous y to discrete bins to operate.
        y_binned = pd.qcut(
            y.to_numpy(),
            q=10,
            labels=False,
            duplicates="drop",
        ).astype(np.int64)
        sc_pi = score_pi(X, eng, y_binned)
        sc_ksg = score_ksg(X, eng, y.to_numpy(), n_neighbors=3, random_state=seed)
        pi_he3 = float(sc_pi[sc_pi["engineered_col"] == "x1__He3"]["engineered_mi"].iloc[0])
        ksg_he3 = float(sc_ksg[sc_ksg["engineered_col"] == "x1__He3"]["engineered_mi"].iloc[0])
        # KSG must recover MORE MI than plug-in on this smooth fixture.
        # The per-seed margin can be small (KSG and plug-in agree to within
        # a few percent on a single seed), but it must be STRICTLY positive;
        # the aggregate 10 % uplift claim lives in
        # ``TestSmoothHe3WinsUnderKsg.test_aggregate_ksg_he3_uplift_vs_plugin``.
        assert ksg_he3 > pi_he3, (
            f"seed={seed}: KSG MI({ksg_he3:.4f}) not above plug-in MI({pi_he3:.4f}) on x1__He3; the smooth-y win the k-NN estimator promises was not realised."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_ksg_mi_nonnegative_finite(self, seed):
        """Mechanical sanity: every KSG MI must be finite and non-negative."""
        score_ksg, _, _ = _import_ksg_fe()
        gen, _, _ = _import_plug_in_fe()
        X, y = _build_continuous_he3(seed)
        eng = gen(X, degrees=(2, 3), basis="hermite")
        sc = score_ksg(X, eng, y.to_numpy(), n_neighbors=3, random_state=seed)
        mis = sc["engineered_mi"].to_numpy()
        assert np.all(np.isfinite(mis)), f"seed={seed}: KSG produced non-finite engineered_mi values"
        # KSG can produce small negative noise on null pairs (Kraskov 2004
        # discusses this); we allow a small tolerance.
        assert mis.min() >= -1e-6, f"seed={seed}: KSG produced materially-negative engineered_mi min={mis.min():.4f}; estimator path is broken."


# ---------------------------------------------------------------------------
# Contract 2: smooth He_3 ranks materially higher under KSG than plug-in
# ---------------------------------------------------------------------------


class TestSmoothHe3WinsUnderKsg:
    """On the smooth continuous-y fixture, ranking by KSG vs plug-in
    delivers different rankings; the KSG ranking gives x1__He3 a higher
    MI value AND a comparable-or-better uplift floor relative to noise.
    """

    def test_aggregate_ksg_he3_uplift_vs_plugin(self):
        """Aggregate over multiple seeds: KSG He_3 MI averages higher
        than plug-in He_3 MI by at least 10 % -- this is the
        statistical witness that binning destroys signal that KSG sees.
        """
        score_ksg, _, _ = _import_ksg_fe()
        gen, score_pi, _ = _import_plug_in_fe()
        pi_he3_mis, ksg_he3_mis = [], []
        for s in (1, 7, 13, 42, 101, 202, 303, 404):
            X, y = _build_continuous_he3(s)
            eng = gen(X, degrees=(2, 3), basis="hermite")
            y_binned = pd.qcut(
                y.to_numpy(),
                q=10,
                labels=False,
                duplicates="drop",
            ).astype(np.int64)
            sc_pi = score_pi(X, eng, y_binned)
            sc_ksg = score_ksg(X, eng, y.to_numpy(), n_neighbors=3, random_state=s)
            pi_he3_mis.append(float(sc_pi[sc_pi["engineered_col"] == "x1__He3"]["engineered_mi"].iloc[0]))
            ksg_he3_mis.append(float(sc_ksg[sc_ksg["engineered_col"] == "x1__He3"]["engineered_mi"].iloc[0]))
        pi_mean = float(np.mean(pi_he3_mis))
        ksg_mean = float(np.mean(ksg_he3_mis))
        assert ksg_mean > pi_mean * 1.10, (
            f"KSG mean MI on x1__He3 ({ksg_mean:.4f}) not materially "
            f"above plug-in mean ({pi_mean:.4f}); smooth-y advantage not "
            f"realised aggregated across 8 seeds."
        )
        # Per-seed: KSG > plug-in on EVERY seed -- the win is not seed-
        # dependent / not driven by one tail outlier.
        wins = sum(k > p for k, p in zip(ksg_he3_mis, pi_he3_mis))
        assert wins == len(pi_he3_mis), f"KSG only beat plug-in on x1__He3 MI in {wins}/{len(pi_he3_mis)} seeds; the per-seed accuracy claim is broken."


# ---------------------------------------------------------------------------
# Contract 3: KSG-augmented LogReg AUC > raw LogReg AUC
# ---------------------------------------------------------------------------


class TestKsgAugmentedAucLift:
    """End-to-end biz_value: augmenting the raw frame with KSG-selected
    orth-poly columns lifts downstream LogReg AUC over the raw baseline
    on every seed. The He_3 cubic signal is a known LogReg blind spot
    (LogReg is linear in its inputs); appending ``x1__He3`` -- which
    KSG recovers correctly -- gives the linear model the missing
    feature it needs.
    """

    def test_ksg_augmented_logreg_auc_beats_raw(self):
        """KSG-augmented LogReg AUC beats raw by >= 0.05 mean lift, and on every seed, on the He_3 classification fixture."""
        _, hybrid_ksg, _ = _import_ksg_fe()
        gen, _, _ = _import_plug_in_fe()
        aucs_raw, aucs_ksg = [], []
        for s in (1, 7, 13, 42, 101, 202, 303, 404):
            X, y = _build_classification_he3(s, n=2000, amp=0.7)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=s,
                stratify=y,
            )
            # Raw baseline -- linear LogReg can't recover the He_3 signal.
            lr_raw = LogisticRegression(max_iter=2000, solver="lbfgs").fit(
                X_tr,
                y_tr,
            )
            aucs_raw.append(
                roc_auc_score(
                    y_te,
                    lr_raw.predict_proba(X_te)[:, 1],
                )
            )
            # KSG-augmented -- top-K under KSG uplift adds ``x1__He3``.
            X_aug_tr, _ = hybrid_ksg(
                X_tr,
                y_tr.to_numpy(),
                degrees=(2, 3),
                basis="hermite",
                top_k=3,
                min_uplift=0.5,
                min_abs_mi_frac=0.0,
                n_neighbors=3,
                random_state=s,
            )
            added = [c for c in X_aug_tr.columns if c not in X_tr.columns]
            # Rebuild the same engineered cols on test (closed-form basis
            # eval; no y leakage by construction).
            eng_te = gen(X_te, degrees=(2, 3), basis="hermite")
            X_aug_te = pd.concat([X_te, eng_te[added]], axis=1) if added else X_te
            lr_aug = LogisticRegression(max_iter=2000, solver="lbfgs").fit(
                X_aug_tr,
                y_tr,
            )
            aucs_ksg.append(
                roc_auc_score(
                    y_te,
                    lr_aug.predict_proba(X_aug_te)[:, 1],
                )
            )
        raw_mean = float(np.mean(aucs_raw))
        ksg_mean = float(np.mean(aucs_ksg))
        # Mean lift must clear at least 5 AUC points -- the He_3 signal is
        # by construction strong enough for any sensible estimator to
        # recover, and we picked top_k=3 / min_uplift=0.5 specifically so
        # KSG actually emits x1__He3 here.
        assert ksg_mean > raw_mean + 0.05, (
            f"KSG-augmented LogReg AUC mean ({ksg_mean:.4f}) not lifted "
            f"by >=0.05 vs raw mean ({raw_mean:.4f}); biz_value lift "
            f"claim violated.\nraw_per_seed={aucs_raw}\n"
            f"ksg_per_seed={aucs_ksg}"
        )
        # Per-seed: lift on EVERY seed -- a robust biz_value floor.
        wins = sum(k > r for k, r in zip(aucs_ksg, aucs_raw))
        assert wins == len(aucs_raw), f"KSG-augmented AUC only beat raw on {wins}/{len(aucs_raw)} seeds; per-seed lift floor violated."


# ---------------------------------------------------------------------------
# Contract 4: default disabled byte-identical with master
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_ksg_enable defaults to False, with the documented n_neighbors/uplift/abs-frac defaults."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_ksg_columns(self, seed):
        """With the flag left at its False default, no KSG columns are appended."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], f"seed={seed}: default fe_hybrid_orth_ksg_enable=False should NOT append any engineered columns; got {added}"

    def test_default_ctor_values(self):
        """Default ctor values match the documented KSG defaults (n_neighbors=3, min_uplift=0.95, min_abs_mi_frac=0.05)."""
        m = _make_mrmr()
        assert m.fe_hybrid_orth_ksg_enable is False
        assert m.fe_hybrid_orth_ksg_n_neighbors == 3
        assert m.fe_hybrid_orth_ksg_min_uplift == 0.95
        assert m.fe_hybrid_orth_ksg_min_abs_mi_frac == 0.05


# ---------------------------------------------------------------------------
# Contract 5: enabling appends engineered columns on a clean signal
# ---------------------------------------------------------------------------


class TestEnableAppendsEngineered:
    """Enabling KSG FE on a clean quadratic-signal fixture must append an x1 basis column."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_enable_appends_x1_basis_column(self, seed):
        """Enabling KSG appends at least one x1-referencing engineered column."""
        X, y = _build_quadratic_classif(seed, n=1500)
        m = _make_mrmr(
            fe_hybrid_orth_ksg_enable=True,
            fe_hybrid_orth_ksg_n_neighbors=3,
            fe_hybrid_orth_degrees=(2, 3),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added, f"seed={seed}: KSG flag ON should append at least one engineered column to hybrid_orth_features_; got {added}"
        assert any(c.startswith("x1__") for c in added), f"seed={seed}: KSG winners should include an x1 basis column on a clean He_2 signal; got {added}"


# ---------------------------------------------------------------------------
# Contract 6: pickle / clone preserve the ctor + recipes round-trip
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """KSG ctor params and recipes must survive clone/pickle round-trips."""

    def test_clone_preserves_ksg_params(self):
        """sklearn clone() copies every fe_hybrid_orth_ksg_* ctor param."""
        m = _make_mrmr(
            fe_hybrid_orth_ksg_enable=True,
            fe_hybrid_orth_ksg_n_neighbors=7,
            fe_hybrid_orth_ksg_min_uplift=0.85,
            fe_hybrid_orth_ksg_min_abs_mi_frac=0.0,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_ksg_enable", True),
            ("fe_hybrid_orth_ksg_n_neighbors", 7),
            ("fe_hybrid_orth_ksg_min_uplift", 0.85),
            ("fe_hybrid_orth_ksg_min_abs_mi_frac", 0.0),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got {getattr(m2, name)}"

    def test_pickle_roundtrip_preserves_ksg_recipes(self):
        """A pickle round-trip preserves feature names, appended columns, and every orth_univariate recipe field."""
        X, y = _build_quadratic_classif(seed=42, n=1500)
        m = _make_mrmr(
            fe_hybrid_orth_ksg_enable=True,
            fe_hybrid_orth_ksg_n_neighbors=3,
            fe_hybrid_orth_degrees=(2, 3),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, f"pickle changed hybrid_orth_features_: before={added_before}, after={added_after}"

        # All KSG-stage recipes are ``orth_univariate`` -- the engineered
        # VALUES are bit-equal to Layer 21, only the SCORING differs.
        def _extract_orth_recipes(model):
            """Return {name: recipe} for the orth_univariate recipes, regardless of container list/dict shape."""
            container = getattr(model, "_engineered_recipes_", None)
            if isinstance(container, dict):
                return {r.name: r for r in container.values() if getattr(r, "kind", None) == "orth_univariate"}
            return {r.name: r for r in (container or []) if getattr(r, "kind", None) == "orth_univariate"}

        recipes_before = _extract_orth_recipes(m)
        recipes_after = _extract_orth_recipes(m2)
        assert set(recipes_before.keys()) == set(recipes_after.keys()), (
            f"pickle dropped or added orth_univariate recipe names: before={set(recipes_before.keys())}, after={set(recipes_after.keys())}"
        )
        for name, r_before in recipes_before.items():
            r_after = recipes_after[name]
            assert r_before.src_names == r_after.src_names, f"pickle changed src_names for {name!r}: before={r_before.src_names}, after={r_after.src_names}"
            for key in ("basis", "degree"):
                assert r_before.extra.get(key) == r_after.extra.get(key), (
                    f"pickle changed '{key}' for recipe {name!r}: before={r_before.extra}, after={r_after.extra}"
                )


# ---------------------------------------------------------------------------
# Contract 7: recipe replay reproduces fit-time values bit-equivalently
# ---------------------------------------------------------------------------


class TestRecipeReplay:
    """apply_recipe at transform time must reproduce the fit-time KSG-selected engineered column."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_recipe_replay_matches_fit_time(self, seed):
        """Replaying the fit-time recipes on the same X reproduces identical engineered values."""
        _, _, hybrid_with_recipes = _import_ksg_fe()
        # Continuous-y smooth He_3 fixture -- KSG / regression path is the
        # one where x1__He3 reliably enters the support via the uplift gate
        # (continuous y gives KSG the headroom to score x1__He3 above the
        # raw x1 baseline). Classification fixtures with strict gates
        # frequently leave the engineered support empty because KSG's
        # k-NN already picks up the non-monotone signal in raw x1.
        X, y = _build_continuous_he3(seed, n=600)
        X_aug, _scores, recipes = hybrid_with_recipes(
            X,
            y.to_numpy(),
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            min_uplift=0.5,
            min_abs_mi_frac=0.0,
            n_neighbors=3,
            random_state=seed,
        )
        if not recipes:
            pytest.fail(f"seed={seed}: KSG hybrid emitted no recipes; replay contract requires at least one recipe on a continuous-y smooth-He_3 fixture.")
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        appended = [c for c in X_aug.columns if c not in X.columns]
        for r in recipes:
            assert r.name in appended, f"seed={seed}: recipe {r.name!r} not in appended columns {appended}"
            assert r.kind == "orth_univariate", (
                f"seed={seed}: KSG-stage recipe {r.name!r} kind="
                f"{r.kind!r}, expected 'orth_univariate' (engineered "
                f"values are bit-equal to Layer 21; only scoring differs)."
            )
            replayed = apply_recipe(r, X)
            fit_time = X_aug[r.name].to_numpy()
            assert np.allclose(replayed, fit_time, rtol=1e-9, atol=1e-12), (
                f"seed={seed}: recipe {r.name!r} replay drift: max|replayed - fit| = {float(np.max(np.abs(replayed - fit_time)))}; extra={dict(r.extra)}"
            )
