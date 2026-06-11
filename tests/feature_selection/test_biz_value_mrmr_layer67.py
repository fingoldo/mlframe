"""Layer 67 biz_value: distance-correlation ranking for hybrid orth-poly FE.

Validates ``distance_correlation`` / ``score_features_by_dcor_uplift`` /
``hybrid_orth_mi_dcor_fe`` (sibling module ``_orthogonal_dcor_fe``)
introduced 2026-06-01. Layers 21 / 65 / 66 are all MI estimators (plug-in,
KSG k-NN, copula); Layer 67 is the Szekely-Rizzo distance correlation --
a NON-MI dependence measure with the universal ``dCor == 0`` iff
independence guarantee that Pearson lacks. Detects ANY relationship
(monotone, non-monotone, non-functional), not just linear / rank.

Contracts pinned
----------------

* ``TestDcorDetectsNonMonotone``: ``y = sin(2 * pi * x)`` -- Pearson is
  near zero (non-monotone), dCor is materially positive.
* ``TestDcorZeroOnIndependence``: independent noise pairs give dCor near
  zero (within the small-sample tail).
* ``TestDcorSymmetric``: ``dCor(x, y) == dCor(y, x)``.
* ``TestAucLiftOnNonMonotone``: end-to-end -- on a non-monotone
  classification fixture the dCor-augmented LogReg AUC beats the raw
  LogReg AUC.
* ``TestDefaultDisabledByteIdentical``: master switch OFF leaves
  ``hybrid_orth_features_`` empty.
* ``TestPickleAndClone``: ``clone`` preserves ctor params; ``pickle``
  preserves appended features and recipe round-trip.

NEVER xfail.
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


def _import_dcor_fe():
    from mlframe.feature_selection.filters._orthogonal_dcor_fe import (
        distance_correlation,
        score_features_by_dcor_uplift,
        hybrid_orth_mi_dcor_fe,
        hybrid_orth_mi_dcor_fe_with_recipes,
    )
    return (
        distance_correlation,
        score_features_by_dcor_uplift,
        hybrid_orth_mi_dcor_fe,
        hybrid_orth_mi_dcor_fe_with_recipes,
    )


def _import_plug_in_fe():
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
    )
    return generate_univariate_basis_features


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders -- non-monotone / oscillatory fixtures
# ---------------------------------------------------------------------------


def _build_sin_signal(seed: int, n: int = 600):
    """``y = cos(pi * x) + small noise`` on ``x ~ Uniform(-1, 1)``.
    Strongly non-monotone with a clean Pearson blind spot: ``cos`` is an
    EVEN function so ``E[x * cos(pi * x)] = 0`` by symmetry of the
    uniform measure -- the population Pearson is exactly zero. dCor
    materialises the true periodic dependence regardless. (Naming kept
    as ``sin_signal`` from the layer brief; the trigonometric choice is
    the textbook Pearson-zero / dCor-positive construction.)
    """
    rng = np.random.default_rng(int(seed))
    x = rng.uniform(low=-1.0, high=1.0, size=n)
    y = np.cos(np.pi * x) + 0.05 * rng.standard_normal(n)
    return x, y


def _build_non_monotone_classif(seed: int, n: int = 1500, n_noise: int = 4):
    """Classification fixture with a non-monotone signal: ``y = sign(x1^2
    + cos(2 pi x1) > thr)``. LogReg cannot recover it from raw x1 (the
    decision boundary is non-monotone in x1); appending a dCor-selected
    He_2 / He_3 column gives the linear model the missing degree of
    freedom.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.uniform(low=-1.5, high=1.5, size=n)
    cols: dict = {"x1": x1}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    signal = x1 ** 2 + 0.4 * np.cos(2.0 * np.pi * x1)
    thr = float(np.median(signal))
    y = ((signal + 0.1 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


def _build_quadratic_classif(seed: int, n: int = 1500, n_noise: int = 5):
    """Clean He_2 signal -- ``y = sign(x^2 > 1)``. Used for the enable /
    pickle contracts where we want a stable winner appended.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    cols: dict = {"x1": x1}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = ((x1 ** 2 + 0.1 * rng.standard_normal(n)) > 1.0).astype(int)
    return X, pd.Series(y, name="y")


from tests.feature_selection._biz_val_synth import _build_linear
# ---------------------------------------------------------------------------
# Contract 1: dCor detects non-monotone dependence where Pearson misses
# ---------------------------------------------------------------------------


class TestDcorDetectsNonMonotone:
    """The headline property: dCor != 0 iff dependent. Pearson is zero
    on non-monotone signals; dCor is materially positive on the same
    pair because the centred distance matrices encode the dependence
    structure regardless of the direction of the slope.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_dcor_above_zero_on_sin_signal(self, seed):
        distance_correlation, _, _, _ = _import_dcor_fe()
        x, y = _build_sin_signal(seed, n=500)
        dcor = float(distance_correlation(
            x, y, n_sample=500, random_state=seed,
        ))
        pearson = float(np.corrcoef(x, y)[0, 1])
        # dCor must clear a 0.40 floor on the periodic dependence at n=500
        # (population dCor for y = cos(pi*x) on Uniform(-1, 1) is around
        # 0.5; the small-sample noise band is ~0.05).
        assert dcor > 0.40, (
            f"seed={seed}: dCor on y = cos(pi*x) = {dcor:.4f}; "
            f"expected > 0.40 (universal-detection contract)."
        )
        # Pearson is exactly zero in the limit (cos is even on a
        # symmetric uniform support); the small-sample tail at n=500 is
        # comfortably below 0.20.
        assert abs(pearson) < 0.20, (
            f"seed={seed}: Pearson on y = cos(pi*x) = {pearson:.4f}; "
            f"expected near zero (Pearson blind spot)."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_dcor_above_zero_on_quadratic(self, seed):
        """Classic Pearson blind spot: ``y = x^2`` with symmetric x. dCor
        sees the quadratic dependence; Pearson is ~zero by symmetry.
        """
        distance_correlation, _, _, _ = _import_dcor_fe()
        rng = np.random.default_rng(int(seed))
        x = rng.standard_normal(500)
        y = x ** 2 + 0.05 * rng.standard_normal(500)
        dcor = float(distance_correlation(
            x, y, n_sample=500, random_state=seed,
        ))
        pearson = float(np.corrcoef(x, y)[0, 1])
        assert dcor > 0.40, (
            f"seed={seed}: dCor on y = x^2 = {dcor:.4f}; expected > 0.40 "
            f"(quadratic dependence at n=500 is unambiguous)."
        )
        # Standard-normal x gives Pearson(x, x^2) = 0 by symmetry; at
        # n=500 the sample correlation tail can reach +- 0.25 on a
        # heavy-tail-x-cubed-residual seed (the x^3 term inside Cov(x, x^2)
        # = E[x^3] is zero only in the limit).
        assert abs(pearson) < 0.25, (
            f"seed={seed}: Pearson on y = x^2 = {pearson:.4f}; expected "
            f"near zero (Pearson blind spot)."
        )


# ---------------------------------------------------------------------------
# Contract 2: dCor near zero on independent noise pairs
# ---------------------------------------------------------------------------


class TestDcorZeroOnIndependence:
    """dCor(X, Y) -> 0 as n -> infty iff X and Y are independent
    (Szekely-Rizzo 2007 Thm 3). At n=500 the small-sample tail leaves a
    floor around 0.05-0.10 -- we pin a comfortable upper bound here so
    independence is reliably identified.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_dcor_near_zero_on_independent_gaussian(self, seed):
        distance_correlation, _, _, _ = _import_dcor_fe()
        rng = np.random.default_rng(int(seed))
        n = 500
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        dcor = float(distance_correlation(
            x, y, n_sample=500, random_state=seed,
        ))
        # Independent N(0, 1) at n=500: dCor floor is well under 0.20.
        assert dcor < 0.20, (
            f"seed={seed}: dCor on independent Gaussian pair = {dcor:.4f}; "
            f"expected < 0.20 (independence sanity at n=500)."
        )

    def test_dcor_signal_above_noise(self):
        """Aggregate witness: dCor on the sin-signal fixture beats dCor
        on a paired independent-noise fixture by >= 3x across 5 seeds.
        """
        distance_correlation, _, _, _ = _import_dcor_fe()
        signals, noises = [], []
        for s in SEEDS:
            x, y = _build_sin_signal(s, n=500)
            signals.append(distance_correlation(
                x, y, n_sample=500, random_state=s,
            ))
            rng = np.random.default_rng(int(s) + 9000)
            xn = rng.standard_normal(500)
            yn = rng.standard_normal(500)
            noises.append(distance_correlation(
                xn, yn, n_sample=500, random_state=s,
            ))
        signal_mean = float(np.mean(signals))
        noise_mean = float(np.mean(noises))
        assert signal_mean > 3.0 * noise_mean, (
            f"signal-mean dCor ({signal_mean:.4f}) not >= 3x noise-mean "
            f"dCor ({noise_mean:.4f}); the signal/noise discrimination "
            f"contract is broken.\nsignals={signals}\nnoises={noises}"
        )


# ---------------------------------------------------------------------------
# Contract 3: dCor symmetric (definition-level property)
# ---------------------------------------------------------------------------


class TestDcorSymmetric:
    """dCor(X, Y) == dCor(Y, X) by construction -- the centred distance
    matrices A and B are interchangeable in the dCov^2 = mean(A * B)
    expression.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_dcor_symmetric_on_signal(self, seed):
        distance_correlation, _, _, _ = _import_dcor_fe()
        x, y = _build_sin_signal(seed, n=400)
        d_xy = float(distance_correlation(
            x, y, n_sample=500, random_state=seed,
        ))
        d_yx = float(distance_correlation(
            y, x, n_sample=500, random_state=seed,
        ))
        assert d_xy == pytest.approx(d_yx, abs=1e-12), (
            f"seed={seed}: dCor not symmetric; dCor(x, y) = {d_xy}, "
            f"dCor(y, x) = {d_yx}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_dcor_symmetric_on_noise(self, seed):
        distance_correlation, _, _, _ = _import_dcor_fe()
        rng = np.random.default_rng(int(seed))
        x = rng.standard_normal(400)
        y = rng.standard_normal(400)
        d_xy = float(distance_correlation(
            x, y, n_sample=500, random_state=seed,
        ))
        d_yx = float(distance_correlation(
            y, x, n_sample=500, random_state=seed,
        ))
        assert d_xy == pytest.approx(d_yx, abs=1e-12), (
            f"seed={seed}: dCor not symmetric on noise pair; "
            f"dCor(x, y) = {d_xy}, dCor(y, x) = {d_yx}"
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC lift on a non-monotone classification fixture
# ---------------------------------------------------------------------------


class TestAucLiftOnNonMonotone:
    """End-to-end biz_value: appending dCor-selected orth-poly columns
    lifts downstream LogReg AUC over the raw baseline on a non-monotone
    classification fixture. Raw LogReg sees only the linear projection of
    x1 onto y, which is near-zero by symmetry of the decision boundary;
    appending ``x1__He2`` / ``x1__He3`` gives it the missing
    non-monotone degrees of freedom.
    """

    def test_dcor_augmented_logreg_auc_beats_raw(self):
        _, _, _, hybrid_with_recipes = _import_dcor_fe()
        gen = _import_plug_in_fe()
        aucs_raw, aucs_dcor = [], []
        for s in (1, 7, 13, 42, 101, 202, 303, 404):
            X, y = _build_non_monotone_classif(s, n=1500)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            lr_raw = LogisticRegression(
                max_iter=2000, solver="lbfgs",
            ).fit(X_tr, y_tr)
            aucs_raw.append(roc_auc_score(
                y_te, lr_raw.predict_proba(X_te)[:, 1],
            ))
            X_aug_tr, _scores, _recipes = hybrid_with_recipes(
                X_tr, y_tr.to_numpy(),
                degrees=(2, 3), basis="hermite",
                top_k=3, min_uplift=0.5, min_abs_mi_frac=0.0,
                n_sample=500, random_state=s,
            )
            added = [c for c in X_aug_tr.columns if c not in X_tr.columns]
            eng_te = gen(X_te, degrees=(2, 3), basis="hermite")
            X_aug_te = (
                pd.concat([X_te, eng_te[added]], axis=1)
                if added else X_te
            )
            lr_aug = LogisticRegression(
                max_iter=2000, solver="lbfgs",
            ).fit(X_aug_tr, y_tr)
            aucs_dcor.append(roc_auc_score(
                y_te, lr_aug.predict_proba(X_aug_te)[:, 1],
            ))
        raw_mean = float(np.mean(aucs_raw))
        dcor_mean = float(np.mean(aucs_dcor))
        # Non-monotone signal: raw LogReg is near 0.5 AUC by symmetry of
        # the decision boundary; dCor augmentation appends He_2 which
        # gives a >= 0.05 AUC lift on a strong-enough fixture.
        assert dcor_mean > raw_mean + 0.05, (
            f"dCor-augmented LogReg AUC mean ({dcor_mean:.4f}) not lifted "
            f"by >= 0.05 vs raw mean ({raw_mean:.4f}); biz_value lift "
            f"claim violated.\nraw_per_seed={aucs_raw}\n"
            f"dcor_per_seed={aucs_dcor}"
        )
        wins = sum(d > r for d, r in zip(aucs_dcor, aucs_raw))
        assert wins >= len(aucs_raw) - 1, (
            f"dCor-augmented AUC beat raw on only {wins}/{len(aucs_raw)} "
            f"seeds; per-seed floor too soft."
        )


# ---------------------------------------------------------------------------
# Contract 5: default disabled byte-identical with master
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_dcor_columns(self, seed):
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], (
            f"seed={seed}: default fe_hybrid_orth_dcor_enable=False "
            f"should NOT append any engineered columns; got {added}"
        )

    def test_default_ctor_values(self):
        m = _make_mrmr()
        assert m.fe_hybrid_orth_dcor_enable is False
        assert m.fe_hybrid_orth_dcor_n_sample == 500


# ---------------------------------------------------------------------------
# Contract 6: pickle / clone preserve the ctor + recipes round-trip
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_clone_preserves_dcor_params(self):
        m = _make_mrmr(
            fe_hybrid_orth_dcor_enable=True,
            fe_hybrid_orth_dcor_n_sample=300,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_dcor_enable", True),
            ("fe_hybrid_orth_dcor_n_sample", 300),
        ]:
            assert getattr(m2, name) == expected, (
                f"clone() dropped {name}: expected {expected}, got "
                f"{getattr(m2, name)}"
            )

    def test_pickle_roundtrip_preserves_dcor_recipes(self):
        X, y = _build_quadratic_classif(seed=42, n=1500)
        m = _make_mrmr(
            fe_hybrid_orth_dcor_enable=True,
            fe_hybrid_orth_dcor_n_sample=500,
            fe_hybrid_orth_degrees=(2, 3),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), (
            "pickle changed feature_names_in_"
        )
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, (
            f"pickle changed hybrid_orth_features_: "
            f"before={added_before}, after={added_after}"
        )

        # dCor-stage recipes are ``orth_univariate`` (engineered VALUES
        # bit-equal to Layer 21; only SCORING differs).
        def _extract_orth_recipes(model):
            container = getattr(model, "_engineered_recipes_", None)
            if isinstance(container, dict):
                return {
                    r.name: r for r in container.values()
                    if getattr(r, "kind", None) == "orth_univariate"
                }
            return {
                r.name: r for r in (container or [])
                if getattr(r, "kind", None) == "orth_univariate"
            }
        recipes_before = _extract_orth_recipes(m)
        recipes_after = _extract_orth_recipes(m2)
        assert set(recipes_before.keys()) == set(recipes_after.keys()), (
            f"pickle dropped or added orth_univariate recipe names: "
            f"before={set(recipes_before.keys())}, "
            f"after={set(recipes_after.keys())}"
        )
        for name, r_before in recipes_before.items():
            r_after = recipes_after[name]
            assert r_before.src_names == r_after.src_names, (
                f"pickle changed src_names for {name!r}: "
                f"before={r_before.src_names}, after={r_after.src_names}"
            )
            for key in ("basis", "degree"):
                assert r_before.extra.get(key) == r_after.extra.get(key), (
                    f"pickle changed '{key}' for recipe {name!r}: "
                    f"before={r_before.extra}, after={r_after.extra}"
                )
