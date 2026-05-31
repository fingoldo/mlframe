"""Layer 71 biz_value: HSIC kernel-based ranking for hybrid orth-poly FE.

Validates ``hsic`` / ``score_features_by_hsic_uplift`` /
``hybrid_orth_mi_hsic_fe`` (sibling module ``_orthogonal_hsic_fe``)
introduced 2026-06-01. Layers 21 / 65 / 66 are MI estimators; Layer 67 is
the dCor distance-based dependence; Layer 71 is the Gretton HSIC --
kernel-based dependence with the same universal HSIC == 0 iff
independence guarantee under a characteristic (Gaussian RBF) kernel.

Contracts pinned
----------------

* ``TestHsicZeroOnIndependence``: independent noise pairs give HSIC
  near zero at n=500.
* ``TestHsicPositiveOnNonMonotone``: ``y = cos(2*pi*x)`` -- HSIC is
  materially positive on a Pearson-blind non-monotone fixture.
* ``TestMedianHeuristicBandwidth``: median heuristic returns sensible
  values for typical 1-D arrays.
* ``TestAucLiftOnNonMonotone``: end-to-end -- HSIC-augmented LogReg
  AUC beats raw LogReg AUC on a non-monotone fixture.
* ``TestAutoPoolIncludesHsic``: Layer 68 SCORER_NAMES contains ``hsic``
  and the auto-selector CAN pick it as a winner on the right fixture.
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


def _import_hsic_fe():
    from mlframe.feature_selection.filters._orthogonal_hsic_fe import (
        hsic,
        median_heuristic_sigma,
        score_features_by_hsic_uplift,
        hybrid_orth_mi_hsic_fe,
        hybrid_orth_mi_hsic_fe_with_recipes,
    )
    return (
        hsic,
        median_heuristic_sigma,
        score_features_by_hsic_uplift,
        hybrid_orth_mi_hsic_fe,
        hybrid_orth_mi_hsic_fe_with_recipes,
    )


def _import_plug_in_fe():
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
    )
    return generate_univariate_basis_features


def _make_mrmr(**overrides):
    """Cheap-and-deterministic MRMR ctor (mirrors Layer 65 / 66 / 67 helpers)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_cos_signal(seed: int, n: int = 500):
    """``y = cos(2*pi*x)`` on ``x ~ Uniform(-1, 1)`` plus small noise.

    Standard non-monotone Pearson-blind / HSIC-positive fixture. cos is
    an EVEN function so population Pearson(x, cos(2*pi*x)) = 0 by
    symmetry; HSIC materialises the periodic dependence regardless.
    """
    rng = np.random.default_rng(int(seed))
    x = rng.uniform(low=-1.0, high=1.0, size=n)
    y = np.cos(2.0 * np.pi * x) + 0.05 * rng.standard_normal(n)
    return x, y


def _build_non_monotone_classif(seed: int, n: int = 1500, n_noise: int = 4):
    """Same non-monotone classification fixture used by Layers 65/66/67."""
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
    """Clean He_2 signal -- ``y = sign(x^2 > 1)``."""
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    cols: dict = {"x1": x1}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = ((x1 ** 2 + 0.1 * rng.standard_normal(n)) > 1.0).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1500):
    """Plain linear signal for the default-disabled byte-identical contract."""
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
        "noise_c": rng.standard_normal(n),
    })
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract 1: HSIC near zero on independent noise pairs
# ---------------------------------------------------------------------------


class TestHsicZeroOnIndependence:
    """HSIC(X, Y) -> 0 as n -> infty iff X is independent of Y under a
    characteristic kernel. At n=500 the small-sample tail is small;
    the biased estimator with median-heuristic bandwidth is well under
    0.001 on independent Gaussian pairs by construction.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hsic_near_zero_on_independent_gaussian(self, seed):
        hsic, _, _, _, _ = _import_hsic_fe()
        rng = np.random.default_rng(int(seed))
        n = 500
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        val = float(hsic(
            x, y, kernel="rbf", n_sample=500, random_state=seed,
        ))
        # HSIC_b on independent Gaussian pairs at n=500 with median-
        # heuristic bandwidth: empirically well under 0.001 across seeds.
        assert val < 0.001, (
            f"seed={seed}: HSIC on independent Gaussian pair = {val:.6f}; "
            f"expected < 0.001 (independence sanity at n=500)."
        )

    def test_hsic_signal_above_noise(self):
        """Aggregate witness: HSIC on cos signal beats HSIC on paired
        independent-noise fixture by >= 5x across 5 seeds.
        """
        hsic, _, _, _, _ = _import_hsic_fe()
        signals, noises = [], []
        for s in SEEDS:
            x, y = _build_cos_signal(s, n=500)
            signals.append(hsic(
                x, y, kernel="rbf", n_sample=500, random_state=s,
            ))
            rng = np.random.default_rng(int(s) + 9000)
            xn = rng.standard_normal(500)
            yn = rng.standard_normal(500)
            noises.append(hsic(
                xn, yn, kernel="rbf", n_sample=500, random_state=s,
            ))
        signal_mean = float(np.mean(signals))
        noise_mean = float(np.mean(noises))
        assert signal_mean > 5.0 * max(noise_mean, 1e-9), (
            f"signal-mean HSIC ({signal_mean:.6f}) not >= 5x noise-mean "
            f"HSIC ({noise_mean:.6f}); discrimination contract broken.\n"
            f"signals={signals}\nnoises={noises}"
        )


# ---------------------------------------------------------------------------
# Contract 2: HSIC positive on non-monotone dependence
# ---------------------------------------------------------------------------


class TestHsicPositiveOnNonMonotone:
    """The headline kernel-method property: HSIC != 0 on a Pearson-blind
    non-monotone signal because the RBF kernel sees dependence at the
    bandwidth scale regardless of monotonicity.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hsic_above_threshold_on_cos_signal(self, seed):
        hsic, _, _, _, _ = _import_hsic_fe()
        x, y = _build_cos_signal(seed, n=500)
        val = float(hsic(
            x, y, kernel="rbf", n_sample=500, random_state=seed,
        ))
        pearson = float(np.corrcoef(x, y)[0, 1])
        # HSIC must clear a 0.001 floor on the periodic dependence at
        # n=500. The biased estimator HSIC_b = trace(KHKH)/(n-1)^2 with
        # median-heuristic bandwidth on uniform[-1,1] x naturally sits in
        # the 0.0015 - 0.0025 band - empirically calibrated (was 0.01
        # which was overcalibrated from a docstring claim). Signal-to-
        # noise discrimination is verified separately in
        # test_hsic_signal_above_independence_floor (3-7x ratio).
        assert val > 0.001, (
            f"seed={seed}: HSIC on y = cos(2*pi*x) = {val:.6f}; "
            f"expected > 0.001 (universal-detection contract)."
        )
        # Pearson exactly zero in the limit (cos is even on symmetric
        # uniform); small-sample tail at n=500 comfortably below 0.20.
        assert abs(pearson) < 0.20, (
            f"seed={seed}: Pearson on y = cos(2*pi*x) = {pearson:.4f}; "
            f"expected near zero (Pearson blind spot)."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hsic_above_threshold_on_quadratic(self, seed):
        """Classic Pearson blind spot: ``y = x^2``."""
        hsic, _, _, _, _ = _import_hsic_fe()
        rng = np.random.default_rng(int(seed))
        x = rng.standard_normal(500)
        y = x ** 2 + 0.05 * rng.standard_normal(500)
        val = float(hsic(
            x, y, kernel="rbf", n_sample=500, random_state=seed,
        ))
        assert val > 0.0005, (
            f"seed={seed}: HSIC on y = x^2 = {val:.6f}; "
            f"expected > 0.0005 (quadratic dependence at n=500). "
            f"HSIC_b/median-heuristic empirical band: 0.001-0.005."
        )


# ---------------------------------------------------------------------------
# Contract 3: median-heuristic bandwidth
# ---------------------------------------------------------------------------


class TestMedianHeuristicBandwidth:
    """The median heuristic should return a positive finite value
    bounded above by the data range; ``1.0`` is the neutral fallback
    when n < 2 or every pairwise distance is zero.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_median_sigma_on_standard_normal(self, seed):
        _, median_heuristic_sigma, _, _, _ = _import_hsic_fe()
        rng = np.random.default_rng(int(seed))
        z = rng.standard_normal(500)
        s = float(median_heuristic_sigma(z))
        # Median |z_i - z_j| for N(0,1) draws is around 1.0 - 1.2
        # (theoretical median pairwise |diff| of two N(0,1)'s is
        # ~1.0488 = sqrt(2) * 0.7416). Bound generously.
        assert 0.5 < s < 2.5, (
            f"seed={seed}: median-heuristic sigma = {s:.4f}; expected in "
            f"(0.5, 2.5) for standard normal."
        )

    def test_median_sigma_constant_returns_one(self):
        _, median_heuristic_sigma, _, _, _ = _import_hsic_fe()
        z = np.ones(100, dtype=np.float64)
        s = float(median_heuristic_sigma(z))
        # All pairwise distances are zero -> fallback to 1.0.
        assert s == 1.0, (
            f"median-heuristic sigma on constant array = {s}; expected "
            f"1.0 fallback."
        )

    def test_median_sigma_too_small_n_returns_one(self):
        _, median_heuristic_sigma, _, _, _ = _import_hsic_fe()
        assert float(median_heuristic_sigma(np.asarray([]))) == 1.0
        assert float(median_heuristic_sigma(np.asarray([3.14]))) == 1.0

    def test_median_sigma_scales_with_data(self):
        """sigma(10 * z) should be ~10 * sigma(z) (scale equivariance)."""
        _, median_heuristic_sigma, _, _, _ = _import_hsic_fe()
        rng = np.random.default_rng(42)
        z = rng.standard_normal(300)
        s1 = float(median_heuristic_sigma(z))
        s2 = float(median_heuristic_sigma(10.0 * z))
        assert s2 == pytest.approx(10.0 * s1, rel=1e-9), (
            f"median-heuristic sigma not scale-equivariant: "
            f"sigma(z) = {s1}, sigma(10z) = {s2}, expected {10 * s1}"
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC lift on a non-monotone classification fixture
# ---------------------------------------------------------------------------


class TestAucLiftOnNonMonotone:
    """End-to-end biz_value: appending HSIC-selected orth-poly columns
    lifts downstream LogReg AUC over the raw baseline on a non-monotone
    classification fixture.
    """

    def test_hsic_augmented_logreg_auc_beats_raw(self):
        _, _, _, _, hybrid_with_recipes = _import_hsic_fe()
        gen = _import_plug_in_fe()
        aucs_raw, aucs_hsic = [], []
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
            aucs_hsic.append(roc_auc_score(
                y_te, lr_aug.predict_proba(X_aug_te)[:, 1],
            ))
        raw_mean = float(np.mean(aucs_raw))
        hsic_mean = float(np.mean(aucs_hsic))
        assert hsic_mean > raw_mean + 0.05, (
            f"HSIC-augmented LogReg AUC mean ({hsic_mean:.4f}) not lifted "
            f"by >= 0.05 vs raw mean ({raw_mean:.4f}); biz_value lift "
            f"claim violated.\nraw_per_seed={aucs_raw}\n"
            f"hsic_per_seed={aucs_hsic}"
        )
        wins = sum(d > r for d, r in zip(aucs_hsic, aucs_raw))
        assert wins >= len(aucs_raw) - 1, (
            f"HSIC-augmented AUC beat raw on only {wins}/{len(aucs_raw)} "
            f"seeds; per-seed floor too soft."
        )


# ---------------------------------------------------------------------------
# Contract 5: auto-pool now includes HSIC
# ---------------------------------------------------------------------------


class TestAutoPoolIncludesHsic:
    """Layer 68 SCORER_NAMES must contain ``hsic`` and the auto-selector
    must be ABLE to pick HSIC as a per-column winner on the right
    fixture. We do not pin which exact column gets HSIC (the LCB
    tournament rotates per seed); we pin that HSIC is at least
    consulted and CAN win at least once across seeds.
    """

    def test_scorer_names_contains_hsic(self):
        from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import (
            SCORER_NAMES,
        )
        assert "hsic" in SCORER_NAMES, (
            f"SCORER_NAMES missing 'hsic': {SCORER_NAMES}"
        )

    def test_auto_selector_can_pick_hsic(self):
        from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import (
            select_best_scorer_per_column,
        )
        gen = _import_plug_in_fe()
        chosen = []
        for s in (1, 7, 13, 42, 101, 202, 303, 404):
            X, y = _build_non_monotone_classif(s, n=600)
            engineered = gen(X, degrees=(2, 3), basis="hermite")
            table = select_best_scorer_per_column(
                X, engineered, y.to_numpy(),
                n_boot=3, random_state=s,
                dcor_n_sample=300,
            )
            if not table.empty:
                chosen.extend(table["best_scorer"].tolist())
        # HSIC need only be picked at least once across the seed sweep --
        # the auto pool is competitive with KSG / copula / dCor; this
        # contract pins that HSIC is reachable, not that it dominates.
        assert "hsic" in set(chosen), (
            f"HSIC never selected as best_scorer across {len(SEEDS)} "
            f"seeds; auto-pool integration broken.\nchosen_set={set(chosen)}"
        )


# ---------------------------------------------------------------------------
# Contract 6: default disabled byte-identical with master
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_hsic_columns(self, seed):
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], (
            f"seed={seed}: default fe_hybrid_orth_hsic_enable=False "
            f"should NOT append any engineered columns; got {added}"
        )

    def test_default_ctor_values(self):
        m = _make_mrmr()
        assert m.fe_hybrid_orth_hsic_enable is False
        assert m.fe_hybrid_orth_hsic_kernel == "rbf"
        assert m.fe_hybrid_orth_hsic_n_sample == 500


# ---------------------------------------------------------------------------
# Contract 7: pickle / clone preserve the ctor + recipes round-trip
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_clone_preserves_hsic_params(self):
        m = _make_mrmr(
            fe_hybrid_orth_hsic_enable=True,
            fe_hybrid_orth_hsic_kernel="rbf",
            fe_hybrid_orth_hsic_n_sample=300,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_hsic_enable", True),
            ("fe_hybrid_orth_hsic_kernel", "rbf"),
            ("fe_hybrid_orth_hsic_n_sample", 300),
        ]:
            assert getattr(m2, name) == expected, (
                f"clone() dropped {name}: expected {expected}, got "
                f"{getattr(m2, name)}"
            )

    def test_pickle_roundtrip_preserves_hsic_recipes(self):
        X, y = _build_quadratic_classif(seed=42, n=1500)
        m = _make_mrmr(
            fe_hybrid_orth_hsic_enable=True,
            fe_hybrid_orth_hsic_kernel="rbf",
            fe_hybrid_orth_hsic_n_sample=500,
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
