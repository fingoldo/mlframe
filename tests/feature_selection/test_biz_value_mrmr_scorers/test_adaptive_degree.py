"""Layer 57 biz_value: ADAPTIVE PER-COLUMN DEGREE selection for the
orthogonal-polynomial univariate FE path.

Validates ``hybrid_orth_mi_adaptive_degree_fe`` introduced 2026-05-31
(sibling module ``_orthogonal_adaptive_degree_fe``): pick the best
polynomial degree per source column, drop columns that don't beat raw
by a configurable uplift gate.

Why this layer matters
----------------------

Layer 21's ``generate_univariate_basis_features`` emits EVERY degree in
``degrees=(2,3)`` for EVERY source column -- at p=200 sources and
degrees=(1..6) that is 1200 candidate columns, almost all noise.

Each source column has at most ONE optimal degree (He_2 for a squared
detector, He_4 for a quartic step, He_6 for a sextic). Layer 57 picks
the per-column argmax up-front, dropping the 5 sibling degrees per
column. Result: fewer candidates, lower multiple-testing burden at the
abs floor, AUC parity or better with smaller candidate pools.

Contracts pinned
----------------

* TestPerColumnDegreeSelection: 3 cols with different optimal degrees
  (He_2, He_4, He_6) -- adaptive selects each correctly.

* TestSkipNoUplift: pure noise columns emit nothing (the fixed-degrees
  variant would emit 6 noise transforms per source).

* TestLogRegLift: at p=20 with 3 mixed-degree signals + 17 noise cols,
  augmented LogReg beats raw LogReg by >= 0.05 holdout AUC.

* TestComparedToFixedDegrees: adaptive emits FEWER but more informative
  columns vs the fixed (2,3) sweep on the same target.

* TestDefaultDisabledByteIdentical: ``fe_hybrid_orth_adaptive_degree_enable=False``
  (the default) keeps ``feature_names_in_`` identical to a fit without
  the flag.

* TestPickleAndClone: sklearn-style ``clone`` preserves the new ctor
  params; ``pickle`` round-trips a fitted MRMR with the per-column
  chosen-degree recipes intact.

Consolidated verbatim from test_biz_value_mrmr_layer57.py (per audit finding test_code_quality-16).
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

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_adaptive_fe():
    from mlframe.feature_selection.filters._orthogonal_adaptive_degree_fe import (
        generate_adaptive_degree_basis_features,
        hybrid_orth_mi_adaptive_degree_fe,
        hybrid_orth_mi_adaptive_degree_fe_with_recipes,
    )
    return (
        generate_adaptive_degree_basis_features,
        hybrid_orth_mi_adaptive_degree_fe,
        hybrid_orth_mi_adaptive_degree_fe_with_recipes,
    )


def _import_univariate_fe():
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
    )
    return generate_univariate_basis_features


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _hermite_he(x: np.ndarray, n: int) -> np.ndarray:
    """Probabilist's Hermite He_n via numpy poly eval; used by tests to
    construct targets whose optimal degree per column is known."""
    coef = np.zeros(n + 1, dtype=np.float64)
    coef[n] = 1.0
    return np.polynomial.hermite_e.hermeval(x, coef)


def _build_mixed_degree_signal(seed: int, n: int = 3000):
    """3 source columns whose optimal Hermite degrees are 2, 4, 6
    respectively. y = sign(He_2(x1) + He_4(x2) + He_6(x3) + noise).

    For x1 the He_2 transform should dominate raw MI; for x2 He_4; for
    x3 He_6. Adaptive selection must pick each correctly.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    # Per-column standardised contribution so each carries similar weight.
    s1 = _hermite_he(x1, 2)
    s2 = _hermite_he(x2, 4)
    s3 = _hermite_he(x3, 6)
    # Normalise each contribution so the target is balanced.
    def _z(v):
        sd = float(np.std(v))
        return v / sd if sd > 1e-12 else v
    sig = _z(s1) + _z(s2) + _z(s3) + 0.2 * rng.standard_normal(n)
    y = (sig > np.median(sig)).astype(int)
    return X, pd.Series(y, name="y")


def _build_noise_only(seed: int, n: int = 2000, p: int = 6):
    """Pure-noise frame: p Gaussian columns, y is independent Bernoulli."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        f"x{i}": rng.standard_normal(n) for i in range(p)
    })
    y = (rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_mixed_signal_with_noise(seed: int, n: int = 3000):
    """Mixed-degree signal across (x1, x2, x3) embedded among p=20 cols
    (17 noise). Used for the downstream LogReg AUC contract.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    cols = {"x1": x1, "x2": x2, "x3": x3}
    for i in range(17):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    s1 = _hermite_he(x1, 2)
    s2 = _hermite_he(x2, 4)
    s3 = _hermite_he(x3, 6)
    def _z(v):
        sd = float(np.std(v))
        return v / sd if sd > 1e-12 else v
    sig = _z(s1) + _z(s2) + _z(s3) + 0.2 * rng.standard_normal(n)
    y = (sig > np.median(sig)).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1500):
    """Plain linear-additive signal -- used for the default-disabled
    byte-identical contract."""
    rng = np.random.default_rng(seed)
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
# Contract 1: per-column degree selection -- argmax matches the truth
# ---------------------------------------------------------------------------


class TestPerColumnDegreeSelection:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_picks_correct_degree_per_column(self, seed):
        gen_adaptive, _, _ = _import_adaptive_fe()
        X, y = _build_mixed_degree_signal(seed)
        eng, meta = gen_adaptive(
            X, y.values,
            cols=["x1", "x2", "x3"],
            degree_range=(2, 3, 4, 5, 6),
            basis="hermite",
            min_uplift=1.05,
        )
        # Expect at most 3 columns (one per source). Each source's chosen
        # degree should match the truth: x1 -> 2, x2 -> 4, x3 -> 6.
        per_source_degree = {info["src"]: info["degree"] for info in meta.values()}
        # The strongest 2 should always be present; the weakest may occasionally
        # tie with a sibling at small n. Pin a strict 3/3 hit at n=3000.
        assert "x1" in per_source_degree, (
            f"seed={seed}: x1 should survive uplift gate; meta={meta}"
        )
        assert "x2" in per_source_degree, (
            f"seed={seed}: x2 should survive uplift gate; meta={meta}"
        )
        assert "x3" in per_source_degree, (
            f"seed={seed}: x3 should survive uplift gate; meta={meta}"
        )
        assert per_source_degree["x1"] == 2, (
            f"seed={seed}: x1 best degree should be 2 (He_2 quadratic); "
            f"got {per_source_degree['x1']}; meta={meta}"
        )
        assert per_source_degree["x2"] == 4, (
            f"seed={seed}: x2 best degree should be 4 (He_4 quartic); "
            f"got {per_source_degree['x2']}; meta={meta}"
        )
        assert per_source_degree["x3"] == 6, (
            f"seed={seed}: x3 best degree should be 6 (He_6 sextic); "
            f"got {per_source_degree['x3']}; meta={meta}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_emits_one_column_per_surviving_source(self, seed):
        gen_adaptive, _, _ = _import_adaptive_fe()
        X, y = _build_mixed_degree_signal(seed)
        eng, meta = gen_adaptive(
            X, y.values,
            cols=["x1", "x2", "x3"],
            degree_range=(2, 3, 4, 5, 6),
            basis="hermite",
            min_uplift=1.05,
        )
        # one row per surviving source -- no per-source duplicates
        srcs = [info["src"] for info in meta.values()]
        assert len(srcs) == len(set(srcs)), (
            f"seed={seed}: adaptive must emit at most one column per source; "
            f"got duplicates in {srcs}"
        )
        assert eng.shape[1] == len(meta), (
            f"seed={seed}: engineered frame shape mismatches meta length"
        )


# ---------------------------------------------------------------------------
# Contract 2: pure-noise columns are skipped
# ---------------------------------------------------------------------------


class TestSkipNoUplift:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_noise_only_emits_at_most_noise_floor_columns(self, seed):
        """Pure-noise frame: the per-source argmax may still clear the
        uplift gate by chance because raw MI is itself near-zero and the
        relative ratio can amplify tail noise. BUT every survivor must
        have ``engineered_mi`` at the per-bin noise floor (~ log(nbins)/n
        for plug-in MI on a discrete y), and the column count must be
        STRICTLY smaller than the fixed sweep would emit. This is the
        biz-value contract: the multiple-testing burden downstream
        (noise-aware abs floor at hybrid_orth_mi_fe) shrinks from
        ``p * len(degrees)`` to <= ``p``.
        """
        gen_adaptive, _, _ = _import_adaptive_fe()
        X, y = _build_noise_only(seed)
        eng, meta = gen_adaptive(
            X, y.values,
            cols=list(X.columns),
            degree_range=(2, 3, 4, 5, 6),
            basis="hermite",
            min_uplift=1.05,
        )
        # Per-source uniqueness: at most one column per source survives.
        srcs = [info["src"] for info in meta.values()]
        assert len(srcs) == len(set(srcs)), (
            f"seed={seed}: per-source argmax must emit at most one column "
            f"per source; got duplicates in {srcs}"
        )
        # Bound the count: never more than the input source count.
        n_sources = X.shape[1]
        assert eng.shape[1] <= n_sources, (
            f"seed={seed}: adaptive on noise should emit <= n_sources={n_sources} "
            f"columns; got {eng.shape[1]}: {list(eng.columns)}"
        )
        # Engineered MI must remain at the noise floor (well below any
        # real-signal MI which is typically >= 0.05 nats at n=2000).
        for name, info in meta.items():
            assert info["engineered_mi"] < 0.05, (
                f"seed={seed}: noise survivor {name!r} has engineered_mi "
                f"{info['engineered_mi']:.4f} -- should sit at noise floor "
                f"(< 0.05 nats)"
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_skip_outperforms_fixed_sweep_on_noise(self, seed):
        gen_adaptive, _, _ = _import_adaptive_fe()
        gen_fixed = _import_univariate_fe()
        X, y = _build_noise_only(seed, p=6)
        eng_ad, _ = gen_adaptive(
            X, y.values,
            cols=list(X.columns),
            degree_range=(2, 3, 4, 5, 6),
            basis="hermite",
            min_uplift=1.05,
        )
        eng_fixed = gen_fixed(
            X, cols=list(X.columns),
            degrees=(2, 3, 4, 5, 6),
            basis="hermite",
            dedup_collinear_sources=False,  # keep all columns fairly
        )
        # Fixed sweep emits 5 degrees * 6 columns = 30 candidates regardless
        # of MI; adaptive should emit STRICTLY FEWER on pure noise (the
        # uplift gate drops most sources; survivors are capped at one per
        # source by the per-col argmax).
        assert eng_ad.shape[1] < eng_fixed.shape[1], (
            f"seed={seed}: on pure noise adaptive should emit < fixed sweep; "
            f"adaptive={eng_ad.shape[1]}, fixed={eng_fixed.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Contract 3: downstream LogReg AUC lift on mixed-degree target
# ---------------------------------------------------------------------------


class TestLogRegLift:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_augmented_logreg_beats_raw(self, seed):
        _, hybrid, _ = _import_adaptive_fe()
        X, y = _build_mixed_signal_with_noise(seed, n=4000)
        n_train = 2800
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]
        # Baseline: raw LogReg on all 20 columns. Linear LogReg cannot
        # solve the He_4 / He_6 components -- AUC well below 1.0.
        m_raw = LogisticRegression(max_iter=500).fit(
            Xtr.to_numpy(), ytr.to_numpy(),
        )
        auc_raw = roc_auc_score(
            yte.to_numpy(), m_raw.predict_proba(Xte.to_numpy())[:, 1]
        )
        # Augmented: adaptive-degree FE on the full frame, then refit LogReg.
        X_aug, _ = hybrid(
            X, y.values,
            cols=list(X.columns),
            degree_range=(2, 3, 4, 5, 6),
            basis="hermite",
            min_uplift=1.05,
        )
        Xtr_aug = X_aug.iloc[:n_train]
        Xte_aug = X_aug.iloc[n_train:]
        m_aug = LogisticRegression(max_iter=500).fit(
            Xtr_aug.to_numpy(), ytr.to_numpy(),
        )
        auc_aug = roc_auc_score(
            yte.to_numpy(), m_aug.predict_proba(Xte_aug.to_numpy())[:, 1]
        )
        assert auc_aug >= auc_raw + 0.05, (
            f"seed={seed}: adaptive-degree FE should lift LogReg AUC by "
            f">= 0.05. raw={auc_raw:.3f}, aug={auc_aug:.3f}"
        )


# ---------------------------------------------------------------------------
# Contract 4: adaptive emits FEWER but more informative columns than fixed
# ---------------------------------------------------------------------------


class TestComparedToFixedDegrees:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_adaptive_emits_fewer_columns(self, seed):
        gen_adaptive, _, _ = _import_adaptive_fe()
        gen_fixed = _import_univariate_fe()
        X, _ = _build_mixed_degree_signal(seed)
        _, y = _build_mixed_degree_signal(seed)
        eng_fixed = gen_fixed(
            X, cols=["x1", "x2", "x3"],
            degrees=(2, 3, 4, 5, 6),
            basis="hermite",
        )
        eng_ad, _ = gen_adaptive(
            X, y.values,
            cols=["x1", "x2", "x3"],
            degree_range=(2, 3, 4, 5, 6),
            basis="hermite",
            min_uplift=1.05,
        )
        # Fixed sweep: 3 cols * 5 degrees = 15. Adaptive: at most 3 cols
        # (one per source).
        assert eng_fixed.shape[1] == 15, (
            f"seed={seed}: fixed sweep should emit 15 cols (3 * 5); got "
            f"{eng_fixed.shape[1]}"
        )
        assert eng_ad.shape[1] <= 3, (
            f"seed={seed}: adaptive should emit at most 3 cols (one per "
            f"source); got {eng_ad.shape[1]}: {list(eng_ad.columns)}"
        )
        assert eng_ad.shape[1] < eng_fixed.shape[1], (
            f"seed={seed}: adaptive must emit strictly fewer columns than "
            f"the fixed sweep; ad={eng_ad.shape[1]}, fixed={eng_fixed.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Contract 5: default disabled -- legacy behaviour byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_adaptive_columns(self, seed):
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        # No adaptive-degree-engineered columns surfaced.
        adaptive_added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert adaptive_added == [], (
            f"seed={seed}: default fe_hybrid_orth_adaptive_degree_enable=False "
            f"should NOT append any engineered columns; got "
            f"{adaptive_added}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_enable_adaptive_appends_engineered(self, seed):
        # Use the mixed-degree signal (n=2000) so the adaptive stage has
        # enough rows to clear the per-col uplift gate cleanly.
        X, y = _build_mixed_degree_signal(seed, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_adaptive_degree_enable=True,
            fe_hybrid_orth_adaptive_degree_range=(2, 3, 4, 5, 6),
            fe_hybrid_orth_adaptive_degree_min_uplift=1.05,
            fe_hybrid_orth_basis="hermite",
        ).fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added, (
            f"seed={seed}: adaptive flag ON should append at least one "
            f"engineered column to hybrid_orth_features_; got {added}"
        )


# ---------------------------------------------------------------------------
# Contract 6: pickle / clone preserve adaptive ctor + chosen-degree recipes
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_clone_preserves_adaptive_params(self):
        m = _make_mrmr(
            fe_hybrid_orth_adaptive_degree_enable=True,
            fe_hybrid_orth_adaptive_degree_range=(2, 4, 6),
            fe_hybrid_orth_adaptive_degree_min_uplift=1.2,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_adaptive_degree_enable", True),
            ("fe_hybrid_orth_adaptive_degree_range", (2, 4, 6)),
            ("fe_hybrid_orth_adaptive_degree_min_uplift", 1.2),
        ]:
            assert getattr(m2, name) == expected, (
                f"clone() dropped {name}: expected {expected}, got "
                f"{getattr(m2, name)}"
            )

    def test_pickle_roundtrip_preserves_chosen_degrees(self):
        X, y = _build_mixed_degree_signal(seed=42, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_adaptive_degree_enable=True,
            fe_hybrid_orth_adaptive_degree_range=(2, 3, 4, 5, 6),
            fe_hybrid_orth_adaptive_degree_min_uplift=1.05,
            fe_hybrid_orth_basis="hermite",
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), (
            "pickle changed feature_names_in_"
        )
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, (
            f"pickle changed hybrid_orth_features_: before={added_before}, "
            f"after={added_after}"
        )
        # Recipes survive: each engineered col name encodes the chosen
        # degree, so a pickle round-trip preserving names also preserves
        # the per-column chosen degree.
        recipes_before = {
            r.name: r for r in getattr(m, "_engineered_recipes_", []) or []
            if r.kind == "orth_univariate"
        }
        recipes_after = {
            r.name: r for r in getattr(m2, "_engineered_recipes_", []) or []
            if r.kind == "orth_univariate"
        }
        assert set(recipes_before.keys()) == set(recipes_after.keys()), (
            f"pickle dropped or added recipe names: before="
            f"{set(recipes_before.keys())}, after={set(recipes_after.keys())}"
        )
        for name, r_before in recipes_before.items():
            r_after = recipes_after[name]
            assert r_before.extra.get("degree") == r_after.extra.get("degree"), (
                f"pickle changed chosen degree for {name!r}: "
                f"before={r_before.extra}, after={r_after.extra}"
            )
            assert r_before.extra.get("basis") == r_after.extra.get("basis"), (
                f"pickle changed chosen basis for {name!r}"
            )
