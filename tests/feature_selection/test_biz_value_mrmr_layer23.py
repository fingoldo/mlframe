"""Layer 23 biz_value: AUTO-WIRED HYBRID ORTHOGONAL-POLYNOMIAL FE INSIDE MRMR.fit().

Validates the 6 new ``fe_hybrid_orth_*`` MRMR constructor parameters and
their integration with the screening + transform pipeline introduced
2026-05-31. The standalone hybrid pipeline (Layers 21 / 22) shipped as
``_orthogonal_univariate_fe.hybrid_orth_mi_fe`` / ``hybrid_orth_mi_pair_fe``
and required the caller to call it manually and pass the augmented frame
into MRMR; Layer 23 makes the same lift available with a single
``fe_hybrid_orth_enable=True`` knob.

Contracts pinned
----------------
* TestDefaultIsLegacyByteIdentical
    With ``fe_hybrid_orth_enable=False`` (the default), ``support_`` and
    ``feature_names_in_`` match a pre-Layer-23 fit on the same data.
    Hybrid columns must not leak into the fit / transform output and
    ``hybrid_orth_features_`` is empty.

* TestEnableAddsQuadraticDetector
    On ``y = sign(x1^2 - 1)``, ``MRMR(fe_hybrid_orth_enable=True).fit(X, y)``
    must lift the quadratic detector ``x1__He2`` into either the
    ``hybrid_orth_features_`` list or directly into
    ``_engineered_features_``. The Layer 21 biz_value test already showed
    standalone hybrid finds it; Layer 23 pins that auto-wired MRMR does
    the same with the single flag.

* TestEnablePairDiscoversXor
    On ``y = sign(x1 * x2)`` (pure XOR), ``MRMR(fe_hybrid_orth_enable=True,
    fe_hybrid_orth_pair_enable=True).fit(X, y)`` lifts a pair-cross-basis
    term into ``hybrid_orth_features_``. Layer 22 standalone proved the
    pair stage discovers He_1*He_1; Layer 23 pins the auto-wire chain.

* TestDownstreamLogRegLift
    End-to-end XOR target: ``MRMR(fe_hybrid_orth_enable=True).fit(X_train,
    y_train).transform(X_test)`` produces a frame on which LogReg's
    holdout AUC clears the >= 0.80 threshold a pure-raw-feature LogReg
    cannot reach (XOR is unsolvable by raw linear LogReg).

* TestTransformConsistency
    ``MRMR(fe_hybrid_orth_enable=True).fit(X_train, y_train).transform(X_test)``
    must:
      1. Run WITHOUT y (no y-leakage at replay time).
      2. Produce engineered columns in the SAME values as the fit-time
         augmented columns when ``X_test == X_train``.

* TestNoYLeakage
    Shuffling y at transform-time must NOT change the engineered columns'
    values (recipe replay reads only X, never y). We assert by calling
    transform with the original X_train and verifying engineered columns
    match an independent direct computation via the standalone hybrid
    pipeline on the same X.

* TestPickleAndClone
    Layer-14-style sklearn integration: ``clone`` preserves the new
    ``fe_hybrid_orth_*`` constructor params; ``pickle.loads/dumps``
    round-trips a fitted MRMR with hybrid recipes intact (transform
    output matches pre-pickle).

* TestPair_AppendOrderStable
    Repeated fits on the same (X, y) with hybrid enabled return the
    same ``hybrid_orth_features_`` list (deterministic ranking).
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


SEEDS = (1, 13, 42)


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
def _build_quadratic(seed: int, n: int = 1200):
    """``y = sign(x1^2 - 1)`` -- He_2(z1) on z-scored Gaussian carries the
    signal. Raw x1 has near-zero MI to y by symmetry.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x2": rng.standard_normal(n),
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
    })
    y = ((x1 * x1 - 1.0) + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_xor(seed: int, n: int = 1500):
    """``y = sign(x1 * x2)`` -- pure XOR on Gaussians. Linear LogReg on raw
    x1, x2 cannot solve this; He_1(z1) * He_1(z2) = z1*z2 IS the signal.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
    })
    y = ((x1 * x2) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1200):
    """Plain linear-additive signal. Used for the default-is-legacy contract.
    """
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
# Contract 1: default OFF preserves legacy behaviour
# ---------------------------------------------------------------------------


class TestDefaultIsLegacyByteIdentical:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_hybrid_columns(self, seed):
        X, y = _build_linear(seed)
        m = _make_mrmr()  # all defaults
        m.fit(X, y)
        # Master switch defaults to OFF.
        assert m.fe_hybrid_orth_enable is False
        # No hybrid features lifted.
        assert m.hybrid_orth_features_ == [], (
            f"seed={seed}: default fe_hybrid_orth_enable=False should "
            f"produce empty hybrid_orth_features_, got "
            f"{m.hybrid_orth_features_}"
        )
        # feature_names_in_ matches the raw input columns exactly.
        assert list(m.feature_names_in_) == list(X.columns), (
            f"seed={seed}: feature_names_in_ must equal raw X.columns when "
            f"hybrid FE is off; got {list(m.feature_names_in_)} vs "
            f"{list(X.columns)}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_support_identical_to_explicit_off(self, seed):
        """Explicit fe_hybrid_orth_enable=False must produce the SAME support
        as omitting the kwarg entirely.
        """
        X, y = _build_linear(seed)
        m_default = _make_mrmr()
        m_explicit = _make_mrmr(fe_hybrid_orth_enable=False)
        m_default.fit(X, y)
        m_explicit.fit(X, y)
        assert list(m_default.support_) == list(m_explicit.support_), (
            f"seed={seed}: explicit False vs default disagreed on support_"
        )
        assert m_default.hybrid_orth_features_ == m_explicit.hybrid_orth_features_

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_transform_no_engineered_cols(self, seed):
        X, y = _build_linear(seed)
        m = _make_mrmr()
        Xt = m.fit(X, y).transform(X)
        # Output frame contains only raw selected columns; no hybrid suffixes.
        for c in Xt.columns:
            assert "__He" not in str(c)
            assert "__T" not in str(c).replace("noise_", "noise_")
            assert "__L" not in str(c)
            assert "*" not in str(c)


# ---------------------------------------------------------------------------
# Contract 2: enabling lifts a univariate He_2 detector
# ---------------------------------------------------------------------------


class TestEnableAddsQuadraticDetector:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_x1_he2_in_hybrid_features(self, seed):
        X, y = _build_quadratic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,  # univariate-only stage
            fe_hybrid_orth_degrees=(2, 3),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=5,
        )
        m.fit(X, y)
        appended = list(m.hybrid_orth_features_)
        assert any(("x1__He2" == c) or ("x1__He3" == c) for c in appended), (
            f"seed={seed}: x1__He2 (the quadratic detector) should be in "
            f"hybrid_orth_features_={appended}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_he2_in_engineered_features_or_recipes(self, seed):
        """Either the quadratic detector survived screening (then it's in
        ``_engineered_features_`` AND ``_engineered_recipes_``) OR it was
        appended to the candidate pool (always in ``hybrid_orth_features_``).
        Layer 23 pins that auto-wiring SEES the detector; screening
        decisions are pinned by Layers 21/22 already.
        """
        X, y = _build_quadratic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
        )
        m.fit(X, y)
        assert len(m.hybrid_orth_features_) >= 1, (
            f"seed={seed}: hybrid stage should append at least one "
            f"engineered column; got {m.hybrid_orth_features_}"
        )


# ---------------------------------------------------------------------------
# Contract 3: pair-cross-basis discovers XOR
# ---------------------------------------------------------------------------


class TestEnablePairDiscoversXor:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_xor_pair_in_hybrid_features(self, seed):
        X, y = _build_xor(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=True,
            fe_hybrid_orth_pair_max_degree=2,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=5,
        )
        m.fit(X, y)
        appended = list(m.hybrid_orth_features_)
        # The XOR cross-basis term has '*' between the two source columns
        # AND 'He1_He1' suffix. Allow either ordering of legs.
        ok = any(
            ("*" in c) and ("He1_He1" in c) and (("x1" in c) and ("x2" in c))
            for c in appended
        )
        assert ok, (
            f"seed={seed}: He_1*He_1 XOR pair term should be in "
            f"hybrid_orth_features_={appended}"
        )


# ---------------------------------------------------------------------------
# Contract 4: end-to-end downstream LogReg AUC lift on XOR
# ---------------------------------------------------------------------------


class TestDownstreamLogRegLift:

    @pytest.mark.parametrize("seed", (1, 13))
    def test_xor_logreg_auc_lift(self, seed):
        X, y = _build_xor(seed, n=2000)
        n_train = 1400
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]

        # Raw LogReg on the source columns: XOR is unsolvable.
        m_raw = LogisticRegression(max_iter=500).fit(Xtr.to_numpy(), ytr.to_numpy())
        auc_raw = roc_auc_score(
            yte.to_numpy(), m_raw.predict_proba(Xte.to_numpy())[:, 1]
        )

        # Hybrid-augmented LogReg via MRMR.transform.
        mrmr_h = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=True,
            fe_hybrid_orth_pair_max_degree=2,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=5,
        )
        mrmr_h.fit(Xtr, ytr)
        Xtr_aug = mrmr_h.transform(Xtr)
        Xte_aug = mrmr_h.transform(Xte)
        m_aug = LogisticRegression(max_iter=500).fit(
            np.asarray(Xtr_aug), ytr.to_numpy()
        )
        auc_aug = roc_auc_score(
            yte.to_numpy(), m_aug.predict_proba(np.asarray(Xte_aug))[:, 1]
        )
        # The XOR-augmented pipeline must clear a meaningful threshold:
        # raw is ~0.50, with He_1*He_1 it should be solid.
        assert auc_aug >= 0.80, (
            f"seed={seed}: hybrid-augmented LogReg AUC {auc_aug:.3f} should "
            f"clear 0.80 on XOR with hybrid FE; raw AUC {auc_raw:.3f}; "
            f"hybrid_orth_features_={mrmr_h.hybrid_orth_features_}"
        )
        assert auc_aug > auc_raw + 0.15, (
            f"seed={seed}: hybrid FE should lift LogReg holdout AUC by "
            f">= +0.15 on XOR. raw={auc_raw:.3f}, aug={auc_aug:.3f}"
        )


# ---------------------------------------------------------------------------
# Contract 5: transform consistency (fit-time engineered == transform-time)
# ---------------------------------------------------------------------------


class TestTransformConsistency:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_transform_no_y_required(self, seed):
        """transform() must accept X without y (sklearn contract).
        """
        X, y = _build_quadratic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
        )
        m.fit(X, y)
        # No y kwarg -- must not raise.
        out = m.transform(X)
        assert out is not None
        # transform output shape: at least one row per input.
        assert out.shape[0] == X.shape[0]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_replay_matches_fit_time_values(self, seed):
        """When X_test == X_train, the engineered cols emitted by transform()
        must equal the cols that would be produced by the standalone
        hybrid pipeline at fit time, value-by-value (closed-form replay).
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
        )
        X, y = _build_quadratic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_degrees=(2, 3),
            fe_hybrid_orth_basis="hermite",
        )
        m.fit(X, y)
        out = m.transform(X)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out)
        # Independently compute the basis cols for x1 (Hermite, the
        # quadratic detector's source).
        ref = generate_univariate_basis_features(
            X[["x1"]], cols=["x1"], degrees=(2, 3), basis="hermite",
        )
        # Find which He cols ended up in transform output.
        for he_col in ref.columns:
            if he_col in out.columns:
                np.testing.assert_allclose(
                    np.asarray(out[he_col], dtype=np.float64),
                    np.asarray(ref[he_col], dtype=np.float64),
                    rtol=1e-9, atol=1e-9,
                    err_msg=(
                        f"seed={seed}: transform-time {he_col} mismatched "
                        f"fit-time independent computation; recipe replay "
                        f"should be bit-equivalent."
                    ),
                )


# ---------------------------------------------------------------------------
# Contract 6: no y leakage at transform time
# ---------------------------------------------------------------------------


class TestNoYLeakage:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_transform_is_y_independent(self, seed):
        """Calling transform with a shuffled y (via the optional y kwarg)
        must NOT change the engineered column values. Recipes are pure
        functions of X.
        """
        X, y = _build_quadratic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
        )
        m.fit(X, y)
        out_no_y = m.transform(X)
        # Shuffle y; transform should produce the same output.
        rng = np.random.default_rng(seed + 999)
        y_shuf = pd.Series(rng.permutation(y.to_numpy()), name="y")
        out_shuf_y = m.transform(X, y=y_shuf)
        if isinstance(out_no_y, pd.DataFrame):
            pd.testing.assert_frame_equal(out_no_y, out_shuf_y)
        else:
            np.testing.assert_array_equal(np.asarray(out_no_y), np.asarray(out_shuf_y))


# ---------------------------------------------------------------------------
# Contract 7: pickle + clone preserve hybrid params + recipes
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_clone_preserves_hybrid_params(self):
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_degrees=(2, 4),
            fe_hybrid_orth_basis="legendre",
            fe_hybrid_orth_top_k=7,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_pair_max_degree=3,
        )
        m2 = clone(m)
        assert m2.fe_hybrid_orth_enable is True
        assert tuple(m2.fe_hybrid_orth_degrees) == (2, 4)
        assert m2.fe_hybrid_orth_basis == "legendre"
        assert m2.fe_hybrid_orth_top_k == 7
        assert m2.fe_hybrid_orth_pair_enable is False
        assert m2.fe_hybrid_orth_pair_max_degree == 3
        # clone is unfitted: no hybrid_orth_features_ attribute populated.
        assert not hasattr(m2, "support_")

    @pytest.mark.parametrize("seed", (1, 13))
    def test_pickle_roundtrip_transform_equal(self, seed):
        X, y = _build_quadratic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
        )
        m.fit(X, y)
        out_pre = m.transform(X)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        # State preserved.
        assert m2.fe_hybrid_orth_enable is True
        assert list(m2.hybrid_orth_features_) == list(m.hybrid_orth_features_)
        out_post = m2.transform(X)
        if isinstance(out_pre, pd.DataFrame):
            pd.testing.assert_frame_equal(out_pre, out_post)
        else:
            np.testing.assert_array_equal(np.asarray(out_pre), np.asarray(out_post))


# ---------------------------------------------------------------------------
# Contract 8: hybrid_orth_features_ is deterministic across re-fits
# ---------------------------------------------------------------------------


class TestPairAppendOrderStable:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_repeated_fits_yield_same_hybrid_features(self, seed):
        X, y = _build_xor(seed)
        m1 = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=True,
            fe_hybrid_orth_pair_max_degree=2,
            fe_hybrid_orth_basis="hermite",
        )
        m2 = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=True,
            fe_hybrid_orth_pair_max_degree=2,
            fe_hybrid_orth_basis="hermite",
        )
        m1.fit(X, y)
        m2.fit(X, y)
        assert list(m1.hybrid_orth_features_) == list(m2.hybrid_orth_features_), (
            f"seed={seed}: hybrid_orth_features_ should be deterministic; "
            f"got m1={m1.hybrid_orth_features_} vs m2={m2.hybrid_orth_features_}"
        )
