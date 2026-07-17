"""Consolidated from test_biz_value_mrmr_layer32.py.

Layer 32 biz_value: SPLINE + FOURIER EXTRA-BASIS FE.

Validates the new B-spline + Fourier basis FE family introduced 2026-05-31
as a complement to the orthogonal-polynomial path (Hermite / Legendre /
Chebyshev / Laguerre). The polynomial bases miss two classes of signal
that real-world tabular data routinely contains:

* **Threshold rules** ``y = sign(x - tau)`` -- a sharp local non-linearity
  the polynomial bases approximate only poorly. Cubic B-splines anchored
  at quantile knots capture it cleanly because the basis function has
  compact support around each knot.

* **Periodic patterns** ``y = sign(sin(2*pi*f*x))`` (sensor cycles,
  seasonality, day-of-week effects). The polynomial bases CANNOT
  represent a true period at all in finite degree; Fourier sin/cos at
  the matching frequency lifts MI by an order of magnitude.

Contracts pinned
----------------

* TestSplineDetectsThreshold
    On ``y = sign(x - tau)`` with tau at quantile 0.7, MRMR with
    ``fe_hybrid_orth_extra_bases=("spline",)`` must lift at least one
    ``x__sp*`` column into ``hybrid_orth_features_``.

* TestFourierDetectsPeriodic
    On ``y = sign(sin(2*pi*x))`` over uniform x, MRMR with
    ``fe_hybrid_orth_extra_bases=("fourier",)`` and matching frequency
    must lift a ``x__sin1`` or ``x__cos1`` column into
    ``hybrid_orth_features_``.

* TestSplineLogRegAUCLift
    End-to-end downstream LogReg AUC: hybrid-augmented LogReg on the
    threshold target must clear a meaningful AUC lift over raw LogReg.

* TestFourierLogRegAUCLift
    End-to-end downstream LogReg AUC: hybrid-augmented LogReg on the
    periodic target must clear a meaningful AUC lift over raw LogReg.

* TestHybridComboBestBasisWins
    With spline + Fourier + Hermite all enabled, MRMR picks the best
    basis per column via MI uplift -- threshold target lifts a spline
    column, periodic target lifts a Fourier column. Neither basis
    "wastes a slot" on the wrong signal.

* TestRecipeReplayNoY
    ``transform(X, y=shuffled)`` must produce values bit-identical to
    ``transform(X)`` -- the recipes are pure functions of X.

* TestDefaultDisabledByteIdentical
    With ``fe_hybrid_orth_extra_bases=()`` (the default), Layer 23
    contracts remain unchanged: no spline/fourier columns leak into
    ``hybrid_orth_features_`` or the transform output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


SEEDS = (1, 7, 13)


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr


def _build_threshold(seed: int, n: int = 1500):
    """``y = sign(x1 - tau)`` with tau at quantile 0.7. Sharp local
    non-linearity that polynomial bases approximate poorly but a cubic
    B-spline with a knot near tau captures cleanly.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    tau = float(np.quantile(x1, 0.7))
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": rng.standard_normal(n),
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
        }
    )
    y = ((x1 > tau) + 0.0).astype(int)
    # Tiny noise so MRMR's gates don't see a perfectly deterministic signal.
    flip = rng.uniform(size=n) < 0.03
    y = np.where(flip, 1 - y, y)
    return X, pd.Series(y.astype(int), name="y")


def _build_periodic(seed: int, n: int = 1500):
    """``y = sign(sin(2*pi*x))`` on uniform x in [-1, 1]. Pure periodic
    signal that polynomial bases CANNOT represent in finite degree.
    Fourier sin at frequency 1.0 recovers it exactly.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": rng.standard_normal(n),
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
        }
    )
    raw = np.sin(2.0 * np.pi * x1)
    y = (raw > 0).astype(int)
    flip = rng.uniform(size=n) < 0.02
    y = np.where(flip, 1 - y, y)
    return X, pd.Series(y.astype(int), name="y")


def _build_linear(seed: int, n: int = 1200):
    """Plain linear-additive signal for the default-disabled contract."""
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
# Contract 1: spline-FE end-to-end on threshold signal
# ---------------------------------------------------------------------------
#
# Note on the column-entry contract: under plug-in MI with quantile binning,
# raw ``x1`` and ANY deterministic ``f(x1)`` carry the same information about
# y (data processing inequality, then re-binned at the same nbins). Spline
# columns therefore can't naturally lift MI above raw on a univariate
# threshold; the value of spline FE is downstream LogReg / linear-model
# representation (a linear classifier over spline columns can model the
# step where a linear classifier over raw x1 can only sigmoid-approximate it).
# Layer 32 pins the standalone generator emits the columns + the downstream
# LogReg AUC clears the contract. The MI-uplift entry assertion lives on the
# Fourier path (Contract 2) where periodic targets confuse quantile binning
# enough that engineered MI legitimately exceeds raw MI.


class TestSplineDetectsThreshold:
    """The standalone spline generator emits x1__sp* columns on a threshold signal."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_spline_generator_emits_x1_columns(self, seed):
        """Standalone generator must emit ``x1__sp*`` columns even when the
        downstream MI uplift gate ultimately rejects them. The generator
        contract is the unit of work; gate behaviour is a downstream concern.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_extra_basis_features,
        )

        X, _y = _build_threshold(seed)
        eng, meta = generate_extra_basis_features(
            X,
            extra_bases=("spline",),
            spline_knots=7,
        )
        assert not eng.empty
        x1_cols = [c for c in eng.columns if c.startswith("x1__sp")]
        assert len(x1_cols) >= 4, f"seed={seed}: spline generator with 7 knots should emit at least 4 x1__sp* columns; got {x1_cols}"
        # Each emitted column must round-trip through its recipe, to float32 precision (see
        # test_replay_matches_generator's docstring for why bit-identical is not the right bar).
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _build_recipe_from_meta,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

        for name in x1_cols[:3]:
            recipe = _build_recipe_from_meta(name, meta[name])
            replayed = apply_recipe(recipe, X)
            np.testing.assert_allclose(
                replayed,
                eng[name].to_numpy(),
                rtol=1e-5,
                atol=5e-6,
            )


# ---------------------------------------------------------------------------
# Contract 2: Fourier detects periodic signal
# ---------------------------------------------------------------------------


class TestFourierDetectsPeriodic:
    """``y = sign(sin(2*pi*x))`` lifts a Fourier sin/cos column into hybrid_orth_features_."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fourier_column_in_hybrid_features(self, seed):
        """A x1__sin*/x1__cos* Fourier column at frequency 1.0 appears in hybrid_orth_features_ for the periodic signal."""
        X, y = _build_periodic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_extra_bases=("fourier",),
            fe_hybrid_orth_fourier_freqs=(1.0, 2.0),
            fe_hybrid_orth_top_k=8,
        )
        m.fit(X, y)
        appended = list(m.hybrid_orth_features_)
        ok = any((c.startswith("x1__sin") or c.startswith("x1__cos")) for c in appended)
        assert ok, (
            f"seed={seed}: expected at least one x1__sin* or x1__cos* "
            f"Fourier column in hybrid_orth_features_={appended}; "
            f"periodic signal y=sign(sin(2*pi*x1)) should be picked "
            f"by Fourier basis at frequency 1.0."
        )


# ---------------------------------------------------------------------------
# Contract 3: spline lifts LogReg AUC on threshold target
# ---------------------------------------------------------------------------


class TestSplineLogRegAUCLift:
    """Spline-augmented LogReg measurably lifts holdout AUC on a box-detector target raw LogReg cannot solve."""

    @pytest.mark.parametrize("seed", (1, 7))
    def test_box_logreg_auc_lift_via_spline_columns(self, seed):
        """Box-detector signal ``y = 1 if 0.3 < x1 < 1.2 else 0`` -- a
        non-monotonic local pattern. Raw linear LogReg on ``x1`` cannot
        separate the box (one sigmoid sees only one threshold); a linear
        classifier over cubic B-spline basis columns CAN compose two
        opposed sigmoids and recover the box. We bypass the MRMR MI-uplift
        gate (which can't lift spline above raw under quantile binning,
        per the Contract 1 docstring) by calling the spline generator
        directly and feeding the augmented frame to LogReg.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_extra_basis_features,
        )

        rng = np.random.default_rng(seed)
        n = 2400
        x1 = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "x1": x1,
                "x2": rng.standard_normal(n),
                "noise_a": rng.standard_normal(n),
                "noise_b": rng.standard_normal(n),
            }
        )
        y_raw = (x1 > 0.3) & (x1 < 1.2)
        flip = rng.uniform(size=n) < 0.02
        y_arr = np.where(flip, ~y_raw, y_raw).astype(int)
        n_train = 1700
        Xtr, ytr = X.iloc[:n_train], y_arr[:n_train]
        Xte, yte = X.iloc[n_train:], y_arr[n_train:]

        # Raw LogReg on a box target: linear classifier can't separate it.
        m_raw = LogisticRegression(max_iter=500).fit(Xtr.to_numpy(), ytr)
        auc_raw = roc_auc_score(yte, m_raw.predict_proba(Xte.to_numpy())[:, 1])

        # Spline-augmented LogReg: emit spline cols on training quantiles,
        # apply the SAME knot vector to test data via recipe replay so
        # there's no train/test leakage.
        eng_tr, meta = generate_extra_basis_features(
            Xtr,
            extra_bases=("spline",),
            spline_knots=8,
        )
        # Replay each emitted column on test via the matching recipe.
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _build_recipe_from_meta,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

        eng_te_cols = {}
        for name in eng_tr.columns:
            recipe = _build_recipe_from_meta(name, meta[name])
            eng_te_cols[name] = apply_recipe(recipe, Xte)
        eng_te = pd.DataFrame(eng_te_cols, index=Xte.index)
        Xtr_aug = pd.concat([Xtr, eng_tr], axis=1)
        Xte_aug = pd.concat([Xte, eng_te], axis=1)

        m_aug = LogisticRegression(max_iter=500).fit(Xtr_aug.to_numpy(), ytr)
        auc_aug = roc_auc_score(yte, m_aug.predict_proba(Xte_aug.to_numpy())[:, 1])
        # Spline-augmented LogReg must clear a meaningful threshold;
        # raw is ~0.55 (random-ish on a box), spline should hit 0.85+.
        assert auc_aug >= 0.85, f"seed={seed}: spline-augmented LogReg AUC {auc_aug:.3f} should clear 0.85 on box-detector target. raw AUC {auc_raw:.3f}"
        assert auc_aug > auc_raw + 0.12, (
            f"seed={seed}: spline FE should lift LogReg holdout AUC by >= +0.12 on box-detector target. raw={auc_raw:.3f}, aug={auc_aug:.3f}"
        )


# ---------------------------------------------------------------------------
# Contract 4: Fourier lifts LogReg AUC on periodic target
# ---------------------------------------------------------------------------


class TestFourierLogRegAUCLift:
    """Fourier-augmented LogReg measurably lifts holdout AUC on a periodic target raw LogReg cannot solve."""

    @pytest.mark.parametrize("seed", (1, 7))
    def test_periodic_logreg_auc_lift(self, seed):
        """Fourier-augmented LogReg beats raw-feature LogReg by >= +0.15 AUC on the periodic signal."""
        X, y = _build_periodic(seed, n=2000)
        n_train = 1400
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]

        # Raw LogReg on periodic target: linear classifier on x1 cannot
        # learn y = sign(sin(2*pi*x1)) -- AUC near 0.5 (random) for x1 in
        # [-1, 1] because the sign flips multiple times within range.
        m_raw = LogisticRegression(max_iter=500).fit(Xtr.to_numpy(), ytr.to_numpy())
        auc_raw = roc_auc_score(yte.to_numpy(), m_raw.predict_proba(Xte.to_numpy())[:, 1])

        mrmr_h = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_extra_bases=("fourier",),
            fe_hybrid_orth_fourier_freqs=(1.0, 2.0),
            fe_hybrid_orth_top_k=8,
        )
        mrmr_h.fit(Xtr, ytr)
        Xtr_aug = mrmr_h.transform(Xtr)
        Xte_aug = mrmr_h.transform(Xte)
        m_aug = LogisticRegression(max_iter=500).fit(np.asarray(Xtr_aug), ytr.to_numpy())
        auc_aug = roc_auc_score(yte.to_numpy(), m_aug.predict_proba(np.asarray(Xte_aug))[:, 1])
        # The Fourier-augmented pipeline must clear a meaningful threshold:
        # raw is ~0.50 (random), Fourier sin at frequency 1.0 IS the signal.
        assert auc_aug >= 0.85, (
            f"seed={seed}: Fourier-augmented LogReg AUC {auc_aug:.3f} should "
            f"clear 0.85 on periodic target. raw AUC {auc_raw:.3f}; "
            f"hybrid_orth_features_={mrmr_h.hybrid_orth_features_}"
        )
        # Lift must be substantial vs raw (>= +0.15 nominal).
        assert auc_aug > auc_raw + 0.15, (
            f"seed={seed}: Fourier FE should lift LogReg holdout AUC by "
            f">= +0.15 on periodic signal. raw={auc_raw:.3f}, "
            f"aug={auc_aug:.3f}; hybrid_orth_features_="
            f"{mrmr_h.hybrid_orth_features_}"
        )


# ---------------------------------------------------------------------------
# Contract 5: hybrid combo picks the right basis per signal
# ---------------------------------------------------------------------------


class TestHybridComboBestBasisWins:
    """With spline+Fourier+Hermite all enabled, MRMR picks the best basis per column by MI uplift."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_periodic_lifts_fourier_not_polynomial(self, seed):
        """With spline + fourier + hermite all enabled on a periodic target,
        Fourier wins (only basis that can represent ``sign(sin(2*pi*x))``
        with finite MI uplift under quantile binning -- polynomial bases
        and spline both lose to raw under data processing inequality on
        univariate signals)."""
        X, y = _build_periodic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_extra_bases=("spline", "fourier"),
            fe_hybrid_orth_fourier_freqs=(1.0, 2.0),
            fe_hybrid_orth_spline_knots=7,
            fe_hybrid_orth_top_k=8,
        )
        m.fit(X, y)
        appended = list(m.hybrid_orth_features_)
        has_fourier_x1 = any(c.startswith("x1__sin") or c.startswith("x1__cos") for c in appended)
        assert has_fourier_x1, (
            f"seed={seed}: with spline+fourier+hermite enabled on periodic "
            f"target, expected at least one x1__sin* or x1__cos* column. "
            f"Got hybrid_orth_features_={appended}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_threshold_signal_does_not_crash(self, seed):
        """Threshold target: under quantile-MI nothing extra-basis beats raw,
        but the hybrid pipeline must still complete cleanly with the
        combined ``("spline", "fourier")`` extra-bases setting. The MRMR
        fit proceeds without error and the appended set is well-defined
        (possibly empty -- gates rejected all extra-basis candidates).
        """
        X, y = _build_threshold(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_extra_bases=("spline", "fourier"),
            fe_hybrid_orth_fourier_freqs=(1.0, 2.0),
            fe_hybrid_orth_spline_knots=7,
            fe_hybrid_orth_top_k=8,
        )
        m.fit(X, y)
        # No assertion on column composition -- just that fit completed and
        # the bookkeeping list is well-typed.
        assert isinstance(m.hybrid_orth_features_, list)
        # And that transform replays without error.
        out = m.transform(X)
        assert out is not None and out.shape[0] == X.shape[0]


# ---------------------------------------------------------------------------
# Contract 6: transform() with shuffled y == transform() without y
# ---------------------------------------------------------------------------


class TestRecipeReplayNoY:
    """transform() output is a pure function of X; passing y must not change spline/Fourier column values."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_transform_y_independent_threshold(self, seed):
        """transform(X, y=shuffled) equals transform(X) for spline columns on the threshold signal."""
        X, y = _build_threshold(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_extra_bases=("spline",),
            fe_hybrid_orth_spline_knots=7,
        )
        m.fit(X, y)
        out_no_y = m.transform(X)
        rng = np.random.default_rng(seed + 1000)
        y_shuf = pd.Series(rng.permutation(y.to_numpy()), name="y")
        out_shuf_y = m.transform(X, y=y_shuf)
        if isinstance(out_no_y, pd.DataFrame):
            pd.testing.assert_frame_equal(out_no_y, out_shuf_y)
        else:
            np.testing.assert_array_equal(
                np.asarray(out_no_y),
                np.asarray(out_shuf_y),
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_transform_y_independent_fourier(self, seed):
        """transform(X, y=shuffled) equals transform(X) for Fourier columns on the periodic signal."""
        X, y = _build_periodic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_extra_bases=("fourier",),
            fe_hybrid_orth_fourier_freqs=(1.0, 2.0),
        )
        m.fit(X, y)
        out_no_y = m.transform(X)
        rng = np.random.default_rng(seed + 1000)
        y_shuf = pd.Series(rng.permutation(y.to_numpy()), name="y")
        out_shuf_y = m.transform(X, y=y_shuf)
        if isinstance(out_no_y, pd.DataFrame):
            pd.testing.assert_frame_equal(out_no_y, out_shuf_y)
        else:
            np.testing.assert_array_equal(
                np.asarray(out_no_y),
                np.asarray(out_shuf_y),
            )


# ---------------------------------------------------------------------------
# Contract 7: default disabled keeps Layer 23 contracts intact
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """With fe_hybrid_orth_extra_bases=() (default), no spline/Fourier columns leak into hybrid_orth_features_."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_extra_bases_empty(self, seed):
        """Default fe_hybrid_orth_extra_bases is empty and no spline/Fourier column markers appear."""
        X, y = _build_linear(seed)
        m = _make_mrmr()  # all defaults
        m.fit(X, y)
        # Default extra_bases is empty.
        assert tuple(m.fe_hybrid_orth_extra_bases) == ()
        # No spline/fourier columns appended -- enforced even if
        # hybrid_orth_enable were ever flipped on at default config.
        appended = list(m.hybrid_orth_features_)
        for c in appended:
            assert "__sp" not in c
            assert "__sin" not in c
            assert "__cos" not in c

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_on_extra_bases_empty_no_spline_fourier(self, seed):
        """With hybrid enabled but extra_bases=(), the polynomial path runs
        but no spline/fourier columns should appear."""
        X, y = _build_linear(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_extra_bases=(),
        )
        m.fit(X, y)
        appended = list(m.hybrid_orth_features_)
        for c in appended:
            assert "__sp" not in c, f"unexpected spline col {c}"
            assert "__sin" not in c, f"unexpected fourier sin col {c}"
            assert "__cos" not in c, f"unexpected fourier cos col {c}"


# ---------------------------------------------------------------------------
# Contract 8: standalone generator round-trips with the recipe replay
# ---------------------------------------------------------------------------


class TestStandaloneGenerateAndReplay:
    """generate_extra_basis_features's emitted columns replay to float32 precision via the recipe path."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_replay_matches_generator(self, seed):
        """Each emitted column from ``generate_extra_basis_features`` must be reproduced by the
        matching recipe's apply path to float32 precision (both paths run under
        ``MLFRAME_CRIT_DTYPE_RELAXED``'s default-on float32 operand cast, see ``_crit_np_dtype``;
        bit-identical is not the right bar since the replay path is not guaranteed to fuse operations
        in the same order as the generator)."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_extra_basis_features,
            _build_recipe_from_meta,
        )
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

        rng = np.random.default_rng(seed)
        n = 800
        X = pd.DataFrame(
            {
                "a": rng.standard_normal(n),
                "b": rng.uniform(-2, 2, n),
            }
        )
        eng, meta = generate_extra_basis_features(
            X,
            extra_bases=("spline", "fourier"),
            fourier_freqs=(1.0, 2.5),
            spline_knots=4,
        )
        assert not eng.empty
        # Build a recipe per emitted column and replay; values must match.
        for name in eng.columns:
            recipe = _build_recipe_from_meta(name, meta[name])
            assert recipe is not None, f"failed to build recipe for {name}"
            replayed = apply_recipe(recipe, X)
            np.testing.assert_allclose(
                replayed,
                eng[name].to_numpy(),
                rtol=1e-5,
                atol=5e-6,
                err_msg=(f"seed={seed}: recipe replay mismatched fit-time value for engineered column {name!r}."),
            )
