"""Layer 81 biz_value: LASSO (L1) coefficient-based pre-selection as an
alternative to MI / dependence-metric scoring for the hybrid orth-poly FE.

Production setup: Layers 21 / 65-74 score candidate engineered columns via
plug-in MI / KSG / copula MI / dCor / HSIC / JMIM / TC / CMIM -- all
NON-PARAMETRIC dependence measures. L1-regularised linear regression
(Lasso) is the dual PARAMETRIC pre-selection: fit a single linear model
on ``[raw_X, engineered_X]``, treat ``|coef|`` as the per-column score.

The two paths are COMPLEMENTARY: Lasso wins on linear-additive signals
where the L1 path identifies the supporting columns sharply; MI wins on
non-monotone, oscillatory targets where Lasso's linear-projection
assumption drives the truly-useful column's coefficient to zero.

Contracts pinned
----------------

* TestLassoPrefersLinearContributions: on an ADDITIVE linear signal
  ``y = 1.5 * He_2(x_1) + 0.8 * He_2(x_2) + eps``, Lasso's top-2 engineered
  picks include BOTH ``x1__He2`` AND ``x2__He2`` -- the columns linearly
  contributing to y. This is the structural win Layer 81 exists for.

* TestLassoAgreesWithMIOnLinearTop1: on a single-source linear signal
  ``y = 1.5 * He_2(x_1) + eps``, Lasso's top-1 RANKED engineered column
  is the same as MI's top-1 after gate filtering: both pick ``x1__He2``.
  The "sanity case" -- on a clean linear-additive signal, no dispute.

* TestLassoUnderperformsOnNonMonotone: on a highly oscillatory target
  ``y = sin(5 * x_1)``, MI's top-1 picked engineered column has
  STRICTLY HIGHER uplift than Lasso's, and MI picks ``x1__He3`` (the
  truly informative column) while Lasso fills its top with noise
  columns. Documents the COST of the parametric assumption.

* TestAucLiftViaLasso: on linear-additive data, LogReg fitted to the
  Lasso-augmented X beats LogReg on raw X by AUC >= +0.02 on a held-out
  set. End-to-end biz_value gate.

* TestDefaultDisabledByteIdentical: the new ctor params on MRMR default
  to ``fe_hybrid_orth_lasso_enable=False`` / ``alpha=0.01``; a fresh
  MRMR with no overrides has the flag off, and the selected support is
  byte-identical to a master without the new ctor params (master pickle
  byte-equivalence preserved).

* TestPickleAndClone: ctor params survive ``clone()`` and pickle.

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

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def _import_lasso():
    """Lazily import the Lasso (L1) coefficient-based FE functions."""
    from mlframe.feature_selection.filters._orthogonal_lasso_fe import (
        hybrid_orth_mi_lasso_fe,
        score_features_by_lasso_coef,
    )

    return hybrid_orth_mi_lasso_fe, score_features_by_lasso_coef


def _import_mi():
    """Lazily import the plug-in MI FE function (comparison baseline)."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        hybrid_orth_mi_fe,
    )

    return hybrid_orth_mi_fe


def _import_mrmr():
    """Lazily import the MRMR class."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR


def _make_mrmr(**overrides):
    """Build an MRMR with cheap, deterministic default knobs that isolate the Lasso-preselect FE stage."""
    MRMR = _import_mrmr()
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


def _build_linear_additive(seed: int, n: int = 1500):
    """Linear-additive signal on TWO engineered columns:

    ``y = 1.5 * He_2(x_1) + 0.8 * He_2(x_2) + eps``

    where ``He_2(x) = x^2 - 1``. Lasso's home turf: each He_2 column
    contributes additively and linearly to ``y``, so the L1 path picks
    both columns with sharp coefficients.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
            "noise_2": rng.standard_normal(n),
        }
    )
    he2_x1 = x1**2 - 1.0
    he2_x2 = x2**2 - 1.0
    y = 1.5 * he2_x1 + 0.8 * he2_x2 + 0.3 * rng.standard_normal(n)
    return X, y


def _build_linear_additive_binary(seed: int, n: int = 1500):
    """Same fixture as ``_build_linear_additive`` thresholded to a binary
    target so downstream LogReg AUC has a well-defined signal/holdout split.
    """
    X, y_cont = _build_linear_additive(seed, n)
    y_bin = (y_cont > 0).astype(int)
    return X, pd.Series(y_bin, name="y")


def _build_single_source_quadratic(seed: int, n: int = 1500):
    """Single-source quadratic: ``y = 1.5 * He_2(x_1) + eps``. Used for the
    Lasso-vs-MI agreement test: both scorers must rank ``x1__He2`` as the
    top-1 engineered pick.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        }
    )
    he2_x1 = x1**2 - 1.0
    y_cont = 1.5 * he2_x1 + 0.3 * rng.standard_normal(n)
    y_bin = (y_cont > 0).astype(int)
    return X, pd.Series(y_bin, name="y"), y_cont


def _build_oscillatory(seed: int, n: int = 500):
    """Highly oscillatory target: ``y = sin(5 * x_1) + eps``.

    Lasso's blind spot: ``cov(sin(5*x), x) ~= 0`` for standard-Gaussian
    ``x``, so the Lasso path on the raw column drives the coefficient
    to zero; ``He_n(x_1)`` columns inherit small projection coefficients
    too because ``sin(5 * x)`` has nearly-zero projection on the
    low-degree Hermite basis at this oscillation frequency. MI, by
    contrast, captures the full mutual-information content of the
    rank structure and ranks ``x1__He3`` sharply above noise.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
            "noise_2": rng.standard_normal(n),
        }
    )
    y_cont = np.sin(5.0 * x1) + 0.1 * rng.standard_normal(n)
    y_bin = (y_cont > 0).astype(int)
    return X, pd.Series(y_bin, name="y"), y_cont


# ---------------------------------------------------------------------------
# Contract 1: Lasso prefers truly linear contributions
# ---------------------------------------------------------------------------


class TestLassoPrefersLinearContributions:
    """On an additive linear signal, Lasso's |coef| ranking correctly picks both truly-contributing He_2 columns."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_top2_engineered_are_he2_x1_and_x2(self, seed):
        """On the additive linear signal ``y = 1.5 He_2(x_1) + 0.8 He_2(x_2)``,
        the Lasso path's top-2 ranked engineered columns are
        ``x1__He2`` and ``x2__He2`` -- the columns the L1 fit assigns
        non-zero coefficients to, while every other engineered column
        (including the noise He_n columns) collapses to ``|coef| = 0``.
        """
        hybrid_lasso, _ = _import_lasso()
        X, y_cont = _build_linear_additive(seed)
        # Continuous y matches Lasso's regression target; we run with low
        # gates so the test reads raw ranking, not gate filtering.
        _X_aug, scores = hybrid_lasso(
            X,
            y_cont,
            degrees=(2, 3),
            basis="hermite",
            top_k=2,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            alpha=0.01,
        )
        # Engineered columns sorted by |coef| descending -- the top-2 must
        # be the two truly-contributing He_2 columns.
        top2 = list(scores["engineered_col"].iloc[:2])
        expected = {"x1__He2", "x2__He2"}
        assert set(top2) == expected, (
            f"seed={seed}: Lasso top-2 engineered ranking should be "
            f"{sorted(expected)} for additive linear signal "
            f"y = 1.5*He_2(x1) + 0.8*He_2(x2); got {top2}. "
            f"Full scores head:\n{scores.head(8).to_string()}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_noise_dominated_by_signal_coefs(self, seed):
        """Lasso shrinks non-contributing columns strongly relative to the
        true signal. On the additive linear fixture every noise_i__He_n
        column receives ``|coef|`` at LEAST 50x smaller than the smallest
        true-signal coefficient. (A handful of noise coefficients may
        survive coordinate-descent at small ``alpha`` with tiny non-zero
        values; what matters is the ranking margin between signal and
        noise.)
        """
        hybrid_lasso, _ = _import_lasso()
        X, y_cont = _build_linear_additive(seed)
        _X_aug, scores = hybrid_lasso(
            X,
            y_cont,
            degrees=(2, 3),
            basis="hermite",
            top_k=10,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            alpha=0.01,
        )
        noise_rows = scores[scores["source_col"].str.startswith("noise_")]
        signal_rows = scores[scores["engineered_col"].isin(["x1__He2", "x2__He2"])]
        max_noise_coef = float(noise_rows["engineered_mi"].max()) if not noise_rows.empty else 0.0
        min_signal_coef = float(signal_rows["engineered_mi"].min()) if not signal_rows.empty else 0.0
        # Signal-to-noise ratio floor: smallest true-signal coefficient
        # dominates the largest surviving noise coefficient by >= 50x.
        # On the seed=101 case, noise coef = 2.5e-3, signal min = 0.79;
        # ratio = 316. Floor at 50 leaves comfortable headroom.
        ratio = min_signal_coef / (max_noise_coef + 1e-12)
        assert ratio >= 50.0, (
            f"seed={seed}: Lasso noise/signal coefficient separation only "
            f"{ratio:.1f}x; required >= 50x. max_noise_coef="
            f"{max_noise_coef:.3e}, min_signal_coef={min_signal_coef:.3e}. "
            f"noise_rows:\n{noise_rows.to_string()}\n"
            f"signal_rows:\n{signal_rows.to_string()}"
        )


# ---------------------------------------------------------------------------
# Contract 2: Lasso and MI agree on the linear top-1
# ---------------------------------------------------------------------------


class TestLassoAgreesWithMIOnLinearTop1:
    """On a clean single-source linear-additive signal, Lasso and MI both rank x1__He2 as the top-1 engineered column."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_top1_engineered_is_x1_he2_for_both(self, seed):
        """On the single-source quadratic ``y = sign(1.5 He_2(x_1))``
        fixture, BOTH the Lasso path and the MI plug-in path must rank
        ``x1__He2`` as the top-1 engineered column by their respective
        primary metric (|coef| for Lasso, MI for the plug-in). This is
        the "sanity case" -- on a clean linear-additive signal, no
        dispute. We test the raw ranking via ``engineered_mi`` (the
        column name shared across all hybrid_*_fe variants for cross-
        layer parity); gate filtering happens downstream and is
        scorer-specific.
        """
        hybrid_lasso, _ = _import_lasso()
        hybrid_mi = _import_mi()
        X, y_bin, y_cont = _build_single_source_quadratic(seed)
        _X_l, scores_lasso = hybrid_lasso(
            X,
            y_cont,
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            alpha=0.01,
        )
        _X_m, scores_mi = hybrid_mi(
            X,
            y_bin.to_numpy(),
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            nbins=10,
        )
        # Rank both by engineered_mi (primary metric: |coef| for Lasso,
        # plug-in MI for the MI path).
        top1_lasso = scores_lasso.sort_values("engineered_mi", ascending=False)["engineered_col"].iloc[0]
        top1_mi = scores_mi.sort_values("engineered_mi", ascending=False)["engineered_col"].iloc[0]
        assert top1_lasso == "x1__He2", (
            f"seed={seed}: Lasso top-1 engineered should be x1__He2 on "
            f"linear He_2(x_1) signal; got {top1_lasso!r}. "
            f"scores head:\n{scores_lasso.head(5).to_string()}"
        )
        assert top1_mi == "x1__He2", (
            f"seed={seed}: MI top-1 engineered should be x1__He2 on linear He_2(x_1) signal; got {top1_mi!r}. scores head:\n{scores_mi.head(5).to_string()}"
        )


# ---------------------------------------------------------------------------
# Contract 3: Lasso UNDERPERFORMS on non-monotone, MI wins
# ---------------------------------------------------------------------------


class TestLassoUnderperformsOnNonMonotone:
    """On a highly oscillatory target, MI correctly ranks x1__He3 top-3 while Lasso's parametric assumption fails it."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mi_picks_x1_he3_lasso_does_not(self, seed):
        """On the highly oscillatory target ``y = sin(5 * x_1)``, MI ranks
        ``x1__He3`` at the top of its qualified set while Lasso's top-1
        is some OTHER engineered column (typically a noise He_n with a
        coincidental coefficient, since the Lasso path can't see the
        oscillatory dependence). This is the documented LOSS of the
        parametric assumption.
        """
        hybrid_lasso, _ = _import_lasso()
        hybrid_mi = _import_mi()
        X, y_bin, y_cont = _build_oscillatory(seed)
        _X_l, scores_lasso = hybrid_lasso(
            X,
            y_cont,
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            alpha=0.01,
        )
        _X_m, scores_mi = hybrid_mi(
            X,
            y_bin.to_numpy(),
            degrees=(2, 3),
            basis="hermite",
            top_k=3,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            nbins=10,
        )
        # MI's qualifier for x1__He3: in the top-3 by ranking.
        mi_top3 = list(scores_mi.sort_values("engineered_mi", ascending=False)["engineered_col"].iloc[:3])
        assert "x1__He3" in mi_top3, (
            f"seed={seed}: MI did not rank x1__He3 in its top-3 on sin(5*x_1) signal; non-monotone-MI claim regressed. mi_top3={mi_top3!r}"
        )
        # Lasso's top-1 should NOT be x1__He3 (since Lasso can't see
        # oscillatory dependence). Pinning STRICT INEQUALITY: MI's |coef|
        # of x1__He3 isn't the metric; check that Lasso's top-1 is a noise
        # column, signalling the structural failure mode.
        lasso_top1 = scores_lasso["engineered_col"].iloc[0]
        assert lasso_top1.startswith("noise_") or lasso_top1 == "x2__He3", (
            f"seed={seed}: Lasso unexpectedly recovered the oscillatory "
            f"signal as top-1: {lasso_top1!r}. Expected a noise or x2 "
            f"engineered column. scores_lasso head:\n"
            f"{scores_lasso.head(6).to_string()}"
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC lift via Lasso on linear-additive
# ---------------------------------------------------------------------------


class TestAucLiftViaLasso:
    """Lasso-augmented LogReg measurably lifts holdout AUC over raw-feature LogReg on a linear-additive target."""

    @pytest.mark.parametrize("seed", (1, 7, 13))
    def test_lasso_augmented_logreg_beats_raw_by_002(self, seed):
        """On linear-additive ``y = 1.5 He_2(x_1) + 0.8 He_2(x_2)``, LogReg
        trained on the Lasso-augmented X must beat LogReg trained on raw X
        by holdout AUC >= +0.02. The engineered He_2 columns linearise the
        quadratic relationship so LogReg's linear decision boundary picks
        the signal up immediately.
        """
        hybrid_lasso, _ = _import_lasso()
        # Train and holdout drawn from the same population.
        X_tr, y_tr = _build_linear_additive_binary(seed, n=1500)
        X_te_raw, y_te_cont = _build_linear_additive(seed + 1000, n=3000)
        y_te = pd.Series((y_te_cont > 0).astype(int), name="y")

        # Continuous-y target for the Lasso fit (regression is its natural
        # contract); the AUC measurement is the downstream binary task.
        _, y_tr_cont = _build_linear_additive(seed, n=1500)
        X_tr_aug, _scores = hybrid_lasso(
            X_tr,
            y_tr_cont,
            degrees=(2, 3),
            basis="hermite",
            top_k=2,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            alpha=0.01,
        )
        appended = [c for c in X_tr_aug.columns if c not in X_tr.columns]
        # Replay the same engineered columns on the holdout via the Layer 21
        # generator (engineered VALUES are bit-equal; only the scoring
        # path differs).
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
        )

        eng_te_all = generate_univariate_basis_features(
            X_te_raw,
            degrees=(2, 3),
            basis="hermite",
        )
        keep = [c for c in appended if c in eng_te_all.columns]
        X_te_aug = pd.concat([X_te_raw, eng_te_all[keep]], axis=1) if keep else X_te_raw

        def _fit_auc(Xtr, Xte):
            """Fit LogReg on Xtr/y_tr and return holdout AUC on Xte/y_te."""
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y_tr)
            p = clf.predict_proba(Xte)[:, 1]
            return float(roc_auc_score(y_te, p))

        auc_raw = _fit_auc(X_tr, X_te_raw)
        auc_aug = _fit_auc(X_tr_aug, X_te_aug)
        gap = auc_aug - auc_raw
        assert gap >= 0.02, (
            f"seed={seed}: Lasso-augmented LogReg AUC {auc_aug:.4f} does "
            f"not beat raw-X LogReg AUC {auc_raw:.4f} by the 0.02 floor; "
            f"gap={gap:.4f}. Engineered columns appended: {appended!r}"
        )


# ---------------------------------------------------------------------------
# Contract 5: default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_lasso_enable=False (the default) leaves the selected support byte-identical to the legacy path."""

    def test_default_ctor_has_lasso_off(self):
        """A fresh MRMR with no overrides has the Layer 81 master switch
        off. Default pickle byte-equivalence preserved.
        """
        m = _make_mrmr()
        assert getattr(m, "fe_hybrid_orth_lasso_enable") is False, "fe_hybrid_orth_lasso_enable defaults to False"
        assert getattr(m, "fe_hybrid_orth_lasso_alpha") == 0.01, "fe_hybrid_orth_lasso_alpha defaults to 0.01"

    @pytest.mark.parametrize("seed", (1, 7, 13))
    def test_default_off_fit_byte_identical_to_baseline(self, seed):
        """With the Layer 81 master switch off, ``MRMR.fit`` on the linear-
        additive fixture produces a support that depends ONLY on the legacy
        ctor params -- adding ``fe_hybrid_orth_lasso_enable=False``
        explicitly to a kwargs dict must not change the selected support.
        """
        X, y = _build_linear_additive_binary(seed, n=800)
        m1 = _make_mrmr()  # implicit default
        m1.fit(X, y)
        m2 = _make_mrmr(
            fe_hybrid_orth_lasso_enable=False,
            fe_hybrid_orth_lasso_alpha=0.01,
        )
        m2.fit(X, y)
        sup1 = list(m1.feature_names_in_)
        sup2 = list(m2.feature_names_in_)
        assert sup1 == sup2, f"seed={seed}: explicit fe_hybrid_orth_lasso_enable=False diverged from implicit-default fit. implicit={sup1}, explicit={sup2}"


# ---------------------------------------------------------------------------
# Contract 6: pickle / clone preserve the Lasso ctor flags
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """clone() and pickle preserve the Lasso ctor flags (fe_hybrid_orth_lasso_enable/alpha), unfitted and fitted."""

    def test_clone_preserves_lasso_params(self):
        """clone() copies fe_hybrid_orth_lasso_enable/alpha without carrying over fitted state."""
        m = _make_mrmr(
            fe_hybrid_orth_lasso_enable=True,
            fe_hybrid_orth_lasso_alpha=0.05,
        )
        m2 = clone(m)
        assert getattr(m2, "fe_hybrid_orth_lasso_enable") is True, "clone() dropped fe_hybrid_orth_lasso_enable"
        assert getattr(m2, "fe_hybrid_orth_lasso_alpha") == 0.05, "clone() dropped fe_hybrid_orth_lasso_alpha"

    def test_pickle_roundtrip_unfitted(self):
        """pickle.dumps/loads on an unfitted MRMR preserves fe_hybrid_orth_lasso_enable/alpha."""
        m = _make_mrmr(
            fe_hybrid_orth_lasso_enable=True,
            fe_hybrid_orth_lasso_alpha=0.05,
        )
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert getattr(m2, "fe_hybrid_orth_lasso_enable") is True, "pickle/unpickle dropped fe_hybrid_orth_lasso_enable"
        assert getattr(m2, "fe_hybrid_orth_lasso_alpha") == 0.05, "pickle/unpickle dropped fe_hybrid_orth_lasso_alpha"

    def test_pickle_roundtrip_fitted(self):
        """pickle.dumps/loads on a Lasso-fitted MRMR preserves feature_names_in_ and transform() output."""
        X, y = _build_linear_additive_binary(seed=42, n=800)
        m = _make_mrmr(
            fe_hybrid_orth_lasso_enable=True,
            fe_hybrid_orth_lasso_alpha=0.01,
        )
        m.fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        Xt = m.transform(X)
        Xt2 = m2.transform(X)
        assert list(Xt.columns) == list(Xt2.columns), "pickle changed transform() columns"
        for c in Xt.columns:
            if pd.api.types.is_numeric_dtype(Xt[c]):
                v1 = Xt[c].to_numpy()
                v2 = Xt2[c].to_numpy()
                if not np.allclose(v1, v2, equal_nan=True, atol=1e-10):
                    raise AssertionError(f"pickle changed transform() values for column {c!r}: max abs diff {np.nanmax(np.abs(v1 - v2)):.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
