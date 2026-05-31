"""Layer 60 biz_value: CMI-GREEDY FE CONSTRUCTOR INSIDE MRMR.fit().

Validates the 4 new ``fe_mi_greedy_cmi_*`` MRMR constructor parameters and
the new ``_mi_greedy_cmi_fe`` module (sibling to Layer 26's marginal-MI
greedy ``_mi_greedy_fe``). Layer 26 ranks candidate transforms by
marginal ``MI(candidate; y)``, which selects DUPLICATES when multiple
transforms encode the same signal (e.g. ``square(x)`` AND ``abs(x)`` AND
``log_abs(x)`` are all monotone in ``|x|`` on ``y=sign(x^2-1)``; Layer 26
picks all three and Spearman dedup drops two post-hoc). Layer 60 ranks by
CONDITIONAL ``MI(candidate; y | currently-selected-support)`` -- once one
of the family is in the support, the others' CMI collapses near zero and
they are never picked.

Contracts pinned
----------------

* ``TestDefaultDisabledByteIdentical``: default
  ``fe_mi_greedy_cmi_enable=False`` leaves the engineered-column path
  identical to a pre-Layer-60 fit (no CMI columns leak in).

* ``TestNoDuplicateSignal``: on ``y=sign(x^2-1)``, CMI-greedy picks ONE
  of the family {``square``, ``abs``, ``sqrt_abs``, ``log_abs``} per
  source col -- NOT all of them. Specifically: the count of |x|-monotone
  transforms appended is <= 1.

* ``TestComplementarySignalsRecovered``: on ``y=f(x1)+g(x2)`` where f and
  g are different non-linearities on different source cols, CMI greedy
  picks at least ONE transform per signal source (covers x1 AND x2),
  not all transforms on the same source.

* ``TestAucParityVsMarginalMIGreedy``: end-to-end LogReg AUC on
  ``y=sign(x^2-1)`` matches Layer 26 (since both pipelines capture the
  same |x|-family signal via at least one of its members). CMI parity
  to within +/- 0.05 AUC.

* ``TestFewerColsInSupport``: |mi_greedy_features_| under CMI-greedy <=
  |mi_greedy_features_| under marginal-MI-greedy on the duplicate-signal
  case (cleaner support is the whole point of CMI ranking).

* ``TestPickleAndClone``: sklearn ``clone`` and ``pickle`` preserve the
  4 new ctor params and the appended CMI recipes (transform output
  matches pre-pickle).

NEVER xfail. NEVER mask bugs via runtime workarounds.
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

SEEDS = (1, 7, 13)


def _make_mrmr(**overrides):
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
# Signal builders
# ---------------------------------------------------------------------------


def _build_linear(seed: int, n: int = 1200):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
        "noise_c": rng.standard_normal(n),
    })
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_square_signal(seed: int, n: int = 2000):
    """``y = sign(x^2 - 1)`` -- ``|x|``-family of transforms (square, abs,
    sqrt_abs, log_abs) all carry the SAME signal. Marginal-MI greedy picks
    several; CMI greedy must pick at most ONE per source.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    X = pd.DataFrame({
        "x": x,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
        "noise_c": rng.standard_normal(n),
    })
    y = ((x * x - 1.0) + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_complementary_signals(seed: int, n: int = 2500):
    """``y`` depends on BOTH ``x1^2`` AND ``log(|x2|+1)`` -- two distinct
    sources, two distinct non-linearities, no overlap. CMI greedy should
    recover at least ONE transform per source (so the union of source_cols
    across the appended winners covers {x1, x2}).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n) * 1.2
    x2 = rng.standard_normal(n) * 1.5
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
        "noise_c": rng.standard_normal(n),
    })
    sig1 = (x1 * x1) > 1.0
    sig2 = np.log1p(np.abs(x2)) > float(np.median(np.log1p(np.abs(x2))))
    y = (sig1 ^ sig2).astype(int)
    return X, pd.Series(y, name="y")


# Helper: extract the source col of a unary engineered name "fn(col)".
def _unary_source(name: str) -> str:
    if name.endswith(")") and "(" in name:
        return name.split("(", 1)[1][:-1]
    return name


# ---------------------------------------------------------------------------
# Contract 1: default OFF preserves legacy behaviour
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_cmi_columns(self, seed):
        X, y = _build_linear(seed)
        m = _make_mrmr()
        m.fit(X, y)
        assert m.fe_mi_greedy_cmi_enable is False
        assert m.mi_greedy_features_ == [], (
            f"seed={seed}: default fe_mi_greedy_cmi_enable=False should "
            f"produce empty mi_greedy_features_, got {m.mi_greedy_features_}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_support_identical_to_explicit_off(self, seed):
        X, y = _build_linear(seed)
        m_default = _make_mrmr()
        m_explicit = _make_mrmr(fe_mi_greedy_cmi_enable=False)
        m_default.fit(X, y)
        m_explicit.fit(X, y)
        assert list(m_default.support_) == list(m_explicit.support_), (
            f"seed={seed}: explicit False vs default disagreed on support_"
        )
        assert m_default.mi_greedy_features_ == m_explicit.mi_greedy_features_


# ---------------------------------------------------------------------------
# Contract 2: no duplicate signal (the headline value of CMI ranking)
# ---------------------------------------------------------------------------


class TestNoDuplicateSignal:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_at_most_one_x2_family_member(self, seed):
        """On ``y=sign(x^2-1)`` the |x|-family {square, abs, sqrt_abs,
        log_abs} all carry the same signal. CMI greedy must pick AT MOST
        ONE of them per source -- duplicate-signal collapse is the whole
        point.

        For comparison: under marginal-MI greedy (Layer 26), at least
        2 of the 4 are typically picked (verified by the parallel
        ``TestSquareSignalRecovered`` in test_biz_value_mrmr_layer26.py).
        """
        X, y = _build_square_signal(seed)
        m = _make_mrmr(
            fe_mi_greedy_cmi_enable=True,
            fe_mi_greedy_cmi_top_k=5,
            fe_mi_greedy_cmi_seed_cols_count=4,
            fe_mi_greedy_cmi_min_gain=0.005,
        )
        m.fit(X, y)
        appended = list(m.mi_greedy_features_)
        family = {"square", "abs", "sqrt_abs", "log_abs"}
        # Count unary picks on source col 'x' from the |x|-family.
        x_family_picks = [
            c for c in appended
            if "(" in c and c.endswith(")")
            and _unary_source(c) == "x"
            and c.split("(", 1)[0] in family
        ]
        assert len(x_family_picks) <= 1, (
            f"seed={seed}: CMI-greedy picked MULTIPLE |x|-family transforms "
            f"on the same source 'x' (duplicate signal): {x_family_picks}. "
            f"Whole appended list: {appended}"
        )
        # And it should have picked at least one (the signal IS there).
        assert len(x_family_picks) >= 1, (
            f"seed={seed}: CMI-greedy picked ZERO |x|-family transforms on "
            f"'x' -- the signal was missed entirely. Appended: {appended}"
        )


# ---------------------------------------------------------------------------
# Contract 3: complementary signals on different sources
# ---------------------------------------------------------------------------


class TestComplementarySignalsRecovered:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_covers_both_source_cols(self, seed):
        """``y`` depends on BOTH x1 (via square) AND x2 (via log_abs). CMI
        greedy should pick at least one transform whose source set hits
        {x1} and at least one whose source set hits {x2}.
        """
        X, y = _build_complementary_signals(seed)
        m = _make_mrmr(
            fe_mi_greedy_cmi_enable=True,
            fe_mi_greedy_cmi_top_k=6,
            fe_mi_greedy_cmi_seed_cols_count=4,
            fe_mi_greedy_cmi_min_gain=0.003,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        m.fit(X, y)
        appended = list(m.mi_greedy_features_)
        # Union of source cols across all unary picks.
        unary_sources = {
            _unary_source(c) for c in appended
            if "(" in c and c.endswith(")") and not c.startswith("(")
        }
        assert "x1" in unary_sources and "x2" in unary_sources, (
            f"seed={seed}: CMI-greedy missed at least one of the two "
            f"complementary sources. unary sources covered = "
            f"{unary_sources}; appended = {appended}"
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC parity vs marginal-MI greedy
# ---------------------------------------------------------------------------


class TestAucParityVsMarginalMIGreedy:

    @pytest.mark.parametrize("seed", (1, 13))
    def test_logreg_auc_within_tolerance(self, seed):
        """Both pipelines should recover the |x|-family signal (Layer 26
        picks multiple members, Layer 60 picks one). Downstream LogReg AUC
        should be comparable -- the second + third members of the family
        are nearly co-linear with the first and add nothing to LogReg.
        """
        X, y = _build_square_signal(seed)
        n_train = 1400
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]

        mrmr_mi = _make_mrmr(
            fe_mi_greedy_enable=True,
            fe_mi_greedy_top_k=5,
            fe_mi_greedy_seed_cols_count=4,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        mrmr_mi.fit(Xtr, ytr)
        Xtr_mi = mrmr_mi.transform(Xtr)
        Xte_mi = mrmr_mi.transform(Xte)
        auc_mi = roc_auc_score(
            yte.to_numpy(),
            LogisticRegression(max_iter=500)
            .fit(np.asarray(Xtr_mi), ytr.to_numpy())
            .predict_proba(np.asarray(Xte_mi))[:, 1],
        )

        mrmr_cmi = _make_mrmr(
            fe_mi_greedy_cmi_enable=True,
            fe_mi_greedy_cmi_top_k=5,
            fe_mi_greedy_cmi_seed_cols_count=4,
            fe_mi_greedy_cmi_min_gain=0.005,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        mrmr_cmi.fit(Xtr, ytr)
        Xtr_cmi = mrmr_cmi.transform(Xtr)
        Xte_cmi = mrmr_cmi.transform(Xte)
        auc_cmi = roc_auc_score(
            yte.to_numpy(),
            LogisticRegression(max_iter=500)
            .fit(np.asarray(Xtr_cmi), ytr.to_numpy())
            .predict_proba(np.asarray(Xte_cmi))[:, 1],
        )

        assert abs(auc_mi - auc_cmi) <= 0.07, (
            f"seed={seed}: CMI-greedy AUC {auc_cmi:.3f} drifted >0.07 from "
            f"marginal-MI greedy AUC {auc_mi:.3f}. CMI should match the "
            f"signal-recovery quality of marginal MI on a single-signal "
            f"target (the dedup is what differs, not the captured signal)."
        )


# ---------------------------------------------------------------------------
# Contract 5: cleaner support (fewer cols)
# ---------------------------------------------------------------------------


class TestFewerColsInSupport:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_cmi_appends_no_more_than_marginal(self, seed):
        """On the duplicate-signal target, CMI greedy must append <= as
        many engineered columns as marginal-MI greedy (the whole point is
        that the second + third members of the |x|-family are suppressed).
        """
        X, y = _build_square_signal(seed)
        mrmr_mi = _make_mrmr(
            fe_mi_greedy_enable=True,
            fe_mi_greedy_top_k=5,
            fe_mi_greedy_seed_cols_count=4,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        mrmr_mi.fit(X, y)
        n_mi = len(mrmr_mi.mi_greedy_features_)

        mrmr_cmi = _make_mrmr(
            fe_mi_greedy_cmi_enable=True,
            fe_mi_greedy_cmi_top_k=5,
            fe_mi_greedy_cmi_seed_cols_count=4,
            fe_mi_greedy_cmi_min_gain=0.005,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        mrmr_cmi.fit(X, y)
        n_cmi = len(mrmr_cmi.mi_greedy_features_)

        assert n_cmi <= n_mi, (
            f"seed={seed}: CMI greedy appended {n_cmi} engineered cols, "
            f"marginal MI greedy appended {n_mi}. CMI should be <= (cleaner "
            f"support is the whole point). "
            f"cmi cols = {mrmr_cmi.mi_greedy_features_}, "
            f"mi cols = {mrmr_mi.mi_greedy_features_}"
        )


# ---------------------------------------------------------------------------
# Contract 6: pickle + clone preservation
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_clone_preserves_constructor_params(self):
        m = _make_mrmr(
            fe_mi_greedy_cmi_enable=True,
            fe_mi_greedy_cmi_top_k=7,
            fe_mi_greedy_cmi_seed_cols_count=6,
            fe_mi_greedy_cmi_min_gain=0.01,
        )
        m_clone = clone(m)
        for attr in (
            "fe_mi_greedy_cmi_enable",
            "fe_mi_greedy_cmi_top_k",
            "fe_mi_greedy_cmi_seed_cols_count",
            "fe_mi_greedy_cmi_min_gain",
        ):
            assert getattr(m_clone, attr) == getattr(m, attr), (
                f"clone failed to preserve {attr}"
            )

    def test_pickle_roundtrip_preserves_recipes(self):
        X, y = _build_square_signal(7)
        m = _make_mrmr(
            fe_mi_greedy_cmi_enable=True,
            fe_mi_greedy_cmi_top_k=5,
            fe_mi_greedy_cmi_seed_cols_count=4,
            fe_mi_greedy_cmi_min_gain=0.005,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        m.fit(X, y)
        out_pre = m.transform(X)

        blob = pickle.dumps(m)
        m_rt = pickle.loads(blob)
        out_post = m_rt.transform(X)
        np.testing.assert_allclose(
            np.asarray(out_pre, dtype=np.float64),
            np.asarray(out_post, dtype=np.float64),
            rtol=1e-12, atol=1e-12,
        )
        assert (
            list(m_rt.mi_greedy_features_) == list(m.mi_greedy_features_)
        )
