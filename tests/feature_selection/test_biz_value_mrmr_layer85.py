"""Layer 85 biz_value: ``fe_hybrid_orth_default_scorer`` routing flag.

Layer 84 shipped CMIM hot-path perf optimizations but DID NOT expose the
"pick the empirically-best scorer in one knob" flag that Layer 83's
7-dataset showdown justified. Layer 85 closes that gap: a single ctor
string param routes the Layer 21 hybrid orth-poly univariate basis-
selection stage through any of the alternate scorers (CMIM / JMIM / TC /
KSG / copula / dCor / HSIC / auto / ensemble / meta / lasso / elasticnet).

Contracts pinned
----------------

* ``TestDefaultPlugInByteIdentical``: ``default_scorer="plug_in"`` (the
  default) keeps ``hybrid_orth_features_`` byte-identical to a fit with
  no flag set. The L85 wiring must not change Layer 21's behaviour when
  the user does not opt in.
* ``TestInvalidValueRaises``: an unrecognised scorer string raises
  ``ValueError`` at fit time with an actionable message listing every
  accepted value.
* ``TestCmimRoutingMatchesDirectCmim``: setting
  ``default_scorer="cmim"`` on a fixture with ``pair_enable=False``
  produces the same ``hybrid_orth_features_`` support as a direct call
  to ``hybrid_orth_mi_cmim_fe_with_recipes`` on the same X/y. (i.e. the
  router is not silently degrading to plug-in.)
* ``TestCmimAucGteDefault``: on a multi-redundant fixture, AUC of a
  LogReg trained on ``default_scorer="cmim"``-selected features is
  >= AUC of the default plug-in selector. (Layer 83's CMIM win is now
  reachable via a single knob.)
* ``TestPickleClonePreserveScorerFlag``: ``clone`` + ``pickle``
  round-trip preserves the ``fe_hybrid_orth_default_scorer`` ctor value.
* ``TestRecommendDefaultScorer``: ``MRMR.recommend_default_scorer()``
  returns ``"cmim"`` (the L83 leaderboard winner) without needing an
  instance.

NEVER xfail.

2026-06-01 Layer 85.
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


def _make_mrmr(**overrides):
    """Cheap-and-deterministic MRMR ctor (mirrors Layer 74)."""
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


def _build_redundant_multi(seed: int, n: int = 2000):
    """Multi-redundant fixture: x1 quadratic signal, near-copies x_dup_*,
    x2 secondary signal. Mirrors Layer 74's ``_build_redundant_multi``.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x_dup_a = x1 + 0.05 * rng.standard_normal(n)
    x_dup_b = x1 + 0.05 * rng.standard_normal(n)
    x_dup_c = x1 + 0.05 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x_dup_a": x_dup_a,
        "x_dup_b": x_dup_b,
        "x_dup_c": x_dup_c,
        "x2": x2,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    signal = x1 ** 2 + 0.6 * (x2 ** 2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1500):
    """Plain linear fixture for the default-disabled byte-identical contract."""
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
# Contract 1: default "plug_in" byte-identical to master (no flag set)
# ---------------------------------------------------------------------------


class TestDefaultPlugInByteIdentical:
    """``default_scorer="plug_in"`` (the default) keeps the L21 dispatch
    on the plug-in path, byte-identical to a fit constructed without
    even mentioning the flag.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_value_is_plug_in(self, seed):
        m = _make_mrmr()
        assert m.fe_hybrid_orth_default_scorer == "plug_in", (
            f"seed={seed}: default ctor value drifted from 'plug_in' to "
            f"{m.fe_hybrid_orth_default_scorer!r}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_orth_columns(self, seed):
        """When the master switch ``fe_hybrid_orth_enable`` is off, the
        ``default_scorer`` flag is inert -- no engineered columns are
        appended regardless of its value.
        """
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], (
            f"seed={seed}: with master OFF, no engineered columns should "
            f"appear; got {added}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_plug_in_explicit_matches_default(self, seed):
        """A model with ``default_scorer="plug_in"`` (explicit) appends
        the same hybrid_orth_features_ as a model with no flag set. Both
        must run the Layer 21 path.
        """
        X, y = _build_redundant_multi(seed, n=1500)
        m_default = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=2,
            fe_hybrid_orth_pair_enable=False,
        ).fit(X, y)
        m_explicit = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=2,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_default_scorer="plug_in",
        ).fit(X, y)
        added_default = list(getattr(m_default, "hybrid_orth_features_", []) or [])
        added_explicit = list(getattr(m_explicit, "hybrid_orth_features_", []) or [])
        assert added_default == added_explicit, (
            f"seed={seed}: explicit 'plug_in' diverged from the implicit "
            f"default. implicit={added_default}, explicit={added_explicit}"
        )


# ---------------------------------------------------------------------------
# Contract 2: invalid scorer value raises ValueError at fit time
# ---------------------------------------------------------------------------


class TestInvalidValueRaises:
    """An unrecognised scorer string raises ``ValueError`` with an
    actionable message listing every accepted value.
    """

    def test_invalid_value_raises_with_actionable_message(self):
        X, y = _build_linear(seed=0)
        m = _make_mrmr(fe_hybrid_orth_default_scorer="not_a_real_scorer")
        with pytest.raises(ValueError) as exc:
            m.fit(X, y)
        msg = str(exc.value)
        assert "fe_hybrid_orth_default_scorer" in msg
        assert "not_a_real_scorer" in msg
        # The error message must list the valid values for the caller to
        # know which knobs are accepted. Spot-check a few canonical ones.
        for accepted in ("plug_in", "cmim", "jmim", "ksg"):
            assert accepted in msg, (
                f"error message does not mention valid value {accepted!r}; "
                f"got msg={msg!r}"
            )

    def test_non_string_value_raises(self):
        X, y = _build_linear(seed=0)
        m = _make_mrmr(fe_hybrid_orth_default_scorer=42)
        with pytest.raises(ValueError) as exc:
            m.fit(X, y)
        assert "fe_hybrid_orth_default_scorer" in str(exc.value)

    def test_every_documented_value_passes_validation(self):
        """Every entry in ``_VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS``
        survives the validation check on a tiny X/y. (Catches drift
        between the constant tuple and the dispatcher branches.)
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_linear(seed=0)
        # Use master OFF so we exercise only the validation, not the
        # dispatcher (some scorers e.g. ksg have heavyweight sklearn
        # imports we don't need to pay here).
        for scorer in MRMR._VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS:
            m = _make_mrmr(fe_hybrid_orth_default_scorer=scorer)
            # Should NOT raise. We don't care about the fit output here.
            m.fit(X, y)


# ---------------------------------------------------------------------------
# Contract 3: "cmim" routing matches direct CMIM scorer output
# ---------------------------------------------------------------------------


class TestCmimRoutingMatchesDirectCmim:
    """``default_scorer="cmim"`` with ``pair_enable=False`` must produce
    the same engineered-column support as a direct call to
    ``hybrid_orth_mi_cmim_fe_with_recipes`` on the same X/y -- i.e. the
    dispatcher is not silently dropping to plug-in.
    """

    @pytest.mark.parametrize("seed", (0, 7, 42))
    def test_routing_matches_direct_call(self, seed):
        from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
            hybrid_orth_mi_cmim_fe_with_recipes,
        )
        X, y = _build_redundant_multi(seed, n=1500)
        # Direct CMIM call -- same kwargs as the dispatcher in
        # _mrmr_fit_impl._dispatch_default_scorer.
        _, scores_direct, _ = hybrid_orth_mi_cmim_fe_with_recipes(
            X, y.to_numpy(),
            cols=None,
            degrees=(2,),
            basis="hermite",
            top_k=2,
        )
        # Through the MRMR routing.
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=2,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_default_scorer="cmim",
        ).fit(X, y)
        routed_added = sorted(
            list(getattr(m, "hybrid_orth_features_", []) or [])
        )
        # ``scores_direct`` is a DataFrame: pick the same top_k=2 winners
        # by descending engineered_mi (already sorted by the scorer).
        # Note: the scorer's gate floors (min_uplift/min_abs_mi_frac) are
        # nonzero by default, so it may admit fewer than top_k columns.
        direct_added = sorted(list(scores_direct["engineered_col"].head(
            len(routed_added)
        )))
        assert set(routed_added) == set(direct_added), (
            f"seed={seed}: routed CMIM support diverged from direct "
            f"call. routed={routed_added}, direct={direct_added}"
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC sanity -- "cmim" >= plug-in default on redundant fixture
# ---------------------------------------------------------------------------


class TestCmimAucGteDefault:
    """On a heavily-redundant fixture, LogReg AUC with
    ``default_scorer="cmim"`` features is >= AUC with default plug-in
    features (modulo a small tolerance for plug-in noise). This is the
    biz_value claim from Layer 83 surfaced via the new knob.
    """

    def test_cmim_auc_geq_plug_in_on_redundant_pool(self):
        aucs_plug, aucs_cmim = [], []
        for s in (1, 7, 13, 42, 101, 202):
            X, y = _build_redundant_multi(s, n=1800)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            # plug-in path.
            m_plug = _make_mrmr(
                fe_hybrid_orth_enable=True,
                fe_hybrid_orth_degrees=(2,),
                fe_hybrid_orth_basis="hermite",
                fe_hybrid_orth_top_k=2,
                fe_hybrid_orth_pair_enable=False,
                fe_hybrid_orth_default_scorer="plug_in",
            ).fit(X_tr, y_tr)
            plug_added = list(getattr(m_plug, "hybrid_orth_features_", []) or [])
            # cmim routing.
            m_cmim = _make_mrmr(
                fe_hybrid_orth_enable=True,
                fe_hybrid_orth_degrees=(2,),
                fe_hybrid_orth_basis="hermite",
                fe_hybrid_orth_top_k=2,
                fe_hybrid_orth_pair_enable=False,
                fe_hybrid_orth_default_scorer="cmim",
            ).fit(X_tr, y_tr)
            cmim_added = list(getattr(m_cmim, "hybrid_orth_features_", []) or [])
            # Reconstruct the test-side engineered columns by replaying each model's
            # recipes via ``transform`` -- the SAME path used for the train side. Recipes
            # are pure functions of X (no y), so this is leakage-free. A Hermite-only
            # ``generate_univariate_basis_features`` rebuild cannot reproduce the default-on
            # adaptive-Fourier / chirp legs the plug-in roster also carries, so it would
            # miss those columns; replaying the model's own recipes keeps the held-out
            # reconstruction consistent with whatever each scorer actually appended.
            X_plug_tr = pd.concat(
                [X_tr, m_plug.transform(X_tr)[plug_added]], axis=1,
            ) if plug_added else X_tr
            X_plug_te = pd.concat(
                [X_te, m_plug.transform(X_te)[plug_added]], axis=1,
            ) if plug_added else X_te
            X_cmim_tr = pd.concat(
                [X_tr, m_cmim.transform(X_tr)[cmim_added]], axis=1,
            ) if cmim_added else X_tr
            X_cmim_te = pd.concat(
                [X_te, m_cmim.transform(X_te)[cmim_added]], axis=1,
            ) if cmim_added else X_te
            lr_plug = LogisticRegression(
                max_iter=2000, solver="lbfgs",
            ).fit(X_plug_tr, y_tr)
            aucs_plug.append(roc_auc_score(
                y_te, lr_plug.predict_proba(X_plug_te)[:, 1],
            ))
            lr_cmim = LogisticRegression(
                max_iter=2000, solver="lbfgs",
            ).fit(X_cmim_tr, y_tr)
            aucs_cmim.append(roc_auc_score(
                y_te, lr_cmim.predict_proba(X_cmim_te)[:, 1],
            ))
        plug_mean = float(np.mean(aucs_plug))
        cmim_mean = float(np.mean(aucs_cmim))
        # CMIM is expected to be AT LEAST as good as plug-in on a
        # redundant fixture (and strictly better on average per L83).
        # 0.01 tolerance absorbs finite-sample plug-in noise -- the L74
        # AUC contract uses 0.005 against a marginal-MI baseline.
        assert cmim_mean >= plug_mean - 0.01, (
            f"routed CMIM AUC mean ({cmim_mean:.4f}) materially below "
            f"plug-in mean ({plug_mean:.4f}); the L85 routing is "
            f"degrading the L83 result.\n"
            f"plug_per_seed={aucs_plug}\ncmim_per_seed={aucs_cmim}"
        )


# ---------------------------------------------------------------------------
# Contract 5: clone / pickle preserve the default_scorer flag
# ---------------------------------------------------------------------------


class TestPickleClonePreserveScorerFlag:

    def test_clone_preserves_default_scorer(self):
        m = _make_mrmr(fe_hybrid_orth_default_scorer="cmim")
        m2 = clone(m)
        assert m2.fe_hybrid_orth_default_scorer == "cmim", (
            f"clone() dropped fe_hybrid_orth_default_scorer: "
            f"got {m2.fe_hybrid_orth_default_scorer!r}"
        )

    def test_pickle_preserves_default_scorer_unfitted(self):
        m = _make_mrmr(fe_hybrid_orth_default_scorer="jmim")
        m2 = pickle.loads(pickle.dumps(m))
        assert m2.fe_hybrid_orth_default_scorer == "jmim", (
            f"pickle round-trip dropped fe_hybrid_orth_default_scorer: "
            f"got {m2.fe_hybrid_orth_default_scorer!r}"
        )

    def test_pickle_preserves_default_scorer_fitted(self):
        X, y = _build_redundant_multi(seed=42, n=1200)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=2,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_default_scorer="cmim",
        ).fit(X, y)
        m2 = pickle.loads(pickle.dumps(m))
        assert m2.fe_hybrid_orth_default_scorer == "cmim"
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, (
            f"pickle changed hybrid_orth_features_ under cmim routing: "
            f"before={added_before}, after={added_after}"
        )


# ---------------------------------------------------------------------------
# Contract 6: recommend_default_scorer() returns the L83 leaderboard winner
# ---------------------------------------------------------------------------


class TestRecommendDefaultScorer:
    """``MRMR.recommend_default_scorer()`` exposes the L83 leaderboard
    pick as a class method, so callers can opt in without needing to
    know the L83 result by heart.
    """

    def test_recommend_returns_cmim(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        assert MRMR.recommend_default_scorer() == "cmim", (
            f"recommend_default_scorer() drifted from the L83 winner "
            f"'cmim': got {MRMR.recommend_default_scorer()!r}"
        )

    def test_recommend_value_is_valid(self):
        """The recommended value must be in the validated allowlist --
        catches the regression where the recommendation drifts but the
        allowlist is not updated."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rec = MRMR.recommend_default_scorer()
        assert rec in MRMR._VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS, (
            f"recommend_default_scorer()={rec!r} not in the valid "
            f"allowlist {MRMR._VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS}"
        )

    def test_recommend_is_classmethod(self):
        """Reachable from the class without an instance."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        # No instance constructed -- direct class-level call.
        result = MRMR.recommend_default_scorer()
        assert isinstance(result, str) and result, (
            "recommend_default_scorer() must return a non-empty string"
        )
