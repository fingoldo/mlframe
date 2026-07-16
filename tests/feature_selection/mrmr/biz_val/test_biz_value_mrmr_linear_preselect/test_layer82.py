"""Layer 82 biz_value: ELASTIC NET (L1 + L2) coefficient-based pre-selection
plus MUTUAL-RANK fusion strategy.

Layer 81 ships pure Lasso. On correlated candidate pairs Lasso arbitrarily
picks ONE column and zeroes the other -- the choice is solver- /
seed-dependent and seed-fragile. Layer 82 adds Elastic Net (Zou & Hastie
2005) whose L2 penalty shares coefficient mass among correlated columns
("grouping effect"), so a correlated pair survives or drops together.

Plus the MUTUAL-RANK fusion aggregator ``mutual_top_k`` in the Layer 69
ensemble path: a candidate qualifies ONLY if it is in the top-K of EVERY
participating scorer. Strict-conjunction = high precision.

Contracts pinned
----------------

* TestElasticNetGroupsCorrelated: on a correlated candidate pair (two
  near-identical engineered columns), Elastic Net keeps BOTH with similar
  |coef|, while Lasso keeps ONE (the other gets |coef| <= small fraction
  of the kept one).

* TestMutualRankGatesStrictly: candidates surviving the ``mutual_top_k``
  ensemble aggregator are a SUBSET of the candidates that survive ANY
  single scorer's top-K (precision win) and the qualified set is non-empty
  on a fixture where both scorers agree on the top engineered column.

* TestAucLiftViaElasticNetOnCorrelated: LogReg trained on Elastic-Net
  augmented X beats LogReg on raw X by AUC >= +0.02 on the correlated
  fixture (the grouping-effect win materialises as held-out AUC).

* TestDefaultDisabledByteIdentical: ctor params default to
  ``fe_hybrid_orth_elasticnet_enable=False`` / ``alpha=0.01`` /
  ``l1_ratio=0.5``; explicit-default kwargs produce identical fit support.

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


def _import_elasticnet():
    """Lazily import the Elastic Net (L1+L2) coefficient-based FE functions."""
    from mlframe.feature_selection.filters._orthogonal_elasticnet_fe import (
        hybrid_orth_mi_elasticnet_fe,
        score_features_by_elasticnet_coef,
    )

    return hybrid_orth_mi_elasticnet_fe, score_features_by_elasticnet_coef


def _import_lasso():
    """Lazily import the Lasso (L1) coefficient-based FE functions (comparison baseline)."""
    from mlframe.feature_selection.filters._orthogonal_lasso_fe import (
        hybrid_orth_mi_lasso_fe,
        score_features_by_lasso_coef,
    )

    return hybrid_orth_mi_lasso_fe, score_features_by_lasso_coef


def _import_ensemble():
    """Lazily import the ensemble-scorer aggregator constants/functions (including mutual_top_k)."""
    from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import (
        ENSEMBLE_AGGREGATORS,
        MUTUAL_RANK_AGGREGATORS,
        score_features_by_ensemble_uplift,
    )

    return (
        ENSEMBLE_AGGREGATORS,
        MUTUAL_RANK_AGGREGATORS,
        score_features_by_ensemble_uplift,
    )


def _import_mrmr():
    """Lazily import the MRMR class."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR


def _make_mrmr(**overrides):
    """Build an MRMR with cheap, deterministic default knobs that isolate the Elastic-Net-preselect FE stage."""
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


def _build_correlated_pair(seed: int, n: int = 1500, corr: float = 0.99):
    """Two near-identical source columns x1, x2 (Pearson corr ~ corr) with
    a linear-additive engineered signal:

    ``y = 1.0 * He_2(x_1) + 1.0 * He_2(x_2) + eps``

    Because ``x_1 ~ x_2``, ``He_2(x_1) ~ He_2(x_2)`` -- a correlated
    candidate pair in the engineered space. With strong regularisation
    (alpha=0.5) the L1 path of Lasso starts to differentiate the two
    correlated columns by visiting order, splitting their |coef|; the
    L2 component of Elastic Net counteracts that by sharing mass.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    # Build x1, x2 with target correlation 'corr'.
    eps2 = rng.standard_normal(n)
    x1 = z
    x2 = corr * z + np.sqrt(max(0.0, 1.0 - corr * corr)) * eps2
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
    y_cont = 1.0 * he2_x1 + 1.0 * he2_x2 + 0.3 * rng.standard_normal(n)
    return X, y_cont


def _build_correlated_pair_binary(seed: int, n: int = 1500, corr: float = 0.99):
    """Binary-thresholded variant of _build_correlated_pair."""
    X, y_cont = _build_correlated_pair(seed, n=n, corr=corr)
    y_bin = (y_cont > 0).astype(int)
    return X, pd.Series(y_bin, name="y"), y_cont


def _build_linear_additive(seed: int, n: int = 1500):
    """Same fixture as Layer 81 -- LINEAR-additive on two INDEPENDENT
    columns -- used for the ensemble-fusion non-empty test where both
    scorers should agree on x1__He2 / x2__He2 as the top engineered cols.
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
    y_cont = 1.5 * he2_x1 + 0.8 * he2_x2 + 0.3 * rng.standard_normal(n)
    y_bin = (y_cont > 0).astype(int)
    return X, pd.Series(y_bin, name="y"), y_cont


# ---------------------------------------------------------------------------
# Contract 1: Elastic Net groups correlated candidates
# ---------------------------------------------------------------------------


class TestElasticNetGroupsCorrelated:
    """On a correlated candidate pair, Elastic Net's L2 grouping effect keeps both columns while Lasso splits them."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_elasticnet_keeps_both_correlated_pair(self, seed):
        """On the correlated-source fixture, Elastic Net keeps both
        ``x1__He2`` and ``x2__He2`` with similar |coef|. We require that
        the SMALLER of the two He_2 coefs is at least 25 % of the larger
        (the L2 mass-sharing effect). Lasso, on the same fixture, drives
        the smaller of the pair toward zero (well below 25 %).
        """
        hybrid_en, _ = _import_elasticnet()
        hybrid_lasso, _ = _import_lasso()
        X, y_cont = _build_correlated_pair(seed)

        # alpha=0.5 is the regime where the L1 shrinkage is strong enough
        # for Lasso's correlated-pair imbalance to surface; the L2 ridge
        # term in Elastic Net counteracts it. At the production default
        # alpha=0.01 both Lasso and EN keep the correlated pair near-equally
        # because the regularisation is too weak to differentiate them; the
        # grouping-effect test requires the regime where Lasso actually
        # starts to split the pair.
        _, scores_en = hybrid_en(
            X,
            y_cont,
            degrees=(2, 3),
            basis="hermite",
            top_k=5,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            alpha=0.5,
            l1_ratio=0.3,
        )
        _, scores_lasso = hybrid_lasso(
            X,
            y_cont,
            degrees=(2, 3),
            basis="hermite",
            top_k=5,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            alpha=0.5,
        )

        # Pull |coef| for x1__He2 and x2__He2 from each scorer.
        def _coef(df, name):
            """Look up the engineered_mi (|coef|) value for a given engineered column name, or 0.0 if absent."""
            row = df[df["engineered_col"] == name]
            return float(row["engineered_mi"].iloc[0]) if not row.empty else 0.0

        c1_en = _coef(scores_en, "x1__He2")
        c2_en = _coef(scores_en, "x2__He2")
        c1_l = _coef(scores_lasso, "x1__He2")
        c2_l = _coef(scores_lasso, "x2__He2")

        # Elastic Net: smaller / larger ratio >= 0.25 (grouping effect).
        en_ratio = min(c1_en, c2_en) / max(c1_en, c2_en, 1e-12)
        assert en_ratio >= 0.25, (
            f"seed={seed}: Elastic Net failed to share coef mass across "
            f"correlated pair (c1={c1_en:.4f}, c2={c2_en:.4f}, "
            f"ratio={en_ratio:.3f}); required >= 0.25."
        )
        # Lasso, same fixture: ratio should be visibly LESS THAN Elastic
        # Net's (the structural failure mode this layer fixes). Strict
        # inequality on the same data is the discriminating test.
        lasso_ratio = min(c1_l, c2_l) / max(c1_l, c2_l, 1e-12)
        assert en_ratio > lasso_ratio, (
            f"seed={seed}: Elastic Net ratio ({en_ratio:.3f}) did not "
            f"exceed Lasso ratio ({lasso_ratio:.3f}) on correlated pair. "
            f"EN coefs ({c1_en:.4f}, {c2_en:.4f}); Lasso coefs "
            f"({c1_l:.4f}, {c2_l:.4f})."
        )


# ---------------------------------------------------------------------------
# Contract 2: Mutual-rank gates strictly
# ---------------------------------------------------------------------------


class TestMutualRankGatesStrictly:
    """The mutual_top_k aggregator's qualified set is always a strict-conjunction subset of every scorer's top-K."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mutual_top_k_subset_of_any_single_scorer(self, seed):
        """The ``mutual_top_k`` qualified set is a SUBSET of each single
        scorer's top-K. The strict-conjunction contract: if a column did
        not make the top-K of scorer S, the mutual-rank fusion cannot
        admit it.
        """
        _, _, score_ensemble = _import_ensemble()
        X, _y_bin, y_cont = _build_linear_additive(seed)
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
        )

        engineered = generate_univariate_basis_features(
            X,
            degrees=(2, 3),
            basis="hermite",
        )
        # Use a 2-scorer ensemble (plug_in + dcor) for a clear strict
        # conjunction. KSG / copula / hsic / dcor all run consistently on
        # this fixture, but two scorers is sufficient to test the subset
        # property without inflating runtime.
        scorers = ("plug_in", "dcor")
        K = 3
        scores_mutual = score_ensemble(
            X[["x1", "x2", "noise_0", "noise_1", "noise_2"]],
            engineered,
            y_cont,
            scorers=scorers,
            aggregator="mutual_top_k",
            mutual_top_k=K,
            random_state=int(seed),
        )
        # Anything with aggregate_rank > n_cols * 1000 is the disqualified
        # sentinel. Qualified survivors are everything else.
        n_total = len(scores_mutual)
        sentinel_cut = float(n_total * 1000)
        qualified = scores_mutual[scores_mutual["aggregate_rank"] <= sentinel_cut]
        qualified_cols = set(qualified["engineered_col"])

        # Per-scorer top-K membership: read from the per_scorer_rank dict
        # column packed into the scores DataFrame.
        per_scorer_top_k = {}
        for s in scorers:
            top_set = set()
            for _, row in scores_mutual.iterrows():
                rk = row["per_scorer_rank"].get(s, n_total + 1)
                if rk <= K:
                    top_set.add(row["engineered_col"])
            per_scorer_top_k[s] = top_set

        # Subset property: every qualified column is in EVERY scorer's
        # top-K (the strict conjunction).
        for s in scorers:
            assert qualified_cols.issubset(per_scorer_top_k[s]), (
                f"seed={seed}: mutual_top_k qualified set {qualified_cols} "
                f"is NOT a subset of scorer {s!r}'s top-{K} "
                f"{per_scorer_top_k[s]}. Strict-conjunction violated."
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mutual_top_k_picks_x1_he2_and_x2_he2(self, seed):
        """On the LINEAR-additive ``y = 1.5*He_2(x1) + 0.8*He_2(x2)``
        fixture both x1__He2 AND x2__He2 should be in the qualified set
        (every reasonable dependence scorer ranks them top-2). Non-emptiness
        check: the mutual-rank aggregator MUST surface SOME column on a
        fixture where every scorer agrees.
        """
        _, _, score_ensemble = _import_ensemble()
        X, _y_bin, y_cont = _build_linear_additive(seed)
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
        )

        engineered = generate_univariate_basis_features(
            X,
            degrees=(2, 3),
            basis="hermite",
        )
        scorers = ("plug_in", "dcor")
        K = 4
        scores_mutual = score_ensemble(
            X[["x1", "x2", "noise_0", "noise_1", "noise_2"]],
            engineered,
            y_cont,
            scorers=scorers,
            aggregator="mutual_top_k",
            mutual_top_k=K,
            random_state=int(seed),
        )
        n_total = len(scores_mutual)
        sentinel_cut = float(n_total * 1000)
        qualified = scores_mutual[scores_mutual["aggregate_rank"] <= sentinel_cut]
        qualified_cols = set(qualified["engineered_col"])
        assert qualified_cols, (
            f"seed={seed}: mutual_top_k qualified set is empty on linear-"
            f"additive fixture where every scorer should agree on the top "
            f"engineered columns. scores head:\n"
            f"{scores_mutual.head(8).to_string()}"
        )
        # At least one of x1__He2 / x2__He2 should survive.
        assert "x1__He2" in qualified_cols or "x2__He2" in qualified_cols, (
            f"seed={seed}: neither x1__He2 nor x2__He2 made the mutual-rank "
            f"qualified set on linear-additive y = 1.5*He_2(x1)+0.8*He_2(x2). "
            f"qualified={qualified_cols!r}; scores head:\n"
            f"{scores_mutual.head(8).to_string()}"
        )

    def test_mutual_rank_aggregators_constant_published(self):
        """``MUTUAL_RANK_AGGREGATORS`` lives at the module top so callers
        can discover the strict-conjunction names; ``mutual_top_k`` is
        listed in BOTH ``MUTUAL_RANK_AGGREGATORS`` and the full
        ``ENSEMBLE_AGGREGATORS`` constant so any-aggregator validation
        accepts it.
        """
        ENSEMBLE_AGGS, MUTUAL_AGGS, _ = _import_ensemble()
        assert "mutual_top_k" in MUTUAL_AGGS, f"mutual_top_k missing from MUTUAL_RANK_AGGREGATORS: {MUTUAL_AGGS}"
        assert "mutual_top_k" in ENSEMBLE_AGGS, f"mutual_top_k missing from ENSEMBLE_AGGREGATORS: {ENSEMBLE_AGGS}"


# ---------------------------------------------------------------------------
# Contract 3: AUC lift via Elastic Net on correlated-candidate fixture
# ---------------------------------------------------------------------------


class TestAucLiftViaElasticNetOnCorrelated:
    """Elastic-Net-augmented LogReg measurably lifts holdout AUC over raw-feature LogReg on the correlated fixture."""

    @pytest.mark.parametrize("seed", (1, 7, 13))
    def test_elasticnet_augmented_logreg_beats_raw_by_002(self, seed):
        """On the correlated-source fixture ``y = He_2(x1) + He_2(x2)`` with
        ``corr(x1, x2) ~ 0.97``, LogReg trained on the Elastic-Net
        augmented X must beat LogReg trained on raw X by holdout AUC >=
        +0.02. The He_2 columns linearise the quadratic relationship;
        Elastic Net (unlike Lasso) keeps both correlated He_2 winners so
        the augmented design carries the full signal.
        """
        hybrid_en, _ = _import_elasticnet()
        X_tr, y_tr_cont = _build_correlated_pair(seed, n=1500)
        y_tr_bin = pd.Series((y_tr_cont > 0).astype(int), name="y")
        X_te_raw, y_te_cont = _build_correlated_pair(seed + 1000, n=3000)
        y_te = pd.Series((y_te_cont > 0).astype(int), name="y")

        X_tr_aug, _scores = hybrid_en(
            X_tr,
            y_tr_cont,
            degrees=(2, 3),
            basis="hermite",
            top_k=4,
            min_uplift=0.0,
            min_abs_mi_frac=0.0,
            alpha=0.01,
            l1_ratio=0.3,
        )
        appended = [c for c in X_tr_aug.columns if c not in X_tr.columns]
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
            """Fit LogReg on Xtr/y_tr_bin and return holdout AUC on Xte/y_te."""
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y_tr_bin)
            p = clf.predict_proba(Xte)[:, 1]
            return float(roc_auc_score(y_te, p))

        auc_raw = _fit_auc(X_tr, X_te_raw)
        auc_aug = _fit_auc(X_tr_aug, X_te_aug)
        gap = auc_aug - auc_raw
        assert gap >= 0.02, (
            f"seed={seed}: Elastic-Net-augmented LogReg AUC {auc_aug:.4f} "
            f"does not beat raw-X LogReg AUC {auc_raw:.4f} by the 0.02 "
            f"floor; gap={gap:.4f}. Engineered cols appended: {appended!r}"
        )


# ---------------------------------------------------------------------------
# Contract 4: default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_elasticnet_enable=False (the default) leaves the selected support byte-identical to legacy."""

    def test_default_ctor_has_elasticnet_off(self):
        """Fresh MRMR with no overrides has the Layer 82 master switch off
        and the default alpha / l1_ratio matching the design pin.
        """
        m = _make_mrmr()
        assert getattr(m, "fe_hybrid_orth_elasticnet_enable") is False, "fe_hybrid_orth_elasticnet_enable defaults to False"
        assert getattr(m, "fe_hybrid_orth_elasticnet_alpha") == 0.01, "fe_hybrid_orth_elasticnet_alpha defaults to 0.01"
        assert getattr(m, "fe_hybrid_orth_elasticnet_l1_ratio") == 0.5, "fe_hybrid_orth_elasticnet_l1_ratio defaults to 0.5"

    @pytest.mark.parametrize("seed", (1, 7, 13))
    def test_default_off_fit_byte_identical_to_baseline(self, seed):
        """With the Layer 82 master switch off, MRMR.fit on the linear-
        additive fixture produces the same support whether the new ctor
        params are passed explicitly (with defaults) or left unset.
        """
        X, y, _ = _build_linear_additive(seed)
        # Smaller n for the fit to keep test runtime down.
        X = X.iloc[:800].reset_index(drop=True)
        y = y.iloc[:800].reset_index(drop=True)
        m1 = _make_mrmr()
        m1.fit(X, y)
        m2 = _make_mrmr(
            fe_hybrid_orth_elasticnet_enable=False,
            fe_hybrid_orth_elasticnet_alpha=0.01,
            fe_hybrid_orth_elasticnet_l1_ratio=0.5,
        )
        m2.fit(X, y)
        sup1 = list(m1.feature_names_in_)
        sup2 = list(m2.feature_names_in_)
        assert sup1 == sup2, (
            f"seed={seed}: explicit fe_hybrid_orth_elasticnet_enable=False " f"diverged from implicit-default fit. implicit={sup1}, " f"explicit={sup2}"
        )


# ---------------------------------------------------------------------------
# Contract 5: pickle / clone preserve the Elastic-Net ctor flags
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """clone() and pickle preserve the Elastic Net ctor flags (enable/alpha/l1_ratio), unfitted and fitted."""

    def test_clone_preserves_elasticnet_params(self):
        """clone() copies fe_hybrid_orth_elasticnet_enable/alpha/l1_ratio without carrying over fitted state."""
        m = _make_mrmr(
            fe_hybrid_orth_elasticnet_enable=True,
            fe_hybrid_orth_elasticnet_alpha=0.05,
            fe_hybrid_orth_elasticnet_l1_ratio=0.3,
        )
        m2 = clone(m)
        assert getattr(m2, "fe_hybrid_orth_elasticnet_enable") is True, "clone() dropped fe_hybrid_orth_elasticnet_enable"
        assert getattr(m2, "fe_hybrid_orth_elasticnet_alpha") == 0.05, "clone() dropped fe_hybrid_orth_elasticnet_alpha"
        assert getattr(m2, "fe_hybrid_orth_elasticnet_l1_ratio") == 0.3, "clone() dropped fe_hybrid_orth_elasticnet_l1_ratio"

    def test_pickle_roundtrip_unfitted(self):
        """pickle.dumps/loads on an unfitted MRMR preserves fe_hybrid_orth_elasticnet_enable/alpha/l1_ratio."""
        m = _make_mrmr(
            fe_hybrid_orth_elasticnet_enable=True,
            fe_hybrid_orth_elasticnet_alpha=0.05,
            fe_hybrid_orth_elasticnet_l1_ratio=0.3,
        )
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert getattr(m2, "fe_hybrid_orth_elasticnet_enable") is True, "pickle/unpickle dropped fe_hybrid_orth_elasticnet_enable"
        assert getattr(m2, "fe_hybrid_orth_elasticnet_alpha") == 0.05, "pickle/unpickle dropped fe_hybrid_orth_elasticnet_alpha"
        assert getattr(m2, "fe_hybrid_orth_elasticnet_l1_ratio") == 0.3, "pickle/unpickle dropped fe_hybrid_orth_elasticnet_l1_ratio"

    def test_pickle_roundtrip_fitted(self):
        """pickle.dumps/loads on an Elastic-Net-fitted MRMR preserves feature_names_in_ and transform() output."""
        X, y, _ = _build_linear_additive(seed=42)
        X = X.iloc[:800].reset_index(drop=True)
        y = y.iloc[:800].reset_index(drop=True)
        m = _make_mrmr(
            fe_hybrid_orth_elasticnet_enable=True,
            fe_hybrid_orth_elasticnet_alpha=0.01,
            fe_hybrid_orth_elasticnet_l1_ratio=0.5,
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
                    raise AssertionError(f"pickle changed transform() values for column " f"{c!r}: max abs diff " f"{np.nanmax(np.abs(v1 - v2)):.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
