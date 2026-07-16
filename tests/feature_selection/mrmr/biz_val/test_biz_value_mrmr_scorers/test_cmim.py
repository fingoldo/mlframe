"""Layer 74 biz_value: CMIM (Fleuret 2004) redundancy-aware ranking for
hybrid orth-poly FE.

Validates ``score_features_by_cmim`` / ``hybrid_orth_mi_cmim_fe``
(sibling module ``_orthogonal_cmim_fe``) introduced 2026-06-01. Layer
72 (JMIM, Bennasar 2015) scores ``min_j I((X_k, X_j); Y)`` (joint MI --
rewards complementarity); Layer 74 (CMIM) scores

    J_CMIM(X_k) = min over X_j in S of  CMI(X_k ; Y | X_j)

which is a CONDITIONAL MI (the candidate's contribution GIVEN each
support member individually, then min) -- penalises redundancy via the
conditioning operator rather than rewarding complementarity via the
joint. Adding both gives the user a choice: JMIM wins on heavily-
interacting candidate pools, CMIM wins on heavily-duplicating ones.

Contracts pinned
----------------

* ``TestCmimRanksRedundantLow``: when a candidate is a near-deterministic
  function of an already-selected support member, CMIM collapses near
  zero (the conditional MI given that member is zero) whereas an
  independent candidate keeps its full MI -- the CMIM rank gap between
  the two is large.
* ``TestCmimVsJmimAgreement``: on a clean fixture without near-duplicate
  sources, CMIM's and JMIM's top picks roughly align (Spearman rank
  correlation >= 0.5 across the engineered pool).
* ``TestAucLiftViaCmim``: end-to-end -- CMIM-augmented LogReg AUC beats
  marginal-MI-augmented LogReg AUC on a multi-redundant fixture.
* ``TestDefaultDisabledByteIdentical``: master switch OFF leaves
  ``hybrid_orth_features_`` empty.
* ``TestPickleAndClone``: ``clone`` preserves ctor params; ``pickle``
  preserves appended features and recipe round-trip.

NEVER xfail.

Consolidated verbatim from test_biz_value_mrmr_layer74.py (per audit finding test_code_quality-16).
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


def _import_cmim_fe():
    """Lazily import the Layer-74 CMIM scoring/FE functions."""
    from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
        cmim_score,
        score_features_by_cmim,
        hybrid_orth_mi_cmim_fe,
        hybrid_orth_mi_cmim_fe_with_recipes,
    )
    return (
        cmim_score,
        score_features_by_cmim,
        hybrid_orth_mi_cmim_fe,
        hybrid_orth_mi_cmim_fe_with_recipes,
    )


def _import_jmim_fe():
    """Lazily import the Layer-72 JMIM scoring function used for the agreement contract."""
    from mlframe.feature_selection.filters._orthogonal_jmim_fe import (
        score_features_by_jmim,
    )
    return score_features_by_jmim


def _import_plug_in_fe():
    """Lazily import the Layer-21 plug-in marginal-MI univariate FE functions."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
        hybrid_orth_mi_fe,
    )
    return generate_univariate_basis_features, hybrid_orth_mi_fe


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_redundant_multi(seed: int, n: int = 2000):
    """Multi-redundant-candidate fixture (mirrors Layer 73).

    * ``x1`` carries the primary quadratic signal (``y depends on x1^2``).
    * ``x_dup_a``, ``x_dup_b``, ``x_dup_c`` are NEAR-COPIES of ``x1``.
      Their He_2 expansions all score high on marginal MI with y -- a
      marginal-MI top-K will fill with duplicates.
    * ``x2`` carries an INDEPENDENT secondary quadratic signal.

    Under CMIM with ``current_support = [x1]``, the candidate
    ``x_dup_*__He2`` columns are near-deterministic functions of x1
    (their source is a near-copy of x1) so ``CMI(He2(x_dup); y | x1)``
    collapses near zero -- the conditional MI given x1 strips out all
    the signal x1 already carries. By contrast ``CMI(He2(x2); y | x1)``
    keeps the full secondary signal because x2 is independent of x1.
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


def _build_clean_independent(seed: int, n: int = 2000):
    """Clean fixture with independent sources (no near-duplicates).

    Used for the CMIM-vs-JMIM agreement contract: on a pool without
    near-copies, the conditioning vs joint distinction is dominated by
    the marginal MI ordering both scorers share, so the rankings should
    roughly agree.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
        }
    )
    signal = 0.8 * (x1**2) + 0.6 * (x2**2) + 0.4 * (x3**2) + 0.2 * (x4**2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


from tests.feature_selection._biz_val_synth import _build_linear
# ---------------------------------------------------------------------------
# Contract 1: CMIM penalises redundancy with a support member
# ---------------------------------------------------------------------------


class TestCmimRanksRedundantLow:
    """The signature property of CMIM: a candidate whose source is a
    near-copy of an already-selected support member ``X_j`` posts a
    near-zero ``CMI(candidate; y | X_j)`` (the conditioning removes all
    the signal X_j already carries). An independent-source candidate
    keeps its full CMI because the conditioning on X_j cannot strip
    information X_j does not have.

    Under ``current_support = [x1]``, the CMIM score for
    ``x_dup_*__He2`` (whose source is a near-copy of x1) must be
    materially LOWER than the CMIM score for ``x2__He2`` (whose source
    is independent of x1). This is the CMIM-specific redundancy filter
    that pure marginal MI misses.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_cmim_redundant_far_below_novel(self, seed):
        """CMIM score for redundant x_dup_* columns is materially below the score for the independent x2 signal."""
        gen, _ = _import_plug_in_fe()
        _, score_features_by_cmim, _, _ = _import_cmim_fe()
        X, y = _build_redundant_multi(seed, n=2000)
        engineered = gen(X, degrees=(2,), basis="hermite")
        scores = score_features_by_cmim(
            X, engineered, y.to_numpy(),
            current_support=X[["x1"]],
            n_bins=10,
        )
        s_map = dict(zip(scores["engineered_col"], scores["engineered_mi"]))
        # Pick the He_2 columns we care about.
        eng_dup = [c for c in s_map if c.startswith(("x_dup_a__He", "x_dup_b__He", "x_dup_c__He"))]
        eng_x2 = [c for c in s_map if c.startswith("x2__He")]
        assert eng_dup, f"seed={seed}: no x_dup_*__He columns ranked: {list(s_map)}"
        assert eng_x2, f"seed={seed}: no x2__He columns ranked: {list(s_map)}"
        max_dup = float(max(s_map[c] for c in eng_dup))
        novel = float(s_map[eng_x2[0]])
        # Novel score must be strictly above the largest redundant score
        # by a clean margin. 0.02 nats is the same noise-clean floor
        # Layer 73's analogous test uses at this n.
        assert novel > max_dup + 0.02, (
            f"seed={seed}: CMIM novel x2__He2 ({novel:.4f}) not clearly "
            f"above max redundant x_dup_*__He2 ({max_dup:.4f}); "
            f"redundancy filter contract violated."
        )


# ---------------------------------------------------------------------------
# Contract 2: CMIM and JMIM rough agreement on a clean fixture
# ---------------------------------------------------------------------------


class TestCmimVsJmimAgreement:
    """On a pool of INDEPENDENT sources (no near-duplicates), the
    conditioning vs joint distinction between CMIM and JMIM is dominated
    by the shared marginal MI ordering both scorers produce. The two
    rankings should agree on the top picks even though the absolute
    scale of scores differs.

    Contract: Spearman rank correlation between CMIM and JMIM
    engineered_mi rankings is >= 0.5 (strong positive) averaged across
    seeds. (A stricter threshold like 0.8 would fail on the finite-
    sample plug-in noise floor; 0.5 is the smallest signal-vs-noise
    floor that distinguishes "the two scorers agree" from
    "uncorrelated rankings".)
    """

    def test_cmim_jmim_rank_agreement_clean_fixture(self):
        """CMIM and JMIM rankings show Spearman rho >= 0.5 on a clean independent-source fixture."""
        from scipy.stats import spearmanr
        gen, _ = _import_plug_in_fe()
        _, score_features_by_cmim, _, _ = _import_cmim_fe()
        score_features_by_jmim = _import_jmim_fe()
        corrs = []
        for s in SEEDS:
            X, y = _build_clean_independent(s, n=2000)
            engineered = gen(X, degrees=(2,), basis="hermite")
            cmim_scores = score_features_by_cmim(
                X, engineered, y.to_numpy(),
                current_support=None, n_bins=10,
            )
            jmim_scores = score_features_by_jmim(
                X, engineered, y.to_numpy(),
                current_support=None, n_bins=10,
            )
            # Align by engineered_col for the Spearman comparison.
            c_map = dict(zip(
                cmim_scores["engineered_col"],
                cmim_scores["engineered_mi"],
            ))
            j_map = dict(zip(
                jmim_scores["engineered_col"],
                jmim_scores["engineered_mi"],
            ))
            common = sorted(set(c_map) & set(j_map))
            assert len(common) >= 4, f"seed={s}: too few common engineered cols " f"({len(common)}) for the rank agreement test."
            c_vals = np.array([c_map[k] for k in common])
            j_vals = np.array([j_map[k] for k in common])
            rho, _ = spearmanr(c_vals, j_vals)
            # Float-safe: NaN spearman on constant columns shouldn't
            # silently pass through as zero. In practice the engineered
            # pool here always has variance so we never hit NaN.
            assert np.isfinite(rho), f"seed={s}: Spearman rho is NaN; ranking variance " f"degenerated."
            corrs.append(float(rho))
        mean_rho = float(np.mean(corrs))
        assert mean_rho >= 0.5, (
            f"CMIM-vs-JMIM mean Spearman rank correlation "
            f"({mean_rho:.4f}) below the 0.5 agreement floor on a clean "
            f"independent-source fixture.\nper_seed={corrs}"
        )


# ---------------------------------------------------------------------------
# Contract 3: AUC lift via CMIM-augmented LogReg
# ---------------------------------------------------------------------------


class TestAucLiftViaCmim:
    """End-to-end biz_value: appending CMIM-selected orth-poly columns
    lifts downstream LogReg AUC over the marginal-MI-augmented baseline
    on a fixture engineered to be heavy with multi-redundant candidates.
    """

    def test_cmim_augmented_logreg_auc_beats_marginal_mi(self):
        """CMIM-augmented LogReg AUC beats marginal-MI-augmented LogReg AUC by >= 0.005 on the redundant fixture."""
        gen, hybrid_marginal = _import_plug_in_fe()
        _, _, _, hybrid_cmim_with_recipes = _import_cmim_fe()
        aucs_marg, aucs_cmim = [], []
        for s in (1, 7, 13, 42, 101, 202, 303, 404):
            X, y = _build_redundant_multi(s, n=2000)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            # marginal-MI augmentation baseline.
            X_marg_tr, _ = hybrid_marginal(
                X_tr, y_tr.to_numpy(),
                degrees=(2,), basis="hermite",
                top_k=2, min_uplift=0.0, min_abs_mi_frac=0.0, nbins=10,
            )
            marg_added = [c for c in X_marg_tr.columns if c not in X_tr.columns]
            eng_te_all = gen(X_te, degrees=(2,), basis="hermite")
            X_marg_te = pd.concat([X_te, eng_te_all[marg_added]], axis=1) if marg_added else X_te
            lr_marg = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
            ).fit(X_marg_tr, y_tr)
            aucs_marg.append(
                roc_auc_score(
                    y_te,
                    lr_marg.predict_proba(X_marg_te)[:, 1],
                )
            )
            # CMIM augmentation: condition on x1 as the already-picked
            # support so the CMIM ranking suppresses the redundant
            # He_2(x_dup_*) duplicates and surfaces He_2(x2) instead.
            X_cmim_tr, _scores, _recipes = hybrid_cmim_with_recipes(
                X_tr, y_tr.to_numpy(),
                current_support=X_tr[["x1"]],
                degrees=(2,), basis="hermite",
                top_k=2, min_uplift=0.0, min_abs_mi_frac=0.0, n_bins=10,
            )
            cmim_added = [c for c in X_cmim_tr.columns if c not in X_tr.columns]
            X_cmim_te = pd.concat([X_te, eng_te_all[cmim_added]], axis=1) if cmim_added else X_te
            lr_cmim = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
            ).fit(X_cmim_tr, y_tr)
            aucs_cmim.append(
                roc_auc_score(
                    y_te,
                    lr_cmim.predict_proba(X_cmim_te)[:, 1],
                )
            )
        marg_mean = float(np.mean(aucs_marg))
        cmim_mean = float(np.mean(aucs_cmim))
        # CMIM's conditional-MI redundancy control lets a non-x1
        # secondary signal into the top-K instead of duplicating x1 --
        # worth at least 0.005 absolute AUC lift averaged over seeds
        # (same floor as Layer 72 / 73 analogous biz_value tests).
        assert cmim_mean > marg_mean + 0.005, (
            f"CMIM-augmented LogReg AUC mean ({cmim_mean:.4f}) not "
            f"lifted by >= 0.005 vs marginal-MI mean ({marg_mean:.4f}); "
            f"biz_value lift claim violated.\n"
            f"marg_per_seed={aucs_marg}\ncmim_per_seed={aucs_cmim}"
        )


# ---------------------------------------------------------------------------
# Contract 4: default disabled byte-identical with master
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_cmim_enable defaults to False and leaves hybrid_orth_features_ empty."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_cmim_columns(self, seed):
        """With the flag left at its False default, no CMIM columns are appended."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], f"seed={seed}: default fe_hybrid_orth_cmim_enable=False " f"should NOT append any engineered columns; got {added}"

    def test_default_ctor_values(self):
        """fe_hybrid_orth_cmim_enable defaults to False and fe_hybrid_orth_cmim_n_bins defaults to 10."""
        m = _make_mrmr()
        assert m.fe_hybrid_orth_cmim_enable is False
        assert m.fe_hybrid_orth_cmim_n_bins == 10


# ---------------------------------------------------------------------------
# Contract 5: pickle / clone preserve the ctor + recipes round-trip
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """CMIM ctor params and recipes must survive clone/pickle round-trips."""

    def test_clone_preserves_cmim_params(self):
        """sklearn clone() copies every fe_hybrid_orth_cmim_* ctor param."""
        m = _make_mrmr(
            fe_hybrid_orth_cmim_enable=True,
            fe_hybrid_orth_cmim_n_bins=15,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_cmim_enable", True),
            ("fe_hybrid_orth_cmim_n_bins", 15),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got " f"{getattr(m2, name)}"

    def test_pickle_roundtrip_preserves_cmim_recipes(self):
        """A pickle round-trip preserves feature names, appended columns, and every orth_univariate recipe field."""
        X, y = _build_redundant_multi(seed=42, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_cmim_enable=True,
            fe_hybrid_orth_cmim_n_bins=10,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=2,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, f"pickle changed hybrid_orth_features_: " f"before={added_before}, after={added_after}"

        def _extract_orth_recipes(model):
            """Return {name: recipe} for the orth_univariate recipes, regardless of container list/dict shape."""
            container = getattr(model, "_engineered_recipes_", None)
            if isinstance(container, dict):
                return {r.name: r for r in container.values() if getattr(r, "kind", None) == "orth_univariate"}
            return {r.name: r for r in (container or []) if getattr(r, "kind", None) == "orth_univariate"}

        recipes_before = _extract_orth_recipes(m)
        recipes_after = _extract_orth_recipes(m2)
        assert set(recipes_before.keys()) == set(recipes_after.keys()), (
            f"pickle dropped or added orth_univariate recipe names: " f"before={set(recipes_before.keys())}, " f"after={set(recipes_after.keys())}"
        )
        for name, r_before in recipes_before.items():
            r_after = recipes_after[name]
            assert r_before.src_names == r_after.src_names, f"pickle changed src_names for {name!r}: " f"before={r_before.src_names}, after={r_after.src_names}"
            for key in ("basis", "degree"):
                assert r_before.extra.get(key) == r_after.extra.get(key), (
                    f"pickle changed '{key}' for recipe {name!r}: " f"before={r_before.extra}, after={r_after.extra}"
                )
