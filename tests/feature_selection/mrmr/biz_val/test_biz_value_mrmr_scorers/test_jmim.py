"""Layer 72 biz_value: JMIM (Bennasar 2015) redundancy-aware ranking for
hybrid orth-poly FE.

Validates ``score_features_by_jmim`` / ``hybrid_orth_mi_jmim_fe``
(sibling module ``_orthogonal_jmim_fe``) introduced 2026-06-01. Layers
21 / 65 / 66 / 67 / 71 all rank by MARGINAL dependence with y; Layer 72
ranks by the WORST-CASE joint MI against the already-selected support

    J(X_k) = min over X_j in S of  I((X_k, X_j); Y)

so a candidate redundant with ANY support member is suppressed even if
it has a high marginal MI.

Contracts pinned
----------------

* ``TestJmimBeatsMarginalOnRedundant``: a synthetic redundant-candidate
  fixture where marginal MI ranks the wrong winner; JMIM picks the
  truly novel candidate.
* ``TestSelectionDiversity``: JMIM-augmented support has lower mean
  pairwise MI than marginal-MI-augmented support (cleaner, more
  diverse top-K).
* ``TestAucLiftOnRedundantFixture``: end-to-end -- JMIM-augmented
  LogReg AUC beats marginal-MI-augmented LogReg AUC on a redundant-
  candidate fixture.
* ``TestDefaultDisabledByteIdentical``: master switch OFF leaves
  ``hybrid_orth_features_`` empty.
* ``TestPickleAndClone``: ``clone`` preserves ctor params; ``pickle``
  preserves appended features and recipe round-trip.

NEVER xfail.

Consolidated verbatim from test_biz_value_mrmr_layer72.py (per audit finding test_code_quality-16).
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_jmim_fe():
    """Lazily import the Layer-72 JMIM scoring/FE functions."""
    from mlframe.feature_selection.filters._orthogonal_jmim_fe import (
        jmim_score,
        score_features_by_jmim,
        hybrid_orth_mi_jmim_fe,
        hybrid_orth_mi_jmim_fe_with_recipes,
    )

    return (
        jmim_score,
        score_features_by_jmim,
        hybrid_orth_mi_jmim_fe,
        hybrid_orth_mi_jmim_fe_with_recipes,
    )


def _import_plug_in_fe():
    """Lazily import the Layer-21 plug-in marginal-MI univariate FE functions."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
        score_features_by_mi_uplift,
    )

    return generate_univariate_basis_features, score_features_by_mi_uplift


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_redundant_quadratic(seed: int, n: int = 2000):
    """Quadratic-signal fixture with redundant copies of the SAME source.

    Construction
    ------------

    * ``x1`` carries the true quadratic signal (``y depends on x1 ** 2``).
    * ``x_dup_a``, ``x_dup_b``, ``x_dup_c`` are NEAR-COPIES of ``x1``
      (with small added noise) -- their He_2 polynomial expansions ALL
      give a high marginal MI with y. A marginal-MI ranking will put
      multiple of them in the top-K (redundant duplication); a JMIM
      ranking should suppress them because each duplicate's joint MI
      with x1 (which is also in the support) is no higher than the
      duplicate's marginal MI.
    * ``x2`` carries a SECONDARY independent quadratic signal so the
      "right" top-2 by JMIM is ``{He_2(x1), He_2(x2)}`` rather than
      ``{He_2(x1), He_2(x_dup_*)}``.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x_dup_a = x1 + 0.05 * rng.standard_normal(n)
    x_dup_b = x1 + 0.05 * rng.standard_normal(n)
    x_dup_c = x1 + 0.05 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    cols = {
        "x1": x1,
        "x_dup_a": x_dup_a,
        "x_dup_b": x_dup_b,
        "x_dup_c": x_dup_c,
        "x2": x2,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    }
    X = pd.DataFrame(cols)
    signal = x1**2 + 0.6 * (x2**2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


from tests.feature_selection._biz_val_synth import _build_linear, _quantile_bin_local


def _pairwise_mi_mean(df: pd.DataFrame, nbins: int = 10) -> float:
    """Mean pairwise MI across all unordered pairs of columns of ``df``.

    Lower = more diverse set. Used as the diversity metric for the
    selection-diversity contract.
    """
    if df.shape[1] < 2:
        return 0.0
    bins = [_quantile_bin_local(df[c].to_numpy(), nbins=nbins) for c in df.columns]
    p = df.shape[1]
    total = 0.0
    n_pairs = 0
    for i in range(p):
        for j in range(i + 1, p):
            total += float(mutual_info_score(bins[i], bins[j]))
            n_pairs += 1
    return total / max(1, n_pairs)


# ---------------------------------------------------------------------------
# Contract 1: JMIM beats marginal MI on redundant candidates
# ---------------------------------------------------------------------------


class TestJmimBeatsMarginalOnRedundant:
    """On the redundant-quadratic fixture, the marginal MI ranking will
    put multiple ``x_dup_*__He2`` columns in its top-K because each has
    a high marginal MI with y -- but only one of them carries truly new
    information. JMIM's ``min over support`` construction collapses the
    redundant duplicates' scores because joint MI with x1 (already in
    the support) does not exceed marginal MI(x1).

    The contract: JMIM must pick AT MOST ONE x_dup column in its top-2
    AND must include ``x2__He2`` in its top-2 across a majority of
    seeds; marginal MI typically fills the top-2 with duplicates and
    misses ``x2``.
    """

    def test_jmim_top_picks_include_secondary_signal(self):
        """JMIM includes the independent x2 signal in its top-2 on the majority of seeds."""
        gen, _ = _import_plug_in_fe()
        _, score_features_by_jmim, _, _ = _import_jmim_fe()
        jmim_picks_x2 = 0
        for s in SEEDS:
            X, y = _build_redundant_quadratic(s, n=2000)
            engineered = gen(X, degrees=(2,), basis="hermite")
            scores = score_features_by_jmim(
                X,
                engineered,
                y.to_numpy(),
                current_support=X[["x1"]],  # x1 already in support
                n_bins=10,
            )
            top2_sources = list(scores.head(2)["source_col"])
            if "x2" in top2_sources:
                jmim_picks_x2 += 1
        # JMIM must surface the independent secondary signal (x2) in the
        # top-2 on the majority of seeds. Marginal MI typically fills the
        # top-2 with x_dup_* duplicates of x1.
        assert jmim_picks_x2 >= 3, f"JMIM picked x2__He2 in top-2 on only {jmim_picks_x2}/{len(SEEDS)} seeds; redundancy-control contract violated."

    def test_jmim_suppresses_redundant_dups_more_than_marginal(self):
        """Concrete witness of the redundancy-control mechanism: the
        FRACTION of x_dup_* columns in the JMIM top-3 must be strictly
        less than the fraction in the marginal-MI top-3 averaged over
        seeds.
        """
        gen, score_marginal = _import_plug_in_fe()
        _, score_features_by_jmim, _, _ = _import_jmim_fe()
        jmim_dup_count = 0
        marg_dup_count = 0
        total = 0
        for s in SEEDS:
            X, y = _build_redundant_quadratic(s, n=2000)
            engineered = gen(X, degrees=(2,), basis="hermite")
            jmim_scores = score_features_by_jmim(
                X,
                engineered,
                y.to_numpy(),
                current_support=X[["x1"]],
                n_bins=10,
            )
            marg_scores = score_marginal(
                X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]],
                engineered,
                y.to_numpy(),
                nbins=10,
            )
            jmim_top3 = list(jmim_scores.head(3)["source_col"])
            marg_top3 = list(marg_scores.head(3)["source_col"])
            jmim_dup_count += sum(1 for c in jmim_top3 if c.startswith("x_dup"))
            marg_dup_count += sum(1 for c in marg_top3 if c.startswith("x_dup"))
            total += 3
        # Strict comparative gate: JMIM admits FEWER x_dup_* columns
        # than marginal MI on the same fixture, with margin >= 1 col.
        assert jmim_dup_count + 1 <= marg_dup_count, (
            f"JMIM redundancy suppression not observed: jmim_dup_count="
            f"{jmim_dup_count}, marg_dup_count={marg_dup_count} "
            f"(out of {total} top-3 slots across {len(SEEDS)} seeds)."
        )


# ---------------------------------------------------------------------------
# Contract 2: selection diversity (lower pairwise MI in JMIM-augmented)
# ---------------------------------------------------------------------------


class TestSelectionDiversity:
    """Direct diversity witness: the mean pairwise MI across columns in
    a JMIM-augmented support is lower than across columns in a marginal-
    MI-augmented support on the same redundant-quadratic fixture. JMIM's
    min-over-support construction is the explicit mechanism enforcing
    this.
    """

    def test_jmim_augmented_has_lower_pairwise_mi(self):
        """Compare the RAW top-3 picks of each scorer (gates bypassed) so
        the diversity claim is about the SCORING criterion itself, not
        about how each layer's two-gate selector chooses to admit /
        reject candidates. The marginal-MI hybrid's MAD-noise floor
        rejects all candidates on this fixture (every engineered col
        looks like a real signal when the source pool itself is heavy
        with near-copies); selecting the top-K by raw ranking is the
        apples-to-apples comparison.
        """
        gen, score_marginal = _import_plug_in_fe()
        _, score_features_by_jmim, _, _ = _import_jmim_fe()
        jmim_diversities, marg_diversities = [], []
        for s in SEEDS:
            X, y = _build_redundant_quadratic(s, n=2000)
            engineered = gen(X, degrees=(2,), basis="hermite")
            jmim_scores = score_features_by_jmim(
                X,
                engineered,
                y.to_numpy(),
                current_support=X[["x1"]],
                n_bins=10,
            )
            raw_X = X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]]
            marg_scores = score_marginal(
                raw_X,
                engineered,
                y.to_numpy(),
                nbins=10,
            )
            jmim_top = list(jmim_scores.head(3)["engineered_col"])
            marg_top = list(marg_scores.head(3)["engineered_col"])
            if jmim_top:
                jmim_diversities.append(_pairwise_mi_mean(engineered[jmim_top]))
            if marg_top:
                marg_diversities.append(_pairwise_mi_mean(engineered[marg_top]))
        assert jmim_diversities, "JMIM ranked no columns across seeds"
        assert marg_diversities, "marginal-MI ranked no columns across seeds"
        jmim_mean = float(np.mean(jmim_diversities))
        marg_mean = float(np.mean(marg_diversities))
        assert jmim_mean < marg_mean, (
            f"JMIM top-3 mean pairwise MI ({jmim_mean:.4f}) not strictly "
            f"lower than marginal-MI top-3 ({marg_mean:.4f}); diversity "
            f"contract violated. jmim_per_seed={jmim_diversities}, "
            f"marg_per_seed={marg_diversities}"
        )


# ---------------------------------------------------------------------------
# Contract 3: AUC lift on redundant fixture
# ---------------------------------------------------------------------------


class TestAucLiftOnRedundantFixture:
    """End-to-end biz_value: appending JMIM-selected orth-poly columns
    lifts downstream LogReg AUC over the marginal-MI-augmented baseline
    on a fixture engineered to be heavy with redundant candidates.
    """

    def test_jmim_augmented_logreg_auc_beats_marginal_mi(self):
        """JMIM-augmented LogReg AUC beats marginal-MI-augmented AUC by >= 0.005 on the redundant fixture."""
        gen, _ = _import_plug_in_fe()
        _, _, _, hybrid_with_recipes = _import_jmim_fe()
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            hybrid_orth_mi_fe,
        )

        aucs_marg, aucs_jmim = [], []
        for s in (1, 7, 13, 42, 101, 202, 303, 404):
            X, y = _build_redundant_quadratic(s, n=2000)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=s,
                stratify=y,
            )
            # marginal-MI augmentation baseline
            X_marg_tr, _ = hybrid_orth_mi_fe(
                X_tr,
                y_tr.to_numpy(),
                degrees=(2,),
                basis="hermite",
                top_k=2,
                min_uplift=0.0,
                min_abs_mi_frac=0.0,
                nbins=10,
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
            # JMIM augmentation
            X_jmim_tr, _scores, _recipes = hybrid_with_recipes(
                X_tr,
                y_tr.to_numpy(),
                current_support=X_tr[["x1"]],
                degrees=(2,),
                basis="hermite",
                top_k=2,
                min_uplift=0.0,
                min_abs_mi_frac=0.0,
                n_bins=10,
            )
            jmim_added = [c for c in X_jmim_tr.columns if c not in X_tr.columns]
            X_jmim_te = pd.concat([X_te, eng_te_all[jmim_added]], axis=1) if jmim_added else X_te
            lr_jmim = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
            ).fit(X_jmim_tr, y_tr)
            aucs_jmim.append(
                roc_auc_score(
                    y_te,
                    lr_jmim.predict_proba(X_jmim_te)[:, 1],
                )
            )
        marg_mean = float(np.mean(aucs_marg))
        jmim_mean = float(np.mean(aucs_jmim))
        # JMIM's redundancy control lets a non-x1 secondary signal into
        # the top-K instead of duplicating x1 -- worth at least a 0.005
        # absolute AUC lift averaged over seeds. A 0.005 floor is the
        # smallest noise-clean step distinguishable from seed-noise at
        # this n / class balance.
        assert jmim_mean > marg_mean + 0.005, (
            f"JMIM-augmented LogReg AUC mean ({jmim_mean:.4f}) not lifted "
            f"by >= 0.005 vs marginal-MI mean ({marg_mean:.4f}); biz_value "
            f"lift claim violated.\nmarg_per_seed={aucs_marg}\n"
            f"jmim_per_seed={aucs_jmim}"
        )


# ---------------------------------------------------------------------------
# Contract 4: default disabled byte-identical with master
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_jmim_enable defaults to False and leaves hybrid_orth_features_ empty."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_jmim_columns(self, seed):
        """With the flag left at its False default, no JMIM columns are appended."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], f"seed={seed}: default fe_hybrid_orth_jmim_enable=False should NOT append any engineered columns; got {added}"

    def test_default_ctor_values(self):
        """fe_hybrid_orth_jmim_enable defaults to False and fe_hybrid_orth_jmim_n_bins defaults to 10."""
        m = _make_mrmr()
        assert m.fe_hybrid_orth_jmim_enable is False
        assert m.fe_hybrid_orth_jmim_n_bins == 10


# ---------------------------------------------------------------------------
# Contract 5: pickle / clone preserve the ctor + recipes round-trip
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """JMIM ctor params and recipes must survive clone/pickle round-trips."""

    def test_clone_preserves_jmim_params(self):
        """sklearn clone() copies every fe_hybrid_orth_jmim_* ctor param."""
        m = _make_mrmr(
            fe_hybrid_orth_jmim_enable=True,
            fe_hybrid_orth_jmim_n_bins=15,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_jmim_enable", True),
            ("fe_hybrid_orth_jmim_n_bins", 15),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got {getattr(m2, name)}"

    def test_pickle_roundtrip_preserves_jmim_recipes(self):
        """A pickle round-trip preserves feature names, appended columns, and every orth_univariate recipe field."""
        X, y = _build_redundant_quadratic(seed=42, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_jmim_enable=True,
            fe_hybrid_orth_jmim_n_bins=10,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=2,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, f"pickle changed hybrid_orth_features_: before={added_before}, after={added_after}"

        def _extract_orth_recipes(model):
            """Return {name: recipe} for the orth_univariate recipes, regardless of container list/dict shape."""
            container = getattr(model, "_engineered_recipes_", None)
            if isinstance(container, dict):
                return {r.name: r for r in container.values() if getattr(r, "kind", None) == "orth_univariate"}
            return {r.name: r for r in (container or []) if getattr(r, "kind", None) == "orth_univariate"}

        recipes_before = _extract_orth_recipes(m)
        recipes_after = _extract_orth_recipes(m2)
        assert set(recipes_before.keys()) == set(recipes_after.keys()), (
            f"pickle dropped or added orth_univariate recipe names: before={set(recipes_before.keys())}, after={set(recipes_after.keys())}"
        )
        for name, r_before in recipes_before.items():
            r_after = recipes_after[name]
            assert r_before.src_names == r_after.src_names, f"pickle changed src_names for {name!r}: before={r_before.src_names}, after={r_after.src_names}"
            for key in ("basis", "degree"):
                assert r_before.extra.get(key) == r_after.extra.get(key), (
                    f"pickle changed '{key}' for recipe {name!r}: before={r_before.extra}, after={r_after.extra}"
                )
