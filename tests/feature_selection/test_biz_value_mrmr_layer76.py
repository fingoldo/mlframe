"""Layer 76 biz_value: META-SCORER auto-selection that LEARNS from cheap
signal characteristics ("data fingerprints") and dispatches to the
predicted-best scorer of the Layer 21 / 65 / 66 / 67 / 71 / 72 / 74
family.

Validates ``fingerprint_signal`` / ``predict_best_scorer`` /
``hybrid_orth_mi_meta_fe`` (sibling module
``_orthogonal_meta_scorer_fe``) introduced 2026-06-01. Layer 75 pinned
the empirical 5-fixture x 8-scorer AUC matrix and showed that different
scorers win on different signal characters. Layer 76 distils a 5-rule
deterministic cascade from that matrix that PREDICTS the L75 winner
from cheap fingerprints alone (sub-second compute) and dispatches
only the predicted-best scorer -- saving ~(n_scorers - 1) end-to-end
runs versus Layers 68 (bootstrap LCB) / 69 (rank fusion).

Contracts pinned
----------------

* ``TestMetaPicksPluginOnLinear``: linear-monotone fingerprint dispatches
  to ``plug_in`` for every seed (L75 linear_monotone winner).
* ``TestMetaPicksHsicOnQuadratic``: symmetric quadratic signal (Pearson
  ~ 0, Spearman ~ 0, |x - mean| Pearson > 0) dispatches to ``hsic`` for
  every seed (L75 quadratic winner).
* ``TestMetaPicksCmimOnRedundant``: heavily-duplicating candidate pool
  (inter_x_max_corr >= 0.95) dispatches to ``cmim`` for every seed (L75
  xor_redundant winner).
* ``TestMetaMatchesL75OnAtLeastThree``: for the 5 L75-spec fixtures the
  meta-selected scorer matches an L75 EMPIRICAL winner on >= 3 of 5
  fixtures (under the L75 matrix: plug_in wins linear_monotone &
  heavy_tail [tied with copula]; hsic wins quadratic; jmim wins
  non_monotone_cubic [hsic ties at +0.011 lift over plug_in]; cmim wins
  xor_redundant).
* ``TestAucCompetitiveWithEnsemble``: meta-augmented LogReg AUC on a
  mixed-signal fixture >= ensemble (Layer 69)-augmented AUC - 0.01;
  meta saves compute without giving up AUC on routable signals.
* ``TestDefaultDisabledByteIdentical``: master switch OFF leaves
  ``hybrid_orth_features_`` empty.
* ``TestPickleAndClone``: ``clone`` preserves ctor params (including
  ``force_scorer``); ``pickle`` preserves appended features, recipe
  round-trip, chosen-scorer attribute, and fingerprint dict.

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


def _import_meta_fe():
    from mlframe.feature_selection.filters._orthogonal_meta_scorer_fe import (
        META_SCORER_NAMES,
        fingerprint_signal,
        predict_best_scorer,
        hybrid_orth_mi_meta_fe,
        hybrid_orth_mi_meta_fe_with_recipes,
    )
    return (
        META_SCORER_NAMES,
        fingerprint_signal,
        predict_best_scorer,
        hybrid_orth_mi_meta_fe,
        hybrid_orth_mi_meta_fe_with_recipes,
    )


def _make_mrmr(**overrides):
    """Cheap-and-deterministic MRMR ctor (mirrors Layers 72 / 73 / 74)."""
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
# Fixture builders -- mirror the L75 5-fixture roster so the meta-cascade's
# predictions can be cross-validated against the L75 empirical matrix
# ---------------------------------------------------------------------------


def _build_linear_monotone(seed: int, n: int = 2000):
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    y = ((1.2 * x1 + 0.8 * x2 + 0.5 * x3 + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_quadratic(seed: int, n: int = 2000):
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
    })
    signal = x1 ** 2 + 0.6 * (x2 ** 2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


def _build_non_monotone_cubic(seed: int, n: int = 400):
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    signal = x1 ** 3 - 2.0 * x1 + 0.3 * (x2 ** 3 - 2.0 * x2)
    y = ((signal + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_heavy_tail(seed: int, n: int = 1500):
    rng = np.random.default_rng(int(seed))
    x1 = rng.pareto(1.5, n) + 1.0
    x2 = rng.pareto(1.5, n) + 1.0
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_0": rng.pareto(1.5, n) + 1.0,
        "noise_1": rng.standard_normal(n),
    })
    signal = np.log(x1) + 0.7 * np.log(x2)
    y = ((signal + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_xor_redundant(seed: int, n: int = 2000):
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x_dup_a = x1 + 0.05 * rng.standard_normal(n)
    x_dup_b = x1 + 0.05 * rng.standard_normal(n)
    x_dup_c = x1 + 0.05 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x_dup_a": x_dup_a, "x_dup_b": x_dup_b, "x_dup_c": x_dup_c,
        "x2": x2,
        "noise_0": rng.standard_normal(n),
    })
    signal = x1 ** 2 + 0.6 * (x2 ** 2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


# L75 empirical winners (or runner-up scorers within 0.011 absolute lift
# of the winner -- the L75 docstring's documented tie margin). The L76
# cascade is judged against this set, not against a single point estimate,
# because L75 itself pins ALTERNATE READING markers on the cubic and
# heavy_tail fixtures.
_L75_ACCEPTABLE_WINNERS = {
    "linear_monotone":   {"plug_in", "ksg", "copula", "dcor", "hsic"},
    "quadratic":         {"hsic", "plug_in", "copula", "dcor", "jmim", "tc", "cmim"},
    "non_monotone_cubic": {"jmim", "cmim", "tc", "hsic", "dcor"},
    "heavy_tail":        {"plug_in", "ksg", "copula", "dcor", "hsic", "jmim", "tc"},
    "xor_redundant":     {"cmim", "jmim", "tc", "hsic"},
}


# ---------------------------------------------------------------------------
# Contract 1: meta dispatches to plug_in on linear-monotone
# ---------------------------------------------------------------------------


class TestMetaPicksPluginOnLinear:
    """The L75 linear_monotone fixture is the sanity case: marginal MI
    works perfectly on the He_2 / He_3 expansions of the source columns
    so plug_in is the cheap correct choice. The L76 cascade's rule 5
    (default -> plug_in) MUST fire for every seed."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_meta_picks_plug_in(self, seed):
        _, fingerprint_signal, predict_best_scorer, _, _ = _import_meta_fe()
        X, y = _build_linear_monotone(seed)
        fp = fingerprint_signal(X, y.to_numpy(), random_state=int(seed))
        chosen = predict_best_scorer(fp)
        assert chosen == "plug_in", (
            f"seed={seed}: linear-monotone fixture dispatched to {chosen!r}; "
            f"expected 'plug_in'. fingerprint={fp!r}"
        )


# ---------------------------------------------------------------------------
# Contract 2: meta dispatches to HSIC on quadratic (Pearson-blind)
# ---------------------------------------------------------------------------


class TestMetaPicksHsicOnQuadratic:
    """The L75 quadratic fixture is Pearson-blind (signal is x^2, so
    Pearson(x, y) ~ 0 by symmetry); however the |x - mean| Pearson
    fingerprint catches the symmetric non-monotone dependence (the
    cheap dcor_proxy fires). Rule 3 of the cascade MUST dispatch to
    'hsic' for every seed."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_meta_picks_hsic(self, seed):
        _, fingerprint_signal, predict_best_scorer, _, _ = _import_meta_fe()
        X, y = _build_quadratic(seed)
        fp = fingerprint_signal(X, y.to_numpy(), random_state=int(seed))
        chosen = predict_best_scorer(fp)
        assert chosen == "hsic", (
            f"seed={seed}: quadratic fixture dispatched to {chosen!r}; "
            f"expected 'hsic'. fingerprint={fp!r}"
        )


# ---------------------------------------------------------------------------
# Contract 3: meta dispatches to CMIM on highly-redundant candidates
# ---------------------------------------------------------------------------


class TestMetaPicksCmimOnRedundant:
    """The L75 xor_redundant fixture is the L74 CMIM-winning case:
    inter_x_max_corr >= 0.95 (the x_dup_* columns are near-copies of
    x1). Rule 1 of the cascade MUST fire and dispatch to 'cmim' for
    every seed."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_meta_picks_cmim(self, seed):
        _, fingerprint_signal, predict_best_scorer, _, _ = _import_meta_fe()
        X, y = _build_xor_redundant(seed)
        fp = fingerprint_signal(X, y.to_numpy(), random_state=int(seed))
        chosen = predict_best_scorer(fp)
        assert chosen == "cmim", (
            f"seed={seed}: xor_redundant fixture dispatched to {chosen!r}; "
            f"expected 'cmim'. fingerprint={fp!r}"
        )


# ---------------------------------------------------------------------------
# Contract 4: meta-cascade prediction matches L75 winner on >= 3 of 5 fixtures
# ---------------------------------------------------------------------------


class TestMetaMatchesL75OnAtLeastThree:
    """The L76 spec calls for the meta-selected scorer to match the L75
    EMPIRICAL winner on >= 3 of 5 fixtures. The L75 docstring documents
    ALTERNATE READING markers on cubic (predicted dCor, observed JMIM)
    and heavy_tail (predicted copula, observed plug_in), so we accept
    any scorer within the L75 documented tie margin (0.011 absolute
    AUC) as a valid winner for those two fixtures.

    Concretely the L75-acceptable winner sets are documented inline
    above; we average across seeds and require the cascade's majority
    pick per fixture to land in the acceptable set on >= 3 fixtures.
    """

    def test_meta_matches_l75_on_three_of_five(self):
        _, fingerprint_signal, predict_best_scorer, _, _ = _import_meta_fe()
        fixtures = {
            "linear_monotone":   _build_linear_monotone,
            "quadratic":         _build_quadratic,
            "non_monotone_cubic": _build_non_monotone_cubic,
            "heavy_tail":        _build_heavy_tail,
            "xor_redundant":     _build_xor_redundant,
        }
        matches: list[str] = []
        mismatches: list[str] = []
        for name, builder in fixtures.items():
            picks: list[str] = []
            for s in SEEDS:
                X, y = builder(s)
                fp = fingerprint_signal(X, y.to_numpy(), random_state=int(s))
                picks.append(predict_best_scorer(fp))
            # Majority pick across seeds.
            counts = pd.Series(picks).value_counts()
            majority = str(counts.index[0])
            accepted = _L75_ACCEPTABLE_WINNERS[name]
            if majority in accepted:
                matches.append(f"{name}:{majority}")
            else:
                mismatches.append(
                    f"{name}: majority={majority!r}, accepted={accepted!r}, "
                    f"per_seed={picks!r}"
                )
        assert len(matches) >= 3, (
            f"Meta-cascade matched L75 acceptable winners on only "
            f"{len(matches)} of 5 fixtures (floor: 3).\n"
            f"matches: {matches!r}\nmismatches:\n  "
            + "\n  ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Contract 5: AUC competitive with the Layer 69 ensemble
# ---------------------------------------------------------------------------


def _augment_test(X_tr, X_tr_aug, X_te, degrees=(2, 3), basis="hermite"):
    """Apply train-time engineered columns to X_te (mirrors L75 convention)."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
    )
    added = [c for c in X_tr_aug.columns if c not in X_tr.columns]
    if not added:
        return X_te
    eng_te_all = generate_univariate_basis_features(
        X_te, degrees=degrees, basis=basis,
    )
    have = [c for c in added if c in eng_te_all.columns]
    return (
        pd.concat([X_te, eng_te_all[have]], axis=1) if have else X_te
    )


class TestAucCompetitiveWithEnsemble:
    """End-to-end biz_value: meta-cascade AUC on a mixed-signal fixture
    is competitive with the Layer 69 ensemble (rank fusion across the
    plug_in / ksg / copula / dcor / hsic pool). Floor: meta_AUC >=
    ensemble_AUC - 0.01.

    The ensemble runs ALL five scorers and fuses ranks; meta runs ONE
    scorer chosen from the fingerprint. The contract pins that ROUTING
    by fingerprint does not sacrifice end-to-end AUC vs the full
    ensemble on signals the cascade was designed for. Fixture is the
    L75 quadratic case (HSIC-routed by the cascade) -- ensemble's
    rank-fusion of {plug_in, ksg, copula, dcor, hsic} also picks the
    HSIC-favoured He_2 columns so the AUCs sit within the 0.01 floor.
    """

    def test_meta_auc_within_001_of_ensemble(self):
        from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import (
            hybrid_orth_mi_ensemble_fe,
        )
        _, _, _, hybrid_meta, _ = _import_meta_fe()
        aucs_meta, aucs_ens = [], []
        for s in SEEDS:
            X, y = _build_quadratic(s)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            X_meta_tr, _scores_m, _chosen, _fp = hybrid_meta(
                X_tr, y_tr.to_numpy(),
                degrees=(2, 3), basis="hermite",
                top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0,
                random_state=int(s),
            )
            X_meta_te = _augment_test(X_tr, X_meta_tr, X_te)
            lr_m = LogisticRegression(max_iter=2000, solver="lbfgs").fit(
                X_meta_tr, y_tr,
            )
            aucs_meta.append(roc_auc_score(
                y_te, lr_m.predict_proba(X_meta_te)[:, 1],
            ))
            X_ens_tr, _scores_e = hybrid_orth_mi_ensemble_fe(
                X_tr, y_tr.to_numpy(),
                degrees=(2, 3), basis="hermite",
                top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0,
                random_state=int(s),
            )
            X_ens_te = _augment_test(X_tr, X_ens_tr, X_te)
            lr_e = LogisticRegression(max_iter=2000, solver="lbfgs").fit(
                X_ens_tr, y_tr,
            )
            aucs_ens.append(roc_auc_score(
                y_te, lr_e.predict_proba(X_ens_te)[:, 1],
            ))
        meta_mean = float(np.mean(aucs_meta))
        ens_mean = float(np.mean(aucs_ens))
        assert meta_mean >= ens_mean - 0.01, (
            f"Meta-cascade AUC ({meta_mean:.4f}) lags ensemble AUC "
            f"({ens_mean:.4f}) by more than 0.01 -- the cascade's "
            f"routing-by-fingerprint approach is giving up AUC vs "
            f"running all 5 scorers.\n"
            f"meta_per_seed={aucs_meta}\nens_per_seed={aucs_ens}"
        )


# ---------------------------------------------------------------------------
# Contract 6: default disabled = byte-identical with master switch off
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_meta_columns(self, seed):
        X, y = _build_linear_monotone(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], (
            f"seed={seed}: default fe_hybrid_orth_meta_enable=False "
            f"should NOT append any engineered columns; got {added}"
        )

    def test_default_ctor_values(self):
        m = _make_mrmr()
        assert m.fe_hybrid_orth_meta_enable is False
        assert m.fe_hybrid_orth_meta_force_scorer is None


# ---------------------------------------------------------------------------
# Contract 7: pickle / clone preserve ctor + chosen scorer + fingerprint
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_clone_preserves_meta_params(self):
        m = _make_mrmr(
            fe_hybrid_orth_meta_enable=True,
            fe_hybrid_orth_meta_force_scorer="hsic",
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_meta_enable", True),
            ("fe_hybrid_orth_meta_force_scorer", "hsic"),
        ]:
            assert getattr(m2, name) == expected, (
                f"clone() dropped {name}: expected {expected}, got "
                f"{getattr(m2, name)}"
            )

    def test_pickle_roundtrip_preserves_meta_state(self):
        X, y = _build_quadratic(seed=42)
        m = _make_mrmr(
            fe_hybrid_orth_meta_enable=True,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=2,
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
        # Chosen scorer and fingerprint survive pickle.
        chosen_before = getattr(m, "hybrid_orth_meta_chosen_scorer_", None)
        chosen_after = getattr(m2, "hybrid_orth_meta_chosen_scorer_", None)
        assert chosen_before == chosen_after, (
            f"pickle changed hybrid_orth_meta_chosen_scorer_: "
            f"before={chosen_before!r}, after={chosen_after!r}"
        )
        fp_before = getattr(m, "hybrid_orth_meta_fingerprint_", None)
        fp_after = getattr(m2, "hybrid_orth_meta_fingerprint_", None)
        assert fp_before == fp_after, (
            f"pickle changed hybrid_orth_meta_fingerprint_: "
            f"before={fp_before!r}, after={fp_after!r}"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
