"""Layer 73 biz_value: Total Correlation (Watanabe 1960) multivariate-
redundancy ranking for hybrid orth-poly FE.

Validates ``score_features_by_tc_uplift`` / ``hybrid_orth_mi_tc_fe``
(sibling module ``_orthogonal_total_correlation_fe``) introduced
2026-06-01. Layers 21 / 65 / 66 / 67 / 71 rank by MARGINAL dependence
with y; Layer 72 (JMIM) ranks by the worst PAIRWISE joint MI with the
support. Layer 73 ranks by the FULL-ORDER joint shared information
contribution

    delta_tc = TC([support, c, y]) - TC([support, y])

where ``TC(Z) = sum H(Z_i) - H(Z)``. The full-order joint MI catches
HIGHER-ORDER redundancy (e.g. XOR-style three-variable parity where
every pairwise MI is zero) that all pairwise scorers miss.

Contracts pinned
----------------

* ``TestTCCatchesHigherOrderRedundancy``: three variables pairwise
  near-independent but jointly XOR-redundant -- TC says high; pairwise
  MI says zero.
* ``TestTCRanksNewInfoOverRedundant``: when x1 is already in the
  support, the engineered ``x1__He2`` column scores LOWER on TC uplift
  than a column carrying an independent secondary signal.
* ``TestAucLiftOnMultiRedundantFixture``: end-to-end -- TC-augmented
  LogReg AUC beats marginal-MI-augmented LogReg on a fixture engineered
  to be heavy with multi-redundant candidates.
* ``TestDefaultDisabledByteIdentical``: master switch OFF leaves
  ``hybrid_orth_features_`` empty.
* ``TestPickleAndClone``: ``clone`` preserves ctor params; ``pickle``
  preserves appended features and recipe round-trip.

NEVER xfail.

Consolidated verbatim from test_biz_value_mrmr_layer73.py (per audit finding test_code_quality-16).
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
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


def _import_tc_fe():
    """Lazily import the Layer-73 total-correlation scoring/FE functions."""
    from mlframe.feature_selection.filters._orthogonal_total_correlation_fe import (
        total_correlation,
        score_features_by_tc_uplift,
        hybrid_orth_mi_tc_fe,
        hybrid_orth_mi_tc_fe_with_recipes,
    )

    return (
        total_correlation,
        score_features_by_tc_uplift,
        hybrid_orth_mi_tc_fe,
        hybrid_orth_mi_tc_fe_with_recipes,
    )


def _import_plug_in_fe():
    """Lazily import the Layer-21 plug-in marginal-MI univariate FE functions."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
        score_features_by_mi_uplift,
        hybrid_orth_mi_fe,
    )

    return (
        generate_univariate_basis_features,
        score_features_by_mi_uplift,
        hybrid_orth_mi_fe,
    )


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_xor_triple(seed: int, n: int = 2000):
    """Three-variable XOR-redundant fixture.

    Construction:

    * ``a`` and ``b`` are independent fair coin flips (0/1 columns).
    * ``c = a XOR b`` -- pairwise INDEPENDENT of both ``a`` and ``b``
      (each pair is uniform on {0,1}^2) yet JOINTLY fully determined
      (any two of {a, b, c} determine the third).
    * ``y`` carries a separate signal so the fixture is comparable to
      the other Layer-7x fixtures (independent of (a, b, c)).

    TC of (a, b, c) is exactly ``log 2`` (one bit), whereas every
    pairwise MI ``I(a; b)``, ``I(a; c)``, ``I(b; c)`` is zero.
    """
    rng = np.random.default_rng(int(seed))
    a = rng.integers(0, 2, size=n).astype(np.float64)
    b = rng.integers(0, 2, size=n).astype(np.float64)
    c = (a.astype(int) ^ b.astype(int)).astype(np.float64)
    y_drv = rng.standard_normal(n)
    y = (y_drv > 0.0).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    return X, pd.Series(y, name="y")


def _build_redundant_multi(seed: int, n: int = 2000):
    """Multi-redundant-candidate fixture.

    * ``x1`` carries the primary quadratic signal (``y depends on x1^2``).
    * ``x_dup_a``, ``x_dup_b``, ``x_dup_c`` are NEAR-COPIES of ``x1``.
      Their He_2 expansions all score high on marginal MI with y -- a
      marginal-MI top-K will fill with duplicates.
    * ``x2`` carries an INDEPENDENT secondary quadratic signal.

    TC-uplift against ``current_support = [x1]`` collapses the duplicate
    He_2 columns (jointly redundant with x1) and surfaces ``x2__He2`` --
    the column with genuine new joint information given the support.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x_dup_a = x1 + 0.05 * rng.standard_normal(n)
    x_dup_b = x1 + 0.05 * rng.standard_normal(n)
    x_dup_c = x1 + 0.05 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x_dup_a": x_dup_a,
            "x_dup_b": x_dup_b,
            "x_dup_c": x_dup_c,
            "x2": x2,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        }
    )
    signal = x1**2 + 0.6 * (x2**2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


from tests.feature_selection._biz_val_synth import _build_linear, _quantile_bin_local

# ---------------------------------------------------------------------------
# Contract 1: TC catches higher-order (XOR) redundancy
# ---------------------------------------------------------------------------


class TestTCCatchesHigherOrderRedundancy:
    """The signature property of TC vs pairwise MI: a three-variable XOR
    parity ``c = a XOR b`` is PAIRWISE independent (every ``I(a; b)``,
    ``I(a; c)``, ``I(b; c)`` is zero) yet JOINTLY one-bit redundant.

    TC of the triple must be strictly positive (~ log 2 nats in the
    asymptotic limit) while every pairwise MI is at noise-floor level.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_xor_triple_has_positive_tc_and_zero_pairwise_mi(self, seed):
        """XOR-triple TC is materially positive and dominates the sum of pairwise MIs, which sit at noise floor."""
        total_correlation, _, _, _ = _import_tc_fe()
        X, _y = _build_xor_triple(seed, n=4000)
        cols = X.to_numpy()
        tc = total_correlation(cols, n_bins=10)
        # Compute pairwise MI on the SAME binned columns the local
        # binner produces, so the comparison is apples-to-apples.
        bins = [_quantile_bin_local(cols[:, j], nbins=10) for j in range(cols.shape[1])]
        mi_pairs = [float(mutual_info_score(bins[i], bins[j])) for i in range(3) for j in range(i + 1, 3)]
        max_pairwise = float(max(mi_pairs))
        # TC must be CLEARLY above the noise floor and above max pairwise:
        # log 2 ~ 0.693 in nats; allow >= 0.3 to absorb finite-sample drift
        # plus binning of an already-binary col into 10 quantile bins.
        assert tc >= 0.3, f"seed={seed}: TC({tc:.4f}) of XOR triple is at noise floor; higher-order detection contract violated."
        # Every pairwise MI on independent fair bits is at noise floor
        # (~ 1/n nats); a healthy gate is ``tc >> max_pairwise * 3``.
        assert tc >= 3.0 * max_pairwise + 0.1, (
            f"seed={seed}: TC({tc:.4f}) not clearly above the "
            f"sum-of-pairwise-MI floor (max_pairwise={max_pairwise:.4f}); "
            f"the higher-order-only signal must dominate."
        )


# ---------------------------------------------------------------------------
# Contract 2: TC ranks new info over redundant info
# ---------------------------------------------------------------------------


class TestTCRanksNewInfoOverRedundant:
    """When ``x1`` is already in the support, the engineered ``x1__He2``
    column carries no new joint information (it is a deterministic
    function of x1). The TC uplift ``delta_tc`` for ``x1__He2`` must be
    LOWER than the delta_tc for an engineered column on a TRULY
    independent source (``x2__He2``).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_tc_delta_lower_for_redundant_than_for_novel(self, seed):
        """TC delta for the novel x2__He2 column exceeds the TC delta for the redundant x1__He2 column."""
        gen, _, _ = _import_plug_in_fe()
        _, score_features_by_tc_uplift, _, _ = _import_tc_fe()
        X, y = _build_redundant_multi(seed, n=2000)
        engineered = gen(X, degrees=(2,), basis="hermite")
        scores = score_features_by_tc_uplift(
            X,
            engineered,
            y.to_numpy(),
            current_support=X[["x1"]],
            n_bins=10,
        )
        s_map = dict(zip(scores["engineered_col"], scores["engineered_mi"]))
        # Pick the He_2 columns we care about.
        eng_x1 = [c for c in s_map if c.startswith("x1__He")]
        eng_x2 = [c for c in s_map if c.startswith("x2__He")]
        assert eng_x1, f"seed={seed}: no x1__He columns ranked: {list(s_map)}"
        assert eng_x2, f"seed={seed}: no x2__He columns ranked: {list(s_map)}"
        delta_x1 = float(s_map[eng_x1[0]])
        delta_x2 = float(s_map[eng_x2[0]])
        assert delta_x2 > delta_x1, (
            f"seed={seed}: TC delta for novel x2__He2 ({delta_x2:.4f}) "
            f"not strictly above TC delta for redundant x1__He2 "
            f"({delta_x1:.4f}); the new-info ranking contract is violated."
        )


# ---------------------------------------------------------------------------
# Contract 3: AUC lift on multi-redundant fixture
# ---------------------------------------------------------------------------


class TestAucLiftOnMultiRedundantFixture:
    """End-to-end biz_value: appending TC-selected orth-poly columns
    lifts downstream LogReg AUC over the marginal-MI-augmented baseline
    on a fixture engineered to be heavy with multi-redundant candidates.
    """

    def test_tc_augmented_logreg_auc_beats_marginal_mi(self):
        """TC-augmented LogReg AUC beats marginal-MI-augmented AUC by >= 0.005 on the redundant fixture."""
        gen, _, hybrid_marginal = _import_plug_in_fe()
        _, _, _, hybrid_tc_with_recipes = _import_tc_fe()
        aucs_marg, aucs_tc = [], []
        for s in (1, 7, 13, 42, 101, 202, 303, 404):
            X, y = _build_redundant_multi(s, n=2000)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=s,
                stratify=y,
            )
            # marginal-MI augmentation baseline.
            X_marg_tr, _ = hybrid_marginal(
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
            # TC augmentation: condition on x1 as the already-picked
            # support so the TC uplift suppresses the redundant He_2(x_dup_*)
            # duplicates and surfaces He_2(x2) instead.
            X_tc_tr, _scores, _recipes = hybrid_tc_with_recipes(
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
            tc_added = [c for c in X_tc_tr.columns if c not in X_tr.columns]
            X_tc_te = pd.concat([X_te, eng_te_all[tc_added]], axis=1) if tc_added else X_te
            lr_tc = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
            ).fit(X_tc_tr, y_tr)
            aucs_tc.append(
                roc_auc_score(
                    y_te,
                    lr_tc.predict_proba(X_tc_te)[:, 1],
                )
            )
        marg_mean = float(np.mean(aucs_marg))
        tc_mean = float(np.mean(aucs_tc))
        # TC's higher-order redundancy control lets a non-x1 secondary
        # signal into the top-K instead of duplicating x1 -- worth at
        # least a 0.005 absolute AUC lift averaged over seeds (same floor
        # as Layer 72's analogous biz_value test).
        assert tc_mean > marg_mean + 0.005, (
            f"TC-augmented LogReg AUC mean ({tc_mean:.4f}) not lifted "
            f"by >= 0.005 vs marginal-MI mean ({marg_mean:.4f}); biz_value "
            f"lift claim violated.\nmarg_per_seed={aucs_marg}\n"
            f"tc_per_seed={aucs_tc}"
        )


# ---------------------------------------------------------------------------
# Contract 4: default disabled byte-identical with master
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_tc_enable defaults to False and leaves hybrid_orth_features_ empty."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_tc_columns(self, seed):
        """With the flag left at its False default, no total-correlation columns are appended."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], f"seed={seed}: default fe_hybrid_orth_tc_enable=False should NOT append any engineered columns; got {added}"

    def test_default_ctor_values(self):
        """fe_hybrid_orth_tc_enable defaults to False and fe_hybrid_orth_tc_n_bins defaults to 10."""
        m = _make_mrmr()
        assert m.fe_hybrid_orth_tc_enable is False
        assert m.fe_hybrid_orth_tc_n_bins == 10


# ---------------------------------------------------------------------------
# Contract 5: pickle / clone preserve the ctor + recipes round-trip
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """Total-correlation ctor params and recipes must survive clone/pickle round-trips."""

    def test_clone_preserves_tc_params(self):
        """sklearn clone() copies every fe_hybrid_orth_tc_* ctor param."""
        m = _make_mrmr(
            fe_hybrid_orth_tc_enable=True,
            fe_hybrid_orth_tc_n_bins=15,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_tc_enable", True),
            ("fe_hybrid_orth_tc_n_bins", 15),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got {getattr(m2, name)}"

    def test_pickle_roundtrip_preserves_tc_recipes(self):
        """A pickle round-trip preserves feature names, appended columns, and every orth_univariate recipe field."""
        X, y = _build_redundant_multi(seed=42, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_tc_enable=True,
            fe_hybrid_orth_tc_n_bins=10,
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
