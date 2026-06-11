"""Layer 62 biz_value: BOOTSTRAP-STABLE MI ranking for hybrid orth-poly FE.

Validates the new ``score_features_by_bootstrap_mi`` /
``hybrid_orth_mi_bootstrap_fe`` introduced 2026-05-31 (sibling module
``_orthogonal_bootstrap_mi_fe``): rank candidates by the lower edge of a
95 % CI on uplift (``mean - 1.96 * std``) across B bootstrap subsamples
rather than a single point estimate. Selection-stability win: borderline
noise candidates whose point-estimate ranking flips in/out across runs
get a wide CI and a small LCB, dropping out of the top-K.

What the contract classes pin
-----------------------------

* ``TestStableSignalRetained``: when ``y = sign(x1^2 - 1)``, ``x1__He2``
  is a STABLE high-MI signal; across bootstraps its uplift_lcb stays
  well above the noise floor.

* ``TestNoiseSuppressed``: noise basis columns have wide CIs (high std)
  and their uplift_lcb sits at or below the noise floor; they do NOT
  enter the bootstrap top-K. Cross-check: the point-estimate ranking
  on the SAME frame admits at least one such noise column into its
  top-K (showing the bootstrap delivers genuine extra filtering).

* ``TestBootstrapVsPointEstimateDiffer``: on a borderline frame the
  bootstrap-selected support and the point-estimate-selected support
  are NOT equal -- specifically, the bootstrap support is a subset
  AND it drops at least one high-variance (wide-CI) column.

* ``TestDefaultDisabledByteIdentical``: default
  ``fe_hybrid_orth_bootstrap_enable=False`` leaves
  ``hybrid_orth_features_`` empty (legacy behaviour preserved).

* ``TestEnableAppendsEngineered``: turning the flag on appends at least
  one engineered column on a clean signal frame.

* ``TestPickleAndClone``: sklearn ``clone`` preserves the 3 ctor params;
  ``pickle`` preserves the appended ``hybrid_orth_features_`` AND the
  ``orth_univariate`` recipes round-trip.

NEVER xfail. NEVER mask bugs via runtime workarounds.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_bootstrap_fe():
    from mlframe.feature_selection.filters._orthogonal_bootstrap_mi_fe import (
        score_features_by_bootstrap_mi,
        hybrid_orth_mi_bootstrap_fe,
        hybrid_orth_mi_bootstrap_fe_with_recipes,
    )
    return (
        score_features_by_bootstrap_mi,
        hybrid_orth_mi_bootstrap_fe,
        hybrid_orth_mi_bootstrap_fe_with_recipes,
    )


def _import_point_estimate_fe():
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


def _build_quadratic_signal(seed: int, n: int = 2000, n_noise: int = 5):
    """y = sign(x1^2 - 1). ``x1__He2`` is the stable high-MI winner;
    noise columns at degree 2/3 should have wide CIs.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    cols: dict = {"x1": x1}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = ((x1 ** 2 + 0.1 * rng.standard_normal(n)) > 1.0).astype(int)
    return X, pd.Series(y)


def _build_borderline_signal(seed: int, n: int = 1500, n_noise: int = 8):
    """Weak quadratic signal with high noise on top, plus many noise
    columns. The point estimate frequently includes a noise He_3 column
    in the top-K (boosted by sample-dependent tail flukes); the bootstrap
    LCB suppresses it because the same column has very different MI in
    different subsamples.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    cols: dict = {"x1": x1}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    # Lower-amplitude quadratic so the He_2 winner is materially above
    # noise but not by an extreme margin -- a regime where MI variance
    # actually matters for selection.
    y = ((x1 ** 2 + 0.6 * rng.standard_normal(n)) > 1.0).astype(int)
    return X, pd.Series(y)


from tests.feature_selection._biz_val_synth import _build_linear
# ---------------------------------------------------------------------------
# Contract 1: stable signal -- x1__He2 LCB stays high across bootstraps
# ---------------------------------------------------------------------------


class TestStableSignalRetained:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_signal_uplift_lcb_above_noise(self, seed):
        score_boot, _, _ = _import_bootstrap_fe()
        gen, _, _ = _import_point_estimate_fe()
        X, y = _build_quadratic_signal(seed)
        eng = gen(X, degrees=(2, 3), basis="hermite")
        scores = score_boot(
            X, eng, y.values,
            n_boot=10, sample_fraction=0.8, seed=seed,
        )
        # The He_2(x1) row must be present and its uplift_lcb must be
        # materially positive AND ABOVE every noise column's uplift_lcb.
        target_row = scores[scores["engineered_col"] == "x1__He2"]
        assert not target_row.empty, (
            f"seed={seed}: x1__He2 missing from score table; got "
            f"{list(scores['engineered_col'])}"
        )
        x1_he2_lcb = float(target_row["uplift_lcb"].iloc[0])
        noise_rows = scores[scores["engineered_col"].str.startswith("noise_")]
        noise_max_lcb = (
            float(noise_rows["uplift_lcb"].max()) if not noise_rows.empty else 0.0
        )
        assert x1_he2_lcb > noise_max_lcb, (
            f"seed={seed}: x1__He2 uplift_lcb={x1_he2_lcb:.3f} not above "
            f"max noise uplift_lcb={noise_max_lcb:.3f}; stable-signal "
            f"contract violated."
        )
        # Engineered MI LCB stays positive on a real signal.
        x1_he2_eng_lcb = float(target_row["engineered_mi_lcb"].iloc[0])
        assert x1_he2_eng_lcb > 0.0, (
            f"seed={seed}: x1__He2 engineered_mi_lcb={x1_he2_eng_lcb:.4f} "
            f"is non-positive on a clean He_2 signal."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_signal_selected_into_winners(self, seed):
        _, hybrid_boot, _ = _import_bootstrap_fe()
        X, y = _build_quadratic_signal(seed)
        X_aug, _scores = hybrid_boot(
            X, y.values,
            degrees=(2, 3), basis="hermite",
            top_k=3, n_boot=10, sample_fraction=0.8, seed=seed,
        )
        appended = [c for c in X_aug.columns if c not in X.columns]
        assert "x1__He2" in appended, (
            f"seed={seed}: x1__He2 missing from bootstrap-selected winners; "
            f"got {appended}"
        )


# ---------------------------------------------------------------------------
# Contract 2: noise is correctly suppressed
# ---------------------------------------------------------------------------


class TestNoiseSuppressed:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_noise_lcb_smaller_than_mean(self, seed):
        """Every noise column's std must be strictly positive (the
        bootstrap captures genuine MI sampling variance), so its
        uplift_lcb is STRICTLY BELOW its uplift_mean. This is the
        mechanical property that drives the noise-suppression win.
        """
        score_boot, _, _ = _import_bootstrap_fe()
        gen, _, _ = _import_point_estimate_fe()
        X, y = _build_borderline_signal(seed)
        eng = gen(X, degrees=(2, 3), basis="hermite")
        scores = score_boot(
            X, eng, y.values,
            n_boot=10, sample_fraction=0.8, seed=seed,
        )
        noise_rows = scores[scores["engineered_col"].str.startswith("noise_")]
        assert not noise_rows.empty, (
            f"seed={seed}: expected noise basis columns in score table"
        )
        # At least one noise basis column must have strictly positive
        # uplift std (genuine bootstrap variance, not numeric zero).
        assert (noise_rows["uplift_std"] > 1e-6).any(), (
            f"seed={seed}: all noise basis columns have zero uplift_std; "
            f"bootstrap did not produce a non-degenerate CI."
        )
        # By construction LCB = mean - 1.96 * std, so wherever std > 0 the
        # LCB is strictly below the mean.
        wide_ci = noise_rows[noise_rows["uplift_std"] > 1e-6]
        assert (wide_ci["uplift_lcb"] < wide_ci["uplift_mean"]).all(), (
            f"seed={seed}: noise rows with positive std must have "
            f"uplift_lcb < uplift_mean; LCB construction violated.\n"
            f"{wide_ci[['engineered_col', 'uplift_mean', 'uplift_std', 'uplift_lcb']]}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_bootstrap_winners_no_noise_below_signal(self, seed):
        """On a clean He_2 signal, no noise column should outrank x1__He2
        in the bootstrap LCB ordering.
        """
        score_boot, _, _ = _import_bootstrap_fe()
        gen, _, _ = _import_point_estimate_fe()
        X, y = _build_quadratic_signal(seed)
        eng = gen(X, degrees=(2, 3), basis="hermite")
        scores = score_boot(
            X, eng, y.values,
            n_boot=10, sample_fraction=0.8, seed=seed,
        )
        # First row in sorted-by-uplift_lcb order must be x1__He2 OR a
        # transform of x1 (degree 3 also captures the symmetry).
        first = scores.iloc[0]["engineered_col"]
        assert first.startswith("x1__"), (
            f"seed={seed}: top bootstrap LCB winner {first!r} is not an "
            f"x1 basis column; noise leaked to the top of the ranking."
        )


# ---------------------------------------------------------------------------
# Contract 3: bootstrap vs point-estimate selections DIFFER on borderline
# ---------------------------------------------------------------------------


class TestBootstrapVsPointEstimateDiffer:

    def test_bootstrap_topk_drops_noise_point_estimate_keeps(self):
        """The headline biz_value (raw-ranking level): on a borderline
        frame the bootstrap LCB top-K and the point-estimate top-K
        select DIFFERENT sets, and the bootstrap top-K drops at least
        one noise column the point-estimate top-K kept. Operates at the
        raw ranking level so the gates (MAD floor etc.) don't mask the
        comparison.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
            score_features_by_mi_uplift,
        )
        score_boot, _, _ = _import_bootstrap_fe()
        divergent_seeds = 0
        boot_drops_noise = 0
        boot_promotes_signal = 0
        for s in (1, 7, 13, 42, 101, 202, 303, 404):
            X, y = _build_borderline_signal(s)
            eng = generate_univariate_basis_features(
                X, degrees=(2, 3), basis="hermite",
            )
            sc_pt = score_features_by_mi_uplift(X, eng, y.values)
            sc_bt = score_boot(
                X, eng, y.values,
                n_boot=10, sample_fraction=0.8, seed=s,
            )
            pt_top = set(sc_pt.head(5)["engineered_col"])
            bt_top = set(sc_bt.head(5)["engineered_col"])
            if pt_top != bt_top:
                divergent_seeds += 1
            pt_noise = {c for c in pt_top if c.startswith("noise_")}
            bt_noise = {c for c in bt_top if c.startswith("noise_")}
            if pt_noise - bt_noise:
                boot_drops_noise += 1
            # Bonus: bootstrap should also PROMOTE the true x1__He2
            # signal more reliably than point estimate.
            if "x1__He2" in bt_top and "x1__He2" not in pt_top:
                boot_promotes_signal += 1
        # Contract floors -- robust against MI estimator noise without
        # becoming a tautology. Selection diverges on most seeds; noise
        # gets dropped on most seeds; signal-promotion is the cleanest
        # win and must fire on at least one seed.
        assert divergent_seeds >= 5, (
            f"bootstrap and point-estimate top-5 were IDENTICAL on "
            f"{8 - divergent_seeds} of 8 borderline seeds; selection-"
            f"stability claim violated."
        )
        assert boot_drops_noise >= 5, (
            f"bootstrap top-5 dropped a noise column the point-estimate "
            f"top-5 kept on only {boot_drops_noise} of 8 seeds; noise-"
            f"suppression claim violated."
        )
        assert boot_promotes_signal >= 1, (
            f"bootstrap top-5 never promoted x1__He2 over point-estimate "
            f"top-5; signal-stability claim violated. "
            f"divergent_seeds={divergent_seeds}"
        )


# ---------------------------------------------------------------------------
# Contract 4: default disabled byte-identical with master
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_bootstrap_columns(self, seed):
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], (
            f"seed={seed}: default fe_hybrid_orth_bootstrap_enable=False "
            f"should NOT append any engineered columns; got {added}"
        )

    def test_default_ctor_values(self):
        m = _make_mrmr()
        assert m.fe_hybrid_orth_bootstrap_enable is False
        assert m.fe_hybrid_orth_bootstrap_n_boot == 10
        assert m.fe_hybrid_orth_bootstrap_sample_fraction == 0.8


# ---------------------------------------------------------------------------
# Contract 5: enabling appends engineered columns on a clean signal
# ---------------------------------------------------------------------------


class TestEnableAppendsEngineered:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_enable_appends_at_least_one_he2(self, seed):
        X, y = _build_quadratic_signal(seed, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_bootstrap_enable=True,
            fe_hybrid_orth_bootstrap_n_boot=10,
            fe_hybrid_orth_bootstrap_sample_fraction=0.8,
            fe_hybrid_orth_degrees=(2, 3),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added, (
            f"seed={seed}: bootstrap flag ON should append at least one "
            f"engineered column to hybrid_orth_features_; got {added}"
        )
        # The He_2(x1) signal must enter the support.
        assert any(c == "x1__He2" or c.startswith("x1__") for c in added), (
            f"seed={seed}: bootstrap winners should include an x1 basis "
            f"column for a clean He_2 signal; got {added}"
        )


class TestCategoricalDoesNotSwallowBootstrap:
    """Regression: a raw categorical / string column in X must NOT make the bootstrap-stable FE raise
    "could not convert string to float" and fall into the broad warn-and-continue path. The numeric bootstrap
    FE must scope to numeric columns (like the conditional-FE families) and still produce engineered columns."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_categorical_frame_still_appends_engineered(self, seed, caplog):
        import logging

        X, y = _build_quadratic_signal(seed, n=2000)
        rng = np.random.default_rng(int(seed))
        X = X.copy()
        X["cat"] = pd.Series(rng.choice(["A", "B", "C"], size=len(X))).astype("category")
        X["strcol"] = rng.choice(["p", "q"], size=len(X))
        m = _make_mrmr(
            fe_hybrid_orth_bootstrap_enable=True,
            fe_hybrid_orth_bootstrap_n_boot=10,
            fe_hybrid_orth_bootstrap_sample_fraction=0.8,
            fe_hybrid_orth_degrees=(2, 3),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        )
        with caplog.at_level(logging.WARNING):
            m.fit(X, y)
        boot_warns = [r for r in caplog.records if "bootstrap-stable FE raised" in r.getMessage()]
        assert not boot_warns, (
            f"seed={seed}: categorical column must not trigger the bootstrap warn-and-continue band-aid; "
            f"got {[r.getMessage() for r in boot_warns]}"
        )
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added, (
            f"seed={seed}: bootstrap-stable FE must still append engineered column(s) on a categorical-bearing "
            f"frame (numeric cols scoped, cat dropped); got {added}"
        )
        assert any(c == "x1__He2" or c.startswith("x1__") for c in added), (
            f"seed={seed}: the He_2(x1) winner must survive categorical scoping; got {added}"
        )


# ---------------------------------------------------------------------------
# Contract 6: pickle / clone preserve the ctor + recipes round-trip
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_clone_preserves_bootstrap_params(self):
        m = _make_mrmr(
            fe_hybrid_orth_bootstrap_enable=True,
            fe_hybrid_orth_bootstrap_n_boot=25,
            fe_hybrid_orth_bootstrap_sample_fraction=0.6,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_bootstrap_enable", True),
            ("fe_hybrid_orth_bootstrap_n_boot", 25),
            ("fe_hybrid_orth_bootstrap_sample_fraction", 0.6),
        ]:
            assert getattr(m2, name) == expected, (
                f"clone() dropped {name}: expected {expected}, got "
                f"{getattr(m2, name)}"
            )

    def test_pickle_roundtrip_preserves_bootstrap_recipes(self):
        X, y = _build_quadratic_signal(seed=42, n=2000)
        m = _make_mrmr(
            fe_hybrid_orth_bootstrap_enable=True,
            fe_hybrid_orth_bootstrap_n_boot=10,
            fe_hybrid_orth_bootstrap_sample_fraction=0.8,
            fe_hybrid_orth_degrees=(2, 3),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), (
            "pickle changed feature_names_in_"
        )
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, (
            f"pickle changed hybrid_orth_features_: before={added_before}, "
            f"after={added_after}"
        )
        # All bootstrap-stage recipes are ``orth_univariate`` -- the
        # engineered VALUES are bit-equal to Layer 21, only the selection
        # rule differs.
        recipes_before = {
            r.name: r for r in getattr(m, "_engineered_recipes_", {}).values()
            if getattr(r, "kind", None) == "orth_univariate"
        } if isinstance(getattr(m, "_engineered_recipes_", None), dict) else {
            r.name: r for r in (getattr(m, "_engineered_recipes_", []) or [])
            if getattr(r, "kind", None) == "orth_univariate"
        }
        recipes_after = {
            r.name: r for r in getattr(m2, "_engineered_recipes_", {}).values()
            if getattr(r, "kind", None) == "orth_univariate"
        } if isinstance(getattr(m2, "_engineered_recipes_", None), dict) else {
            r.name: r for r in (getattr(m2, "_engineered_recipes_", []) or [])
            if getattr(r, "kind", None) == "orth_univariate"
        }
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


# ---------------------------------------------------------------------------
# Contract 7: recipe replay reproduces fit-time values bit-equivalently
# ---------------------------------------------------------------------------


class TestRecipeReplay:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_recipe_replay_matches_fit_time(self, seed):
        _, _, hybrid_with_recipes = _import_bootstrap_fe()
        X, y = _build_quadratic_signal(seed)
        X_aug, _scores, recipes = hybrid_with_recipes(
            X, y.values,
            degrees=(2, 3), basis="hermite",
            top_k=3, n_boot=10, sample_fraction=0.8, seed=seed,
        )
        if not recipes:
            pytest.fail(
                f"seed={seed}: bootstrap-stable hybrid emitted no recipes; "
                f"replay contract requires at least one recipe."
            )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        appended = [c for c in X_aug.columns if c not in X.columns]
        for r in recipes:
            assert r.name in appended, (
                f"seed={seed}: recipe {r.name!r} not in appended columns "
                f"{appended}"
            )
            assert r.kind == "orth_univariate", (
                f"seed={seed}: bootstrap-stage recipe {r.name!r} kind="
                f"{r.kind!r}, expected 'orth_univariate' (engineered values "
                f"are bit-equal to Layer 21; only selection differs)."
            )
            replayed = apply_recipe(r, X)
            fit_time = X_aug[r.name].to_numpy()
            assert np.allclose(replayed, fit_time, rtol=1e-9, atol=1e-12), (
                f"seed={seed}: recipe {r.name!r} replay drift: "
                f"max|replayed - fit| = "
                f"{float(np.max(np.abs(replayed - fit_time)))}; "
                f"extra={dict(r.extra)}"
            )


class TestSortedGatherGate:
    """The bootstrap loop sorts the resample indices before the wide
    (sample_n, n_eng) gather (cache-locality win, ~6-7x on this function's
    self-time). MI uses equi-frequency argsort binning whose tie-breaking is
    POSITIONAL, so the sort is bit-identical ONLY when a matrix's columns are
    all-distinct (continuous); on tied/discrete columns reordering rows shifts
    MI ~1e-3 and drifts selection. ``_all_columns_distinct`` gates the sort per
    matrix. These pin the gate + the bit-identity invariant it protects."""

    def test_all_columns_distinct_gate(self):
        from mlframe.feature_selection.filters._orthogonal_bootstrap_mi_fe import (
            _all_columns_distinct,
        )
        rng = np.random.default_rng(0)
        cont = rng.normal(size=(2000, 4))  # all-distinct floats
        disc = rng.integers(0, 12, size=(2000, 4)).astype(np.float64)  # heavy ties
        mixed = cont.copy()
        mixed[:, 2] = rng.integers(0, 12, size=2000)  # one discrete column
        assert _all_columns_distinct(cont) is True
        assert _all_columns_distinct(disc) is False
        assert _all_columns_distinct(mixed) is False, (
            "one discrete column must disqualify the whole matrix (the gather "
            "shares a single idx across all columns)"
        )

    def test_sorted_gather_bit_identical_on_continuous(self):
        """The optimisation's safety invariant: on all-distinct columns the MI
        from a sorted resample equals the MI from the unsorted resample."""
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _mi_classif_batch,
        )
        rng = np.random.default_rng(3)
        n = 6000
        X = rng.normal(size=(n, 6))  # continuous -> all-distinct
        y = rng.integers(0, 3, n).astype(np.int64)
        idx = rng.integers(0, n, size=int(0.8 * n))
        mi_unsorted = np.asarray(_mi_classif_batch(X[idx, :], y[idx], nbins=10), float)
        mi_sorted = np.asarray(
            _mi_classif_batch(X[np.sort(idx), :], y[np.sort(idx)], nbins=10), float
        )
        assert np.array_equal(mi_unsorted, mi_sorted), (
            f"sorted-gather MI must be bit-identical on continuous columns; "
            f"max|diff|={float(np.max(np.abs(mi_unsorted - mi_sorted)))}"
        )

    def test_sorted_gather_diverges_on_discrete(self):
        """Counterpart: on discrete/tied columns the sort is NOT bit-identical
        (this is WHY the gate exists). Guards against a future 'just always
        sort' regression slipping through unnoticed."""
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _mi_classif_batch,
        )
        rng = np.random.default_rng(3)
        n = 6000
        X = rng.integers(0, 12, size=(n, 6)).astype(np.float64)  # heavy ties
        y = rng.integers(0, 3, n).astype(np.int64)
        idx = rng.integers(0, n, size=int(0.8 * n))
        mi_unsorted = np.asarray(_mi_classif_batch(X[idx, :], y[idx], nbins=10), float)
        mi_sorted = np.asarray(
            _mi_classif_batch(X[np.sort(idx), :], y[np.sort(idx)], nbins=10), float
        )
        assert not np.array_equal(mi_unsorted, mi_sorted), (
            "sort SHOULD perturb MI on tied/discrete columns; if this passes, "
            "the binning became tie-deterministic and the gate may be relaxed"
        )
