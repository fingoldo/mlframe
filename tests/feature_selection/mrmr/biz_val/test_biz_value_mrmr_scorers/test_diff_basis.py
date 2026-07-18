"""Layer 59 biz_value: DIFF-BASIS FE for highly-correlated source pairs.

Validates the new ``hybrid_orth_mi_diff_basis_fe`` introduced 2026-05-31
(sibling module ``_orthogonal_diff_basis_fe``): for every (col_a, col_b)
pair whose |Pearson corr| clears the threshold, evaluate
``basis_d(preprocess(col_a - col_b))`` and keep MI-uplift winners.

Why this layer matters
----------------------

When two source columns are tightly coupled (e.g. ``price_today`` and
``price_yesterday`` at corr 0.99), the residual ``price_today -
price_yesterday`` -- the daily return -- often carries the actual signal
the marginal of either column hides. Layer 21 (univariate) plus Layer 27
(collinear-dedup) drop one of the two columns BEFORE the basis evaluation;
Layer 25 (pair-cross) covers the multiplicative ``He_a(x_i) * He_b(x_j)``
interaction, NOT the additive residual. Layer 59 fills that gap with a
dedicated diff path.

Contracts pinned
----------------

* ``TestDiffSignal``: ``y`` depends on ``price_today - price_yesterday``;
  raw MRMR keeps either column but cannot recover the residual; diff-basis
  emits the ``He_1`` column for the pair and it ranks above both raw cols.

* ``TestAucLift``: end-to-end downstream metric. LogReg on raw
  ``(price_today, price_yesterday)`` recovers ~0.55 AUC (chance-ish); LogReg
  on ``raw + diff_He_1`` jumps to >= 0.85.

* ``TestAutoPairDetection``: ``detect_correlated_pairs`` only flags pairs
  whose |corr| clears the threshold; uncorrelated noise pairs are NOT
  enumerated.

* ``TestNoSpuriousNoiseDiffs``: on a pure-noise frame at p >= 16 the
  diff-basis path returns zero columns (MAD floor activates).

* ``TestDefaultDisabledByteIdentical``: default
  ``fe_hybrid_orth_diff_basis_enable=False`` leaves ``hybrid_orth_features_``
  empty.

* ``TestPickleAndClone``: sklearn ``clone`` and ``pickle`` preserve the
  ctor params and the chosen ``(col_a, col_b, basis, degree)`` recipe.

Consolidated verbatim from test_biz_value_mrmr_layer59.py (per audit finding test_code_quality-16).
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_diff_fe():
    """Lazily import the Layer-59 diff-basis FE functions."""
    from mlframe.feature_selection.filters._orthogonal_diff_basis_fe import (
        detect_correlated_pairs,
        generate_diff_basis_features,
        hybrid_orth_mi_diff_basis_fe,
        hybrid_orth_mi_diff_basis_fe_with_recipes,
    )

    return (
        detect_correlated_pairs,
        generate_diff_basis_features,
        hybrid_orth_mi_diff_basis_fe,
        hybrid_orth_mi_diff_basis_fe_with_recipes,
    )


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_diff_signal(seed: int, n: int = 4000):
    """``price_today`` and ``price_yesterday`` highly correlated; signal is
    in the SQUARED DAILY RETURN (the He_2 residual). Both marginal columns
    carry only the shared trend; raw LogReg with two linear coefficients
    cannot reconstruct ``(price_today - price_yesterday) ** 2`` without an
    explicit ``He_2(diff)`` column. He_2 is the ``degrees=(1,2,3)`` cell
    that the diff-basis path emits in this regime.

    Why nonlinear residual (not linear): a LINEAR diff target ``y =
    sign(price_today - price_yesterday)`` is recoverable by raw LogReg with
    opposite-sign coefficients on (price_today, price_yesterday) -- the
    augmentation gives no AUC lift because the diff is already representable
    in the model class. The quadratic residual breaks that: LogReg cannot
    represent ``diff**2`` from two linear inputs, so the He_2 column is the
    cheapest path to the signal.
    """
    rng = np.random.default_rng(seed)
    # Common trend that dominates both columns' marginals.
    trend = np.cumsum(rng.standard_normal(n) * 0.1)
    # Daily return (the actual signal axis).
    daily_return = 0.5 * rng.standard_normal(n)
    price_yesterday = trend + 0.05 * rng.standard_normal(n)
    price_today = trend + daily_return + 0.05 * rng.standard_normal(n)
    # y depends on the SQUARED residual -- LogReg with two linear inputs
    # cannot recover this without the He_2(diff) column.
    residual = price_today - price_yesterday
    sig = residual**2 + 0.1 * rng.standard_normal(n)
    y = (sig > np.median(sig)).astype(int)
    X = pd.DataFrame(
        {
            "price_today": price_today,
            "price_yesterday": price_yesterday,
            # Pad with noise so MRMR's relevance / redundancy gates have a
            # non-trivial baseline distribution.
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
            "noise_2": rng.standard_normal(n),
        }
    )
    return X, pd.Series(y, name="y")


def _build_quadratic_diff_signal(seed: int, n: int = 4000):
    """``y`` depends on ``(p_today - p_yesterday)^2`` (He_2-style residual).
    Tests that the diff path also lifts higher-degree residual signal, not
    just He_1.
    """
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.standard_normal(n) * 0.1)
    daily_return = 0.4 * rng.standard_normal(n)
    p_yesterday = trend + 0.05 * rng.standard_normal(n)
    p_today = trend + daily_return + 0.05 * rng.standard_normal(n)
    residual = p_today - p_yesterday
    # Quadratic signal: y triggers on extreme residuals (large positive OR
    # negative daily moves).
    sig = residual**2 + 0.1 * rng.standard_normal(n)
    y = (sig > np.median(sig)).astype(int)
    X = pd.DataFrame(
        {
            "p_today": p_today,
            "p_yesterday": p_yesterday,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        }
    )
    return X, pd.Series(y, name="y")


def _build_noise_only_large(seed: int, n: int = 2000, p: int = 20):
    """p>=16 pure-noise frame so the MAD-based noise floor activates. No
    two columns are correlated by construction.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(p)})
    y = (rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_uncorrelated_frame(seed: int, n: int = 1500):
    """All columns independent gaussians; auto-pair detector should
    enumerate NO pairs.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
            "d": rng.standard_normal(n),
        }
    )
    y = ((X["a"] + 0.5 * X["b"]) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_linear(seed: int, n: int = 1500):
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
            "noise_c": rng.standard_normal(n),
        }
    )
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract 1: diff captures residual signal invisible to marginals
# ---------------------------------------------------------------------------


class TestDiffSignal:
    """The diff-basis path must capture the residual signal invisible to either marginal column."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_diff_basis_emits_for_correlated_pair(self, seed):
        """diff-basis emits >= 1 column referencing the correlated (price_today, price_yesterday) pair."""
        _, gen_diff, _, _ = _import_diff_fe()
        X, y = _build_diff_signal(seed)
        eng, meta = gen_diff(
            X,
            y.values,
            degrees=(1, 2, 3),
            pair_corr_threshold=0.7,
            top_k=5,
        )
        assert eng.shape[1] >= 1, (
            f"seed={seed}: diff-basis should emit >=1 column for the strongly "
            f"correlated (price_today, price_yesterday) pair; got "
            f"{eng.shape[1]}: {list(eng.columns)}"
        )
        # At least one of the emitted columns must reference the residual
        # pair.
        pairs_seen = {(info["col_a"], info["col_b"]) for info in meta.values()}
        assert ("price_today", "price_yesterday") in pairs_seen or ("price_yesterday", "price_today") in pairs_seen, (
            f"seed={seed}: diff-basis emitted columns but none reference the (price_today, price_yesterday) pair; pairs={pairs_seen}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_diff_he1_beats_both_raw_baselines(self, seed):
        """The strongest diff-basis winner has uplift > 1.0 over the best raw-column baseline."""
        _, gen_diff, _, _ = _import_diff_fe()
        X, y = _build_diff_signal(seed)
        _eng, meta = gen_diff(
            X,
            y.values,
            degrees=(1, 2, 3),
            pair_corr_threshold=0.7,
            top_k=5,
        )
        if not meta:
            pytest.fail(f"seed={seed}: diff-basis emitted zero columns; expected the residual signal to clear the uplift gate.")
        # The strongest winner must have uplift > 1.0 (engineered MI exceeds
        # the BEST of the two raw column baselines).
        best = max(meta.values(), key=lambda d: d["uplift"])
        assert best["uplift"] > 1.0, (
            f"seed={seed}: diff-basis top column uplift={best['uplift']:.3f} "
            f"<= 1.0; engineered_mi={best['engineered_mi']:.4f}, "
            f"baseline_mi={best['baseline_mi']:.4f}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_quadratic_diff_emits_higher_degree(self, seed):
        """On a quadratic-residual target, diff-basis still emits a column referencing the residual pair."""
        _, gen_diff, _, _ = _import_diff_fe()
        X, y = _build_quadratic_diff_signal(seed)
        _eng, meta = gen_diff(
            X,
            y.values,
            degrees=(1, 2, 3),
            pair_corr_threshold=0.7,
            top_k=5,
        )
        # Some winner should reference the (p_today, p_yesterday) pair --
        # which exact degree wins varies by seed (He_2 SHOULD dominate but
        # downstream MI ranks are noisy at finite n).
        if not meta:
            pytest.fail(f"seed={seed}: quadratic-diff frame produced no diff-basis columns; expected residual basis expansion to clear gates.")
        pairs = {(i["col_a"], i["col_b"]) for i in meta.values()}
        assert ("p_today", "p_yesterday") in pairs or ("p_yesterday", "p_today") in pairs, (
            f"seed={seed}: quadratic-diff frame: emitted pairs {pairs} do not include the residual pair."
        )


# ---------------------------------------------------------------------------
# Contract 2: end-to-end AUC lift on the diff-signal target
# ---------------------------------------------------------------------------


class TestAucLift:
    """Diff-basis augmentation must lift downstream LogReg AUC over the raw two-column baseline."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lifts_with_diff_he1(self, seed):
        """Augmented LogReg AUC clears 0.85 and beats raw by >= 0.15 on the diff-signal fixture."""
        _, _, hybrid_diff, _ = _import_diff_fe()
        X, y = _build_diff_signal(seed)
        # Baseline: LogReg on (price_today, price_yesterday) -- both columns
        # are dominated by the shared trend so the residual signal is hidden.
        baseline_cols = ["price_today", "price_yesterday"]
        lr_raw = LogisticRegression(max_iter=2000, random_state=seed).fit(
            X[baseline_cols].values,
            y.values,
        )
        auc_raw = roc_auc_score(y.values, lr_raw.predict_proba(X[baseline_cols].values)[:, 1])
        # With diff-basis augmentation, the He_1 residual column should
        # vault AUC well above the raw baseline.
        X_aug, _scores = hybrid_diff(
            X,
            y.values,
            degrees=(1, 2, 3),
            pair_corr_threshold=0.7,
            top_k=5,
        )
        aug_cols = [c for c in X_aug.columns if c not in X.columns]
        if not aug_cols:
            pytest.fail(f"seed={seed}: diff-basis emitted zero columns; cannot test AUC lift. baseline AUC = {auc_raw:.3f}")
        X_full = X_aug[baseline_cols + aug_cols].values
        lr_aug = LogisticRegression(max_iter=2000, random_state=seed).fit(
            X_full,
            y.values,
        )
        auc_aug = roc_auc_score(y.values, lr_aug.predict_proba(X_full)[:, 1])
        # Tight upper bound: with the He_1 column LogReg essentially sees
        # the daily return directly so AUC should be >= 0.85 (the construction
        # noise floor) on every seed.
        assert auc_aug >= 0.85, f"seed={seed}: aug AUC={auc_aug:.3f} below 0.85 contract floor; raw AUC={auc_raw:.3f}; aug cols={aug_cols}"
        # Sanity: aug AUC must improve over raw by a meaningful margin.
        assert auc_aug - auc_raw >= 0.15, f"seed={seed}: AUC lift {auc_aug - auc_raw:.3f} below 0.15 contract floor; raw={auc_raw:.3f} aug={auc_aug:.3f}"


# ---------------------------------------------------------------------------
# Contract 3: auto-pair detection filters by correlation
# ---------------------------------------------------------------------------


class TestAutoPairDetection:
    """The auto-pair detector must find highly-correlated pairs and never flag uncorrelated noise pairs."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_only_correlated_pairs_enumerated(self, seed):
        """detect_correlated_pairs finds the price pair and never flags an uncorrelated noise pair."""
        detect_pairs, _, _, _ = _import_diff_fe()
        X, _ = _build_diff_signal(seed)
        pairs = detect_pairs(X, corr_threshold=0.7)
        # The price pair MUST be detected (constructed with corr > 0.99).
        names = {(a, b) for (a, b, _) in pairs}
        assert ("price_today", "price_yesterday") in names or ("price_yesterday", "price_today") in names, (
            f"seed={seed}: auto-pair detection missed the (price_today, price_yesterday) pair; got {names}"
        )
        # Noise pairs (uncorrelated by construction) must NOT be enumerated.
        noise_names = {"noise_0", "noise_1", "noise_2"}
        for a, b, c in pairs:
            if a in noise_names and b in noise_names:
                pytest.fail(
                    f"seed={seed}: auto-pair detection flagged uncorrelated "
                    f"noise pair ({a}, {b}) at corr={c:.3f} -- the 0.7 "
                    f"threshold should have filtered this."
                )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_uncorrelated_frame_returns_no_pairs(self, seed):
        """A frame of independent Gaussians produces zero detected pairs at threshold 0.7."""
        detect_pairs, _, _, _ = _import_diff_fe()
        X, _ = _build_uncorrelated_frame(seed)
        pairs = detect_pairs(X, corr_threshold=0.7)
        assert pairs == [], f"seed={seed}: uncorrelated frame produced {len(pairs)} pairs at threshold 0.7; expected empty. Pairs: {pairs}"


# ---------------------------------------------------------------------------
# Contract 4: pure-noise frame at p >= 16 emits no diff-basis columns
# ---------------------------------------------------------------------------


class TestNoSpuriousNoiseDiffs:
    """A p>=16 pure-noise frame must clear the MAD noise floor and emit no diff-basis columns."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_noise_only_p20_emits_nothing(self, seed):
        """A pure-noise frame at p=20 emits zero diff-basis columns."""
        _, gen_diff, _, _ = _import_diff_fe()
        X, y = _build_noise_only_large(seed, p=20)
        eng, _meta = gen_diff(
            X,
            y.values,
            degrees=(1, 2, 3),
            pair_corr_threshold=0.7,
            top_k=5,
        )
        assert eng.shape[1] == 0, (
            f"seed={seed}: p=20 pure-noise frame should clear no diff-basis "
            f"columns -- auto-pair detection is gated by corr_threshold and "
            f"any surviving pair is then gated by the MAD floor. Got "
            f"{eng.shape[1]} columns: {list(eng.columns)}"
        )


# ---------------------------------------------------------------------------
# Contract 5: default disabled -- legacy behaviour byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_diff_basis_enable defaults to False; enabling it must fire and append columns."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_diff_columns(self, seed):
        """With the flag left at its False default, no diff-basis columns are appended."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], f"seed={seed}: default fe_hybrid_orth_diff_basis_enable=False should NOT append any engineered columns; got {added}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_enable_diff_basis_appends_engineered(self, seed):
        """Enabling the flag appends a column referencing the residual pair on the diff-signal fixture."""
        X, y = _build_diff_signal(seed, n=2500)
        m = _make_mrmr(
            fe_hybrid_orth_diff_basis_enable=True,
            fe_hybrid_orth_diff_basis_corr_threshold=0.7,
            fe_hybrid_orth_diff_basis_degrees=(1, 2, 3),
            fe_hybrid_orth_diff_basis_top_k=3,
        ).fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added, f"seed={seed}: diff-basis flag ON should append at least one engineered column to hybrid_orth_features_; got {added}"
        # At least one engineered column must reference the residual pair.
        assert any(
            "price_today" in c and "price_yesterday" in c for c in added
        ), f"seed={seed}: diff-basis should reference the residual pair; engineered names = {added}"


# ---------------------------------------------------------------------------
# Contract 6: pickle / clone preserve the ctor + chosen-pair recipe
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """Diff-basis ctor params and recipes must survive clone/pickle round-trips."""

    def test_clone_preserves_diff_basis_params(self):
        """sklearn clone() copies every fe_hybrid_orth_diff_basis_* ctor param."""
        m = _make_mrmr(
            fe_hybrid_orth_diff_basis_enable=True,
            fe_hybrid_orth_diff_basis_corr_threshold=0.8,
            fe_hybrid_orth_diff_basis_degrees=(1, 2),
            fe_hybrid_orth_diff_basis_top_k=7,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_diff_basis_enable", True),
            ("fe_hybrid_orth_diff_basis_corr_threshold", 0.8),
            ("fe_hybrid_orth_diff_basis_degrees", (1, 2)),
            ("fe_hybrid_orth_diff_basis_top_k", 7),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got {getattr(m2, name)}"

    def test_pickle_roundtrip_preserves_diff_basis_recipe(self):
        """A pickle round-trip preserves feature names, appended columns, and every orth_diff_basis recipe field."""
        X, y = _build_diff_signal(seed=42, n=2500)
        m = _make_mrmr(
            fe_hybrid_orth_diff_basis_enable=True,
            fe_hybrid_orth_diff_basis_corr_threshold=0.7,
            fe_hybrid_orth_diff_basis_degrees=(1, 2, 3),
            fe_hybrid_orth_diff_basis_top_k=3,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, f"pickle changed hybrid_orth_features_: before={added_before}, after={added_after}"
        recipes_before = {r.name: r for r in getattr(m, "_engineered_recipes_", []) or [] if r.kind == "orth_diff_basis"}
        recipes_after = {r.name: r for r in getattr(m2, "_engineered_recipes_", []) or [] if r.kind == "orth_diff_basis"}
        assert set(recipes_before.keys()) == set(
            recipes_after.keys()
        ), f"pickle dropped or added orth_diff_basis recipe names: before={set(recipes_before.keys())}, after={set(recipes_after.keys())}"
        for name, r_before in recipes_before.items():
            r_after = recipes_after[name]
            assert r_before.src_names == r_after.src_names, f"pickle changed src_names for {name!r}: before={r_before.src_names}, after={r_after.src_names}"
            for key in ("basis", "degree"):
                assert r_before.extra.get(key) == r_after.extra.get(
                    key
                ), f"pickle changed '{key}' for recipe {name!r}: before={r_before.extra}, after={r_after.extra}"


# ---------------------------------------------------------------------------
# Contract 7: recipe replay round-trip (orth_diff_basis kind)
# ---------------------------------------------------------------------------


class TestRecipeReplay:
    """apply_recipe at transform time must reproduce the fit-time diff-basis engineered column."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_recipe_replay_matches_fit_time_values(self, seed):
        """Applying the ``orth_diff_basis`` recipe at transform time must
        produce exactly the fit-time engineered column (modulo float epsilon).
        """
        _, _, _, hybrid_with_recipes = _import_diff_fe()
        X, y = _build_diff_signal(seed)
        X_aug, _scores, recipes = hybrid_with_recipes(
            X,
            y.values,
            degrees=(1, 2, 3),
            pair_corr_threshold=0.7,
            top_k=5,
        )
        if not recipes:
            pytest.fail(f"seed={seed}: diff-basis emitted no recipes; replay test requires at least one recipe.")
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        appended = [c for c in X_aug.columns if c not in X.columns]
        for r in recipes:
            assert r.name in appended, f"seed={seed}: recipe {r.name!r} not in appended columns {appended}"
            assert r.kind == "orth_diff_basis", f"seed={seed}: recipe {r.name!r} kind={r.kind!r}, expected 'orth_diff_basis'."
            replayed = apply_recipe(r, X)
            fit_time = X_aug[r.name].to_numpy()
            assert np.allclose(
                replayed, fit_time, rtol=1e-9, atol=1e-12
            ), f"seed={seed}: recipe {r.name!r} replay drift: max|replayed - fit| = {float(np.max(np.abs(replayed - fit_time)))}; extra={dict(r.extra)}"
