"""Consolidated from test_biz_value_mrmr_layer24.py.

Layer 24 biz_value: HYBRID FE LIFT ON PRODUCTION-SHAPED REAL-WORLD SCENARIOS.

Layers 21-23 pinned the API contract: hybrid orthogonal-polynomial FE
auto-wires into ``MRMR.fit`` via ``fe_hybrid_orth_enable=True``, generates
``basis_n(z)`` univariate and ``basis_a(z_i)*basis_b(z_j)`` cross-basis
columns, MI-ranks them, and feeds the winners through the standard
relevance/redundancy screen.

This layer goes a step further: it proves the BIZ_VALUE on five
production-shaped target shapes -- the kind of non-linearity practitioners
actually see in churn / finance / sensor pipelines. For each scenario we
fit baseline MRMR (``fe_hybrid_orth_enable=False``) and hybrid MRMR
(``...=True``), train a downstream LogReg/LinearRegression on the
selected feature subset, score holdout AUC / R^2, and pin a hard lift
floor.

Scenarios (5 seeds each):

* A. **Financial cross-product**: ``y = sign(price * volume + noise)`` --
  the classic XOR-on-continuous scenario. Linear LogReg cannot solve it
  on raw ``(price, volume)`` columns; the pair-cross-basis stage emits
  ``price*volume__He1_He1`` which is exactly the signal. Pin: hybrid
  holdout AUC lift >= +0.20 (target +0.10, achieved ~+0.40 in practice).

* B. **Sensor U-shape**: ``y = sign(temperature^2 - 1)`` -- decision
  boundary at |temp|=1. Linear MI of raw ``temperature`` to ``y`` is near
  zero (symmetric). Hermite ``He_2(z) = z^2 - 1`` captures the signal
  exactly. Pin: ``temperature__He2`` in selected support AND holdout AUC
  lift >= +0.30.

* C. **Asymmetric churn / U-shape**: ``y = sign(usage^2 - threshold)``.
  Single-feature non-linearity that raw MRMR misses entirely (returns
  ``usage`` alone, downstream LogReg ~0.5 AUC). Pin: ``usage__He2``
  enters support AND baseline AUC <= 0.55 (raw MRMR cannot solve) AND
  hybrid AUC >= 0.90.

* D. **Polynomial regression**: continuous ``y = x^2 + 0.2*x + noise``.
  Baseline LinearRegression on raw ``x`` gets R^2 ~= 0.01 (the quadratic
  shape is invisible to a linear fit). With ``x__He2`` added the model
  recovers a clean quadratic fit. Pin: ``x__He2`` in support AND R^2
  lift >= +0.50 (target +0.10, achieved ~+0.93 in practice). EXERCISES
  the regression-y qcut binning fix in ``_mrmr_fit_impl._prepare_y``.

* E. **Mixed bag**: 9 features, 2 with He_2 signal (``a``, ``b``) and 1
  cross-basis pair (``c1*c2``), 5 pure noise. Pin: at least 2 of the 3
  expected engineered winners reach the support AND hybrid AUC lift
  >= +0.25.

* NEGATIVE CONTROL: pure linear additive signal. Hybrid must NOT add
  spurious He_2 / cross-basis columns to the support; pin no engineered
  columns are emitted and AUC parity with baseline.

NEVER xfail. If a real-world contract fails, the prod code is wrong, not
the test. The contract floors are well below the empirically observed
lifts so they tolerate seed-level variance without being trivial.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

warnings.filterwarnings("ignore")


# Five seeds per scenario give us ~5x replication on each contract. Per-seed
# floors are picked so all five pass even on the worst seed observed during
# calibration (no average-across-seeds wiggle).
SEEDS = (1, 7, 13, 42, 101)


def _mrmr_kw(**overrides):
    """Common MRMR knobs that keep the run cheap, deterministic, and isolate
    the hybrid stage from other auto-wired interactions (DCD, cluster
    aggregate, friend graph) so the contract measures hybrid FE alone.
    """
    base = dict(
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
    base.update(overrides)
    return base


def _fit_mrmr(X, y, *, hybrid: bool, pair: bool = True, degrees=(2, 3), top_k: int = 5):
    """Fit MRMR with hybrid orthogonal-basis FE enabled or explicitly disabled."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    if hybrid:
        kw = _mrmr_kw(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=pair,
            fe_hybrid_orth_pair_max_degree=2,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_degrees=degrees,
            fe_hybrid_orth_top_k=top_k,
        )
    else:
        # EXPLICIT off baseline: fe_hybrid_orth_enable defaults to True since 2026-06-21,
        # so the no-hybrid control must disable it explicitly (was relying on the default).
        kw = _mrmr_kw(fe_hybrid_orth_enable=False)
    return MRMR(**kw).fit(X, y)


def _split(X, y, n_tr: int):
    """Split X, y into a train prefix and test suffix at row n_tr."""
    Xtr, Xte = X.iloc[:n_tr], X.iloc[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]
    return Xtr, ytr, Xte, yte


def _classifier_holdout_auc(mrmr_fit, Xtr, ytr, Xte, yte) -> float:
    """Train LogReg on the MRMR-transformed train frame; report holdout AUC."""
    Xtr_t = np.asarray(mrmr_fit.transform(Xtr))
    Xte_t = np.asarray(mrmr_fit.transform(Xte))
    if Xtr_t.shape[1] == 0:
        # All-empty support: degenerate, can't score; treat as random.
        return 0.5
    lr = LogisticRegression(max_iter=500, solver="lbfgs").fit(Xtr_t, ytr)
    return float(roc_auc_score(yte, lr.predict_proba(Xte_t)[:, 1]))


def _regressor_holdout_r2(mrmr_fit, Xtr, ytr, Xte, yte) -> float:
    """Train LinearRegression on the MRMR-transformed train frame; report holdout R^2."""
    Xtr_t = np.asarray(mrmr_fit.transform(Xtr))
    Xte_t = np.asarray(mrmr_fit.transform(Xte))
    if Xtr_t.shape[1] == 0:
        return float("nan")
    lr = LinearRegression().fit(Xtr_t, ytr)
    return float(r2_score(yte, lr.predict(Xte_t)))


# ---------------------------------------------------------------------------
# A. Financial cross-product: y = sign(price * volume + noise)
# ---------------------------------------------------------------------------


def _build_financial(seed: int, n: int = 3000):
    """Two centred-Gaussian factors plus two nuisance columns. Target is the
    sign of the product -- a classic XOR-on-continuous shape that linear
    LogReg on the raw frame cannot solve (it cannot separate the four
    quadrants with a single hyperplane).
    """
    rng = np.random.default_rng(seed)
    price = rng.standard_normal(n)
    volume = rng.standard_normal(n)
    y = ((price * volume) + 0.35 * rng.standard_normal(n) > 0).astype(int)
    X = pd.DataFrame(
        {
            "price": price,
            "volume": volume,
            "macd": rng.standard_normal(n),
            "rsi": rng.standard_normal(n),
        }
    )
    return X, pd.Series(y, name="y")


class TestScenarioAFinancialCross:
    """``y = sign(price*volume)`` -- hybrid FE lifts holdout AUC and the price*volume cross enters support."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_lifts_auc_and_picks_cross(self, seed):
        """Hybrid MRMR clears 0.75 AUC, lifts >= +0.20 over baseline, and picks a price*volume cross-basis column."""
        X, y = _build_financial(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        mh = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=True, pair=True)
        auc_b = _classifier_holdout_auc(mb, Xtr, ytr, Xte, yte)
        auc_h = _classifier_holdout_auc(mh, Xtr, ytr, Xte, yte)
        sup_h = list(mh.get_feature_names_out())

        # Contract A.1: baseline really IS unsolvable by raw linear LogReg on
        # this target shape (sanity check on the scenario, not on the code).
        assert auc_b <= 0.62, (
            f"A seed={seed}: baseline AUC {auc_b:.3f} unexpectedly high; "
            f"scenario is meant to be XOR-shaped (linear-unsolvable). "
            f"Either signal is too clean or noise too low."
        )
        # Contract A.2: hybrid clears 0.75 AUC (target from task spec).
        assert auc_h >= 0.75, f"A seed={seed}: hybrid AUC {auc_h:.3f} should clear 0.75 on price*volume XOR; support={sup_h}"
        # Contract A.3: per-seed lift floor at +0.20 (task floor +0.10,
        # observed range +0.36..+0.44 in calibration -- the +0.20 floor
        # absorbs seed variance while pinning a real lift, not a tie).
        lift = auc_h - auc_b
        assert lift >= 0.20, f"A seed={seed}: hybrid AUC lift {lift:+.3f} should be >= +0.20 (base={auc_b:.3f}, hybrid={auc_h:.3f}); support={sup_h}"
        # Contract A.4: cross-basis term enters the support. Allow either leg
        # ordering ("price*volume" or "volume*price") since both can appear.
        has_cross = any(("*" in c) and ("price" in c) and ("volume" in c) for c in sup_h)
        assert has_cross, f"A seed={seed}: expected a price*volume cross-basis column in hybrid support; got {sup_h}"


# ---------------------------------------------------------------------------
# B. Sensor U-shape: y = sign(temperature^2 - 1)
# ---------------------------------------------------------------------------


def _build_sensor_ushape(seed: int, n: int = 3000):
    """Single sensor + three nuisance channels. Decision boundary at
    |temperature|=1 -- a clean Hermite He_2 target.
    """
    rng = np.random.default_rng(seed)
    temp = rng.standard_normal(n)
    y = ((temp**2 - 1.0) + 0.2 * rng.standard_normal(n) > 0).astype(int)
    X = pd.DataFrame(
        {
            "temperature": temp,
            "humidity": rng.standard_normal(n),
            "pressure": rng.standard_normal(n),
            "wind": rng.standard_normal(n),
        }
    )
    return X, pd.Series(y, name="y")


def _has_univariate_basis_of(support, var):
    """True iff ``support`` contains an orthogonal-basis univariate transform of
    ``var`` (named ``{var}__<basiscode><degree>`` e.g. ``temperature__He2`` /
    ``temperature__T2``); robust to which basis the ``auto`` selector picks."""
    return any(str(s).startswith(f"{var}__") for s in support)


class TestScenarioBSensorUShape:
    """``y = sign(temperature^2 - 1)`` -- the DEFAULT MRMR recovers the U-shape via a univariate basis detector."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_recovers_temperature_ushape(self, seed):
        """Default MRMR selects a temperature univariate basis detector and clears 0.90 AUC."""
        # 2026-06-02: with univariate-basis FE default-ON, the DEFAULT MRMR
        # recovers the y = sign(temperature^2 - 1) U-shape via a single-source
        # ``temperature__He2`` detector -- no explicit hybrid opt-in needed.
        # (Previously the default baseline was weak here and only the hybrid
        # recovered it; closing the univariate-nonlinearity gap made the default
        # strong, so this now pins the DEFAULT's recovery.)
        X, y = _build_sensor_ushape(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        auc_b = _classifier_holdout_auc(mb, Xtr, ytr, Xte, yte)
        sup_b = list(mb.get_feature_names_out())

        assert _has_univariate_basis_of(sup_b, "temperature"), (
            f"B seed={seed}: a temperature univariate basis detector (the U-shape recoverer) must be in the DEFAULT support; got {sup_b}"
        )
        assert auc_b >= 0.90, (
            f"B seed={seed}: DEFAULT AUC {auc_b:.3f} should clear 0.90 -- the univariate He_2 basis captures y = sign(temp^2-1) exactly; support={sup_b}"
        )


# ---------------------------------------------------------------------------
# C. Asymmetric churn: y = sign(usage^2 - threshold). Raw MRMR misses it.
# ---------------------------------------------------------------------------


def _build_asymmetric_churn(seed: int, n: int = 3000):
    """``y = sign(usage^2 - 1)`` -- an even non-linearity invisible to a linear model on raw usage."""
    rng = np.random.default_rng(seed)
    usage = rng.standard_normal(n)
    y = ((usage**2 - 1.0) + 0.15 * rng.standard_normal(n) > 0).astype(int)
    X = pd.DataFrame(
        {
            "usage": usage,
            "tenure": rng.standard_normal(n),
            "contacts": rng.standard_normal(n),
            "fees": rng.standard_normal(n),
            "logins": rng.standard_normal(n),
        }
    )
    return X, pd.Series(y, name="y")


class TestScenarioCAsymmetricChurn:
    """``y = sign(usage^2 - 1)`` -- the DEFAULT MRMR recovers the churn non-linearity via usage__He2."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_recovers_asymmetric_churn(self, seed):
        """Default MRMR selects a usage univariate basis detector and clears 0.90 AUC."""
        # 2026-06-02: y = sign(usage^2 - 1) is invisible to a linear model on RAW
        # usage (usage^2 is even -> raw usage uninformative), which is exactly the
        # univariate-nonlinearity gap. With univariate-basis FE default-ON the
        # DEFAULT MRMR now recovers it via ``usage__He2`` -- this used to be the
        # "raw MRMR fails, only hybrid recovers" pin; the default is now the path
        # that recovers, so it pins the DEFAULT's recovery.
        X, y = _build_asymmetric_churn(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        auc_b = _classifier_holdout_auc(mb, Xtr, ytr, Xte, yte)
        sup_b = list(mb.get_feature_names_out())

        assert _has_univariate_basis_of(sup_b, "usage"), f"C seed={seed}: a usage univariate basis detector must enter the DEFAULT support; got {sup_b}"
        assert auc_b >= 0.90, (
            f"C seed={seed}: DEFAULT AUC {auc_b:.3f} should clear 0.90 -- the univariate basis recovers the sign(usage^2-1) U-shape; support={sup_b}"
        )


# ---------------------------------------------------------------------------
# D. Polynomial regression: continuous y = x^2 + 0.2*x + noise.
# ---------------------------------------------------------------------------


def _build_quadratic_regression(seed: int, n: int = 3000):
    """Continuous target with a quadratic dominant component. Linear
    regression on raw x is dominated by symmetry around 0; the He_2 column
    breaks the symmetry and recovers the quadratic.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (x**2) + 0.2 * x + 0.3 * rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x": x,
            "n1": rng.standard_normal(n),
            "n2": rng.standard_normal(n),
            "n3": rng.standard_normal(n),
        }
    )
    return X, pd.Series(y, name="y")


class TestScenarioDPolynomialRegression:
    """``y = x^2 + 0.2x + noise`` -- the DEFAULT MRMR recovers the quadratic regression via x__He2."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_recovers_quadratic_regression(self, seed):
        """Default MRMR selects an x univariate basis detector and the downstream LinearRegression clears 0.85 R^2."""
        # 2026-06-02: continuous y = x^2 + 0.2x + noise. Linear regression on raw
        # x is dominated by the symmetry around 0; the univariate ``x__He2`` basis
        # breaks it and fits the quadratic. With univariate-basis FE default-ON
        # the DEFAULT MRMR recovers it (used to need the explicit hybrid). ``ytr``
        # is continuous -> the univariate stage qcut-bins it for MI scoring.
        X, y = _build_quadratic_regression(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        r2_b = _regressor_holdout_r2(mb, Xtr, ytr, Xte, yte)
        sup_b = list(mb.get_feature_names_out())

        assert _has_univariate_basis_of(sup_b, "x"), f"D seed={seed}: an x univariate basis detector must enter the DEFAULT regression support; got {sup_b}"
        assert r2_b >= 0.85, (
            f"D seed={seed}: DEFAULT R^2 {r2_b:.3f} should clear 0.85 -- the univariate-basis-augmented LR essentially fits the quadratic; support={sup_b}"
        )


# ---------------------------------------------------------------------------
# E. Mixed bag: 2 univariate He_2 + 1 cross + 5 noise.
# ---------------------------------------------------------------------------


def _build_mixed_bag(seed: int, n: int = 3000):
    """9-column mixed FE scenario. Two single-feature He_2 signals + one
    pair-cross interaction + five pure-noise nuisance columns; the hybrid
    pipeline should recover at least the two univariate winners on every
    seed (the cross sometimes loses to the easier univariates in MRMR's
    redundancy gate but the contract pins >= 2-of-3 winners, not all 3).
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c1 = rng.standard_normal(n)
    c2 = rng.standard_normal(n)
    cols = {"a": a, "b": b, "c1": c1, "c2": c2}
    for i in range(5):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    sig = (a**2 - 1.0) + (b**2 - 1.0) + 1.5 * c1 * c2
    y = (sig + 0.5 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


class TestScenarioEMixedBag:
    """A 9-column mixed FE scenario -- the DEFAULT MRMR recovers at least 2 of the 3 engineered winners."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_recovers_at_least_two_winners(self, seed):
        """Default MRMR recovers >= 2 of (a-univariate, b-univariate, c1*c2 cross) and clears 0.80 AUC."""
        # 2026-06-02: mixed bag = two univariate He_2 signals (a, b) + one cross
        # (c1*c2) + noise. With univariate-basis FE default-ON the DEFAULT MRMR
        # recovers BOTH univariate winners (``a__He2`` + ``b__He2``) -- the two
        # the substitution fix made the hybrid recover, now reached by default.
        # Contract: the DEFAULT recovers >= 2 of the 3 winners + clears 0.80 AUC.
        X, y = _build_mixed_bag(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        auc_b = _classifier_holdout_auc(mb, Xtr, ytr, Xte, yte)
        sup_b = list(mb.get_feature_names_out())

        # The three winners: univariate He_2 detector for a, for b, and ANY
        # c1*c2 cross term (the pair interaction).
        has_a = _has_univariate_basis_of(sup_b, "a")
        has_b = _has_univariate_basis_of(sup_b, "b")
        has_cross = any(("*" in c) and ("c1" in c) and ("c2" in c) for c in sup_b)
        recovered = int(has_a) + int(has_b) + int(has_cross)
        assert recovered >= 2, (
            f"E seed={seed}: DEFAULT should recover >= 2 of (a-univariate, "
            f"b-univariate, c1*c2 cross); got recovered={recovered} (a={has_a}, "
            f"b={has_b}, cross={has_cross}); support={sup_b}"
        )
        assert auc_b >= 0.80, f"E seed={seed}: DEFAULT AUC {auc_b:.3f} should clear 0.80; support={sup_b}"


# ---------------------------------------------------------------------------
# Negative control: pure linear signal -- hybrid must NOT bloat support.
# ---------------------------------------------------------------------------


def _build_linear_negative_control(seed: int, n: int = 3000):
    """A purely linear-additive signal -- hybrid FE must not manufacture spurious engineered columns."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "a": a,
            "b": b,
            "n1": rng.standard_normal(n),
            "n2": rng.standard_normal(n),
            "n3": rng.standard_normal(n),
        }
    )
    y = ((1.2 * a + 0.8 * b) + 0.4 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


class TestNegativeControlLinearSignal:
    """Verify hybrid FE does NOT manufacture engineered columns when the
    underlying signal is purely linear / additive. Spurious He_n entries
    would bloat the model and signal that the absolute-MI floor in
    ``hybrid_orth_mi_fe`` is too permissive on uninformative bases.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_engineered_columns_emitted(self, seed):
        """Hybrid MRMR selects zero engineered columns and stays within a tight AUC-parity band on pure-linear signal."""
        X, y = _build_linear_negative_control(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        mh = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=True, pair=True)
        auc_b = _classifier_holdout_auc(mb, Xtr, ytr, Xte, yte)
        auc_h = _classifier_holdout_auc(mh, Xtr, ytr, Xte, yte)
        sup_h = list(mh.get_feature_names_out())

        # Contract NC.1: no engineered columns appear in the selected support.
        # Both signature flavours are checked: ``__He`` suffix (univariate) and
        # ``*`` (cross-basis pair).
        engineered_in_support = [c for c in sup_h if ("__He" in c) or ("*" in c)]
        assert engineered_in_support == [], (
            f"NC seed={seed}: pure-linear signal should produce ZERO engineered columns in support; got {engineered_in_support}; full support={sup_h}"
        )
        # Contract NC.2: AUC parity within a tight band -- hybrid must not
        # silently degrade by adding noise features either.
        delta = auc_h - auc_b
        assert -0.02 <= delta <= 0.02, (
            f"NC seed={seed}: AUC delta {delta:+.3f} outside parity band "
            f"[-0.02, +0.02] (base={auc_b:.3f}, hybrid={auc_h:.3f}); "
            f"hybrid silently changing behaviour on linear data is a "
            f"regression."
        )
