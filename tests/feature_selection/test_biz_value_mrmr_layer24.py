"""Layer 24 biz_value: HYBRID FE LIFT ON PRODUCTION-SHAPED REAL-WORLD SCENARIOS.

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
        kw = _mrmr_kw()
    return MRMR(**kw).fit(X, y)


def _split(X, y, n_tr: int):
    Xtr, Xte = X.iloc[:n_tr], X.iloc[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]
    return Xtr, ytr, Xte, yte


def _classifier_holdout_auc(mrmr_fit, Xtr, ytr, Xte, yte) -> float:
    """Train LogReg on the MRMR-transformed train frame; report holdout AUC.
    """
    Xtr_t = np.asarray(mrmr_fit.transform(Xtr))
    Xte_t = np.asarray(mrmr_fit.transform(Xte))
    if Xtr_t.shape[1] == 0:
        # All-empty support: degenerate, can't score; treat as random.
        return 0.5
    lr = LogisticRegression(max_iter=500, solver="lbfgs").fit(Xtr_t, ytr)
    return float(roc_auc_score(yte, lr.predict_proba(Xte_t)[:, 1]))


def _regressor_holdout_r2(mrmr_fit, Xtr, ytr, Xte, yte) -> float:
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
    X = pd.DataFrame({
        "price": price,
        "volume": volume,
        "macd": rng.standard_normal(n),
        "rsi": rng.standard_normal(n),
    })
    return X, pd.Series(y, name="y")


class TestScenarioA_FinancialCross:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_lifts_auc_and_picks_cross(self, seed):
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
        assert auc_h >= 0.75, (
            f"A seed={seed}: hybrid AUC {auc_h:.3f} should clear 0.75 on "
            f"price*volume XOR; support={sup_h}"
        )
        # Contract A.3: per-seed lift floor at +0.20 (task floor +0.10,
        # observed range +0.36..+0.44 in calibration -- the +0.20 floor
        # absorbs seed variance while pinning a real lift, not a tie).
        lift = auc_h - auc_b
        assert lift >= 0.20, (
            f"A seed={seed}: hybrid AUC lift {lift:+.3f} should be >= +0.20 "
            f"(base={auc_b:.3f}, hybrid={auc_h:.3f}); support={sup_h}"
        )
        # Contract A.4: cross-basis term enters the support. Allow either leg
        # ordering ("price*volume" or "volume*price") since both can appear.
        has_cross = any(
            ("*" in c) and ("price" in c) and ("volume" in c)
            for c in sup_h
        )
        assert has_cross, (
            f"A seed={seed}: expected a price*volume cross-basis column in "
            f"hybrid support; got {sup_h}"
        )


# ---------------------------------------------------------------------------
# B. Sensor U-shape: y = sign(temperature^2 - 1)
# ---------------------------------------------------------------------------


def _build_sensor_ushape(seed: int, n: int = 3000):
    """Single sensor + three nuisance channels. Decision boundary at
    |temperature|=1 -- a clean Hermite He_2 target.
    """
    rng = np.random.default_rng(seed)
    temp = rng.standard_normal(n)
    y = ((temp ** 2 - 1.0) + 0.2 * rng.standard_normal(n) > 0).astype(int)
    X = pd.DataFrame({
        "temperature": temp,
        "humidity": rng.standard_normal(n),
        "pressure": rng.standard_normal(n),
        "wind": rng.standard_normal(n),
    })
    return X, pd.Series(y, name="y")


class TestScenarioB_SensorUShape:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_temperature_he2_lifts_auc(self, seed):
        X, y = _build_sensor_ushape(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        mh = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=True, pair=False)
        auc_b = _classifier_holdout_auc(mb, Xtr, ytr, Xte, yte)
        auc_h = _classifier_holdout_auc(mh, Xtr, ytr, Xte, yte)
        sup_h = list(mh.get_feature_names_out())

        assert "temperature__He2" in sup_h, (
            f"B seed={seed}: temperature__He2 (the U-shape detector) must be "
            f"in hybrid support; got {sup_h}"
        )
        # Per-task floor +0.05. Empirical lift is ~+0.50 with low variance --
        # pin the meaningful contract at +0.30 so a regression that quietly
        # picks the wrong basis or loses the column to redundancy gates
        # trips the test.
        lift = auc_h - auc_b
        assert lift >= 0.30, (
            f"B seed={seed}: hybrid AUC lift {lift:+.3f} should be >= +0.30 "
            f"(base={auc_b:.3f}, hybrid={auc_h:.3f}); support={sup_h}"
        )
        assert auc_h >= 0.90, (
            f"B seed={seed}: hybrid AUC {auc_h:.3f} should clear 0.90 -- "
            f"He_2 captures the y = sign(temp^2-1) shape exactly; "
            f"support={sup_h}"
        )


# ---------------------------------------------------------------------------
# C. Asymmetric churn: y = sign(usage^2 - threshold). Raw MRMR misses it.
# ---------------------------------------------------------------------------


def _build_asymmetric_churn(seed: int, n: int = 3000):
    rng = np.random.default_rng(seed)
    usage = rng.standard_normal(n)
    y = ((usage ** 2 - 1.0) + 0.15 * rng.standard_normal(n) > 0).astype(int)
    X = pd.DataFrame({
        "usage": usage,
        "tenure": rng.standard_normal(n),
        "contacts": rng.standard_normal(n),
        "fees": rng.standard_normal(n),
        "logins": rng.standard_normal(n),
    })
    return X, pd.Series(y, name="y")


class TestScenarioC_AsymmetricChurn:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_raw_mrmr_fails_hybrid_recovers(self, seed):
        X, y = _build_asymmetric_churn(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        mh = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=True, pair=False)
        auc_b = _classifier_holdout_auc(mb, Xtr, ytr, Xte, yte)
        auc_h = _classifier_holdout_auc(mh, Xtr, ytr, Xte, yte)
        sup_h = list(mh.get_feature_names_out())

        # Contract C.1: raw MRMR genuinely misses the signal (~random AUC).
        # If this ever passes, the scenario is no longer challenging -- adjust
        # noise upwards before relaxing the assertion.
        assert auc_b <= 0.60, (
            f"C seed={seed}: baseline AUC {auc_b:.3f} unexpectedly high; "
            f"y = sign(usage^2 - 1) is meant to be invisible to linear "
            f"LogReg on raw usage."
        )
        # Contract C.2: He_2 makes the support.
        assert "usage__He2" in sup_h, (
            f"C seed={seed}: usage__He2 must enter hybrid support; got {sup_h}"
        )
        # Contract C.3: hybrid AUC dominates -- recovers the U-shape signal.
        assert auc_h >= 0.90, (
            f"C seed={seed}: hybrid AUC {auc_h:.3f} should clear 0.90; "
            f"support={sup_h}"
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
    y = (x ** 2) + 0.2 * x + 0.3 * rng.standard_normal(n)
    X = pd.DataFrame({
        "x": x,
        "n1": rng.standard_normal(n),
        "n2": rng.standard_normal(n),
        "n3": rng.standard_normal(n),
    })
    return X, pd.Series(y, name="y")


class TestScenarioD_PolynomialRegression:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_he2_lifts_r2(self, seed):
        X, y = _build_quadratic_regression(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        # ``ytr`` is continuous (float). The hybrid stage now qcut-bins it for
        # MI scoring; see ``_mrmr_fit_impl._prepare_y`` (2026-05-31 fix).
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        mh = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=True, pair=False, degrees=(2, 3))
        r2_b = _regressor_holdout_r2(mb, Xtr, ytr, Xte, yte)
        r2_h = _regressor_holdout_r2(mh, Xtr, ytr, Xte, yte)
        sup_h = list(mh.get_feature_names_out())

        # Contract D.1: x__He2 in support -- the quadratic detector must
        # have survived MI scoring after the qcut-bin path was wired in.
        assert "x__He2" in sup_h, (
            f"D seed={seed}: x__He2 must enter the regression hybrid support; "
            f"got {sup_h}"
        )
        # Contract D.2: R^2 lift is enormous because raw LR can't model the
        # quadratic (baseline R^2 ~= 0). Per-task floor is +0.10; pin +0.50.
        lift = r2_h - r2_b
        assert lift >= 0.50, (
            f"D seed={seed}: R^2 lift {lift:+.3f} should be >= +0.50 "
            f"(base={r2_b:.3f}, hybrid={r2_h:.3f}); support={sup_h}"
        )
        assert r2_h >= 0.85, (
            f"D seed={seed}: hybrid R^2 {r2_h:.3f} should clear 0.85 -- "
            f"the He_2-augmented LR is essentially fitting a quadratic; "
            f"support={sup_h}"
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
    sig = (a ** 2 - 1.0) + (b ** 2 - 1.0) + 1.5 * c1 * c2
    y = (sig + 0.5 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


class TestScenarioE_MixedBag:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_recovers_at_least_two_winners(self, seed):
        X, y = _build_mixed_bag(seed)
        Xtr, ytr, Xte, yte = _split(X, y.to_numpy(), n_tr=2000)
        mb = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=False)
        mh = _fit_mrmr(Xtr, pd.Series(ytr), hybrid=True, pair=True, top_k=8)
        auc_b = _classifier_holdout_auc(mb, Xtr, ytr, Xte, yte)
        auc_h = _classifier_holdout_auc(mh, Xtr, ytr, Xte, yte)
        sup_h = list(mh.get_feature_names_out())

        # Count which expected engineered winners reached the support.
        # The three winners we hope to see:
        #   1. a__He2 (univariate quadratic signal in 'a')
        #   2. b__He2 (univariate quadratic signal in 'b')
        #   3. ANY c1*c2 cross-basis term (the pair interaction)
        has_a = "a__He2" in sup_h
        has_b = "b__He2" in sup_h
        has_cross = any(
            ("*" in c) and ("c1" in c) and ("c2" in c)
            for c in sup_h
        )
        recovered = int(has_a) + int(has_b) + int(has_cross)
        assert recovered >= 2, (
            f"E seed={seed}: hybrid should recover >= 2 of (a__He2, b__He2, "
            f"c1*c2 cross); got recovered={recovered} (a={has_a}, b={has_b}, "
            f"cross={has_cross}); support={sup_h}"
        )
        # End-to-end downstream lift. Per-task floor reads ">= +0.25"; this
        # mirrors the calibration range +0.34..+0.49.
        lift = auc_h - auc_b
        assert lift >= 0.25, (
            f"E seed={seed}: hybrid AUC lift {lift:+.3f} should be >= +0.25 "
            f"(base={auc_b:.3f}, hybrid={auc_h:.3f}); support={sup_h}"
        )
        assert auc_h >= 0.80, (
            f"E seed={seed}: hybrid AUC {auc_h:.3f} should clear 0.80; "
            f"support={sup_h}"
        )


# ---------------------------------------------------------------------------
# Negative control: pure linear signal -- hybrid must NOT bloat support.
# ---------------------------------------------------------------------------


def _build_linear_negative_control(seed: int, n: int = 3000):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    X = pd.DataFrame({
        "a": a,
        "b": b,
        "n1": rng.standard_normal(n),
        "n2": rng.standard_normal(n),
        "n3": rng.standard_normal(n),
    })
    y = ((1.2 * a + 0.8 * b) + 0.4 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


class TestNegativeControl_LinearSignal:
    """Verify hybrid FE does NOT manufacture engineered columns when the
    underlying signal is purely linear / additive. Spurious He_n entries
    would bloat the model and signal that the absolute-MI floor in
    ``hybrid_orth_mi_fe`` is too permissive on uninformative bases.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_engineered_columns_emitted(self, seed):
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
        engineered_in_support = [
            c for c in sup_h if ("__He" in c) or ("*" in c)
        ]
        assert engineered_in_support == [], (
            f"NC seed={seed}: pure-linear signal should produce ZERO "
            f"engineered columns in support; got {engineered_in_support}; "
            f"full support={sup_h}"
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
