"""Consolidated from test_biz_value_mrmr_layer15.py.

Layer 15 biz_value MRMR contracts: CONTINUOUS REGRESSION targets.

WHY THIS LAYER
--------------
Layers 1-14 covered binary / multiclass classification y exclusively.
But a huge slice of production tabular work is REGRESSION:

* price forecasting (real-estate, equities, FX)
* AUM / revenue / churn-amount prediction
* temperature / load / RUL (remaining useful life) on industrial telemetry
* engagement-time / latency / cost regression for ad-tech

If MRMR silently degrades on continuous y (e.g. collapses target to
``int16`` and trims the dynamic range, or quantizes to 2 bins and loses
the rank-ordering of the gains) then every regression model that uses
it as the feature-selection front-end ships with a hidden ceiling.

The default MRMR pipeline runs the target through the SAME
``categorize_dataset`` discretization as the features
(``_mrmr_fit_impl.py`` line 326-343 reads
``y_for_strategy=_x_for_cat[target_names[0]]`` after target injection at
line 282; the ``nbins_strategy='mdlp'`` Wave-7 default then chooses the
target's own bin count via Fayyad-Irani MDLP). For a continuous y with
N=2500 distinct values MDLP typically produces 8-12 bins -- enough to
preserve the regression signal.

DATA DESIGN
-----------
* n = 2500 rows, p = 9 columns: 3 informative + 6 noise.
* y = 1.5*x_signal_1 + 0.8*x_signal_2 + x3_coef*x_signal_3 + noise.
  Coefficients picked so the strength rank-orders match the absolute
  coefficient ordering (1.5 > 0.8 > 0.5).
* Default config (DCD on, relative-gain stop on, Miller-Madow on,
  cell-budget pre-screen on) is exercised end-to-end.

WHAT WE PIN VS DO NOT
---------------------
Pin:
  * MRMR.fit on continuous float y does NOT raise.
  * support_ contains ``x_signal_1`` and ``x_signal_2`` on every seed
    (the two strongest signals).
  * mrmr_gains_ rank-orders match the coefficient ordering when 3
    signals are selected.
  * No ``noise_*`` column appears in support_ on any seed.
  * Continuous y is binned into >2 bins (else regression info is lost).
  * Heavy-tailed (log-normal) y also keeps all 3 signals; this is the
    canonical "price forecasting" stress shape and gives a cleaner
    rank-order check than the linear case where x3_coef * sigma
    ~= noise floor.
  * MRMR-selected features give a Ridge regressor R2 >> 0 on holdout
    -- the "did MRMR pick semantically-useful columns" anchor.

Do NOT pin:
  * Exact length of support_ on the linear-y design. The default
    DCD + relative-gain stop is conservative: x_signal_3
    (coef=0.5 vs noise_scale=0.3 -> SNR ~0.4) clears or fails the
    floor depending on seed (~40% retention rate in probing). The
    log-normal scenario gives a much stronger x3 signal because of
    the variance-amplifying exp(.) and is the test where we DO pin
    "3 signals every seed".
  * Exact gain numbers -- those depend on MDLP bin edges and Wave-9
    DCD anchor swaps.

DCD + relative-gain stop + Miller-Madow + cardinality cell-budget
pre-screen are ON by default (Wave 9 flip 2026-05-30). Layer 15
respects those defaults and probes them in regression mode.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


N_TOTAL = 2_500
N_NOISE = 6
N_HOLDOUT = 300


def _build_linear_regression_data(
    seed: int,
    x3_coef: float = 0.5,
    noise_scale: float = 0.3,
):
    """y = 1.5*x1 + 0.8*x2 + x3_coef*x3 + noise.

    Coefficients picked so |1.5| > |0.8| > |0.5|, giving a clean
    expected rank order on the MRMR gains when 3 signals are
    selected. ``noise_scale=0.3`` keeps SNR(x3) ~= 0.5 / 0.3 = 1.7,
    high enough that MDLP-binned y carries x3 information past the
    relative-gain floor on most seeds (~40-60% in probing).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TOTAL)
    x2 = rng.standard_normal(N_TOTAL)
    x3 = rng.standard_normal(N_TOTAL)
    y_arr = (
        1.5 * x1
        + 0.8 * x2
        + x3_coef * x3
        + noise_scale * rng.standard_normal(N_TOTAL)
    )
    cols = {"x_signal_1": x1, "x_signal_2": x2, "x_signal_3": x3}
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y_reg")
    return X, y


def _build_lognormal_regression_data(seed: int):
    """Heavy-tailed y = exp(1.5*x1 + 0.8*x2 + 0.5*x3 + noise).

    Production price forecasting target shape: support is (0, +inf),
    distribution is log-normal, tails matter for the business KPI
    (top-decile MAPE). The exp() amplifies the latent linear signal
    and pushes x3 well above the relevance floor on EVERY seed.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TOTAL)
    x2 = rng.standard_normal(N_TOTAL)
    x3 = rng.standard_normal(N_TOTAL)
    latent = (
        1.5 * x1
        + 0.8 * x2
        + 0.5 * x3
        + 0.3 * rng.standard_normal(N_TOTAL)
    )
    y_arr = np.exp(latent)
    cols = {"x_signal_1": x1, "x_signal_2": x2, "x_signal_3": x3}
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y_lognorm")
    return X, y


def _build_bimodal_regression_data(seed: int):
    """y = (1.5*x1 + 0.8*x2 + 0.5*x3 + small_noise) + sign-of-mixture * 2.

    Two well-separated modes; quantile-binning would lump the linear
    signal inside each mode together, so MDLP supervised binning is
    the only way to keep the rank order. Useful to stress that the
    Wave-7 nbins_strategy='mdlp' default is doing its job on
    non-Gaussian regression targets.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TOTAL)
    x2 = rng.standard_normal(N_TOTAL)
    x3 = rng.standard_normal(N_TOTAL)
    raw = (
        1.5 * x1
        + 0.8 * x2
        + 0.5 * x3
        + 0.3 * rng.standard_normal(N_TOTAL)
    )
    # +/- 2 mixture component, independent of x_signal_*
    mode = rng.integers(0, 2, N_TOTAL).astype(np.float64) * 4.0 - 2.0
    y_arr = raw + mode
    cols = {"x_signal_1": x1, "x_signal_2": x2, "x_signal_3": x3}
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y_bimodal")
    return X, y


def _make_mrmr(**overrides):
    """Default-config MRMR for layer 15.

    Layer 15 exercises the production default surface (DCD ON,
    relative-gain stop ON, Miller-Madow ON, cell-budget pre-screen ON).
    Only ``fe_max_steps=0`` and ``interactions_max_order=1`` are pinned
    to keep wall-time bounded -- they don't interact with the continuous-y
    binning path being tested.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _fit_quiet(sel, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sel.fit(X, y)


SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Contract 1: MRMR does not crash on continuous float y
# ---------------------------------------------------------------------------


class TestContinuousYDoesNotCrash:
    """The minimum bar: MRMR must accept float y and return a non-empty
    support_. A crash here means continuous regression callers see a
    hard failure; an empty support_ means downstream pipelines die on
    a 0-column transform.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_continuous_y_fit_runs(self, seed):
        X, y = _build_linear_regression_data(seed)
        # Sanity: y really is continuous (>> 2 unique values).
        assert y.nunique() > 100, (
            f"test bug: continuous y has only {y.nunique()} unique vals"
        )
        assert y.dtype.kind == "f", f"y must be float; got {y.dtype}"
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        assert sel.support_ is not None
        assert sel.n_features_ >= 1, (
            f"MRMR returned empty support_ on continuous y, seed={seed}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_lognormal_y_fit_runs(self, seed):
        X, y = _build_lognormal_regression_data(seed)
        assert y.min() > 0, "log-normal y must be strictly positive"
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        assert sel.n_features_ >= 1

    @pytest.mark.parametrize("seed", SEEDS)
    def test_bimodal_y_fit_runs(self, seed):
        X, y = _build_bimodal_regression_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        assert sel.n_features_ >= 1


# ---------------------------------------------------------------------------
# Contract 2: Top-2 signals always recovered on linear regression y
# ---------------------------------------------------------------------------


class TestTopTwoSignalsRecovered:
    """On the linear-regression y design the two strongest signals
    (coef 1.5 and 0.8) must appear in support_ on EVERY seed. The
    third signal (coef 0.5) is at the edge of the relative-gain
    floor and may flip seed-to-seed; we test it separately on the
    log-normal scenario where SNR is comfortably above the floor.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_x_signal_1_and_2_in_support(self, seed):
        X, y = _build_linear_regression_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        names = list(sel.get_feature_names_out())
        assert "x_signal_1" in names, (
            f"Strongest signal x_signal_1 (coef=1.5) missing from "
            f"support_; seed={seed}, support={names}"
        )
        assert "x_signal_2" in names, (
            f"Second-strongest signal x_signal_2 (coef=0.8) missing "
            f"from support_; seed={seed}, support={names}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_strongest_signal_is_picked_first(self, seed):
        """x_signal_1 has the largest coefficient (1.5) so it should
        be the FIRST feature MRMR selects on the linear design.
        """
        X, y = _build_linear_regression_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        names = list(sel.get_feature_names_out())
        assert names[0] == "x_signal_1", (
            f"MRMR top-1 on linear regression must be x_signal_1 "
            f"(coef=1.5); got {names[0]} on seed={seed}; full "
            f"support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 3: Noise columns always excluded
# ---------------------------------------------------------------------------


class TestNoiseExcluded:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_noise_column_selected_linear(self, seed):
        X, y = _build_linear_regression_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        names = list(sel.get_feature_names_out())
        leaked = [n for n in names if n.startswith("noise_")]
        assert not leaked, (
            f"noise column(s) {leaked} leaked into support_ on "
            f"continuous-y selection; seed={seed}, support={names}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_noise_column_selected_lognormal(self, seed):
        X, y = _build_lognormal_regression_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        names = list(sel.get_feature_names_out())
        leaked = [n for n in names if n.startswith("noise_")]
        assert not leaked, (
            f"noise column(s) {leaked} leaked into log-normal y "
            f"support_; seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 4: Log-normal y captures ALL 3 signals + rank-orders them
# ---------------------------------------------------------------------------


class TestLogNormalAllThreeSignals:
    """The exp() in y amplifies the latent linear combination, so
    every signal clears the relevance floor with margin. We pin both
    "all 3 signals appear" AND "gains rank-order x1 > x2 > x3" on
    every seed.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_lognormal_y_recovers_all_three_signals(self, seed):
        X, y = _build_lognormal_regression_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        names = list(sel.get_feature_names_out())
        for s in ("x_signal_1", "x_signal_2", "x_signal_3"):
            assert s in names, (
                f"signal {s!r} missing from log-normal y support_; "
                f"seed={seed}, support={names}"
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_lognormal_y_gains_rank_order(self, seed):
        """Selection order (gains_[0] > gains_[1] > gains_[2]) must
        match the coefficient ordering 1.5 > 0.8 > 0.5. MRMR's greedy
        path picks features in descending gain so the order in
        ``support_`` is the order in ``mrmr_gains_``.
        """
        X, y = _build_lognormal_regression_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X, y)
        names = list(sel.get_feature_names_out())
        # All 3 must be there for the rank check to be meaningful.
        assert {"x_signal_1", "x_signal_2", "x_signal_3"}.issubset(set(names))
        # Map name -> position in support_ (= selection order).
        pos = {n: i for i, n in enumerate(names)}
        assert pos["x_signal_1"] < pos["x_signal_2"] < pos["x_signal_3"], (
            f"Selection order does not match effect-size order on "
            f"seed={seed}: support={names}, gains={list(sel.mrmr_gains_)}"
        )
        # And the underlying gains are strictly descending across the
        # greedy steps (MRMR's invariant; pinned here as a regression
        # guard against a future change that would let later steps
        # report a higher gain than the previous one).
        gains = [sel.mrmr_gains_[pos[n]] for n in ("x_signal_1", "x_signal_2", "x_signal_3")]
        assert gains[0] > gains[1] > gains[2], (
            f"mrmr_gains_ not strictly descending across the 3 signals "
            f"on seed={seed}: {gains}"
        )


# ---------------------------------------------------------------------------
# Contract 5: y is binned into >2 bins (regression info preserved)
# ---------------------------------------------------------------------------


class TestContinuousYBinning:
    """The MDLP supervised binner (Wave-7 default) must give the
    continuous y at least 3 bins. A 2-bin binning would collapse the
    continuous target to a binary indicator and erase ~all regression
    signal -- noise features would then appear competitive against
    real signals.

    This test bypasses the MRMR public surface and directly invokes
    the categorize_dataset path that ``_mrmr_fit_impl`` uses, since
    MRMR does not expose the discretized target externally.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mdlp_gives_y_more_than_two_bins(self, seed):
        from mlframe.feature_selection.filters.discretization import (
            categorize_dataset,
        )
        X, y = _build_linear_regression_data(seed)
        # Mimic the in-fit injection: y becomes one of the columns of
        # the frame passed to categorize_dataset, AND is also passed
        # via y_for_strategy so the MDLP supervised binner is fed.
        df = X.copy()
        df["targ_y"] = y.values
        data, cols, nbins = categorize_dataset(
            df=df,
            method="quantile",
            n_bins=10,
            dtype=np.int32,
            missing_strategy="separate_bin",
            nbins_strategy="mdlp",  # the default since Wave 7
            nbins_strategy_kwargs=None,
            y_for_strategy=y,
        )
        y_idx = cols.index("targ_y")
        n_distinct_y_codes = int(np.unique(data[:, y_idx]).size)
        assert n_distinct_y_codes >= 3, (
            f"Continuous y collapsed to {n_distinct_y_codes} bins on "
            f"seed={seed}; default Wave-7 MDLP binning must preserve "
            f">= 3 levels for regression MI to carry signal."
        )


# ---------------------------------------------------------------------------
# Contract 6: Downstream Ridge regression beats no-selection baseline
# ---------------------------------------------------------------------------


class TestDownstreamRidgeAnchor:
    """Selected features must be USEFUL: a Ridge regression on the
    MRMR support must achieve R2 well above 0 on holdout. Anchor
    against the "MRMR dropped everything except noise" failure mode
    -- noise-only support would yield R2 ~= 0 (or negative).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_ridge_on_mrmr_support_beats_zero_r2(self, seed):
        X, y = _build_linear_regression_data(seed)
        X_tr, X_te = X.iloc[:-N_HOLDOUT], X.iloc[-N_HOLDOUT:]
        y_tr, y_te = y.iloc[:-N_HOLDOUT], y.iloc[-N_HOLDOUT:]
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X_tr, y_tr)
        Xs_tr = sel.transform(X_tr)
        Xs_te = sel.transform(X_te)
        model = Ridge().fit(Xs_tr, y_tr)
        r2 = r2_score(y_te, model.predict(Xs_te))
        assert r2 > 0.70, (
            f"Ridge on MRMR-selected continuous-y support_ has "
            f"R2={r2:.3f} on holdout; expected > 0.70 for a 1.5*x1 + "
            f"0.8*x2 + 0.5*x3 design with noise_scale=0.3. Selected: "
            f"{list(sel.get_feature_names_out())}; seed={seed}"
        )


# ---------------------------------------------------------------------------
# Contract 7: Float32 y is handled the same as float64
# ---------------------------------------------------------------------------


class TestFloat32YParity:
    """Production frames frequently arrive as float32 (memory budget).
    A silent crash or a regression-mode quality drop because of
    float32 quantisation would be a hidden ceiling. We check that
    float32 y produces the same support as float64 y on the same
    seed.
    """

    @pytest.mark.parametrize("seed", (7, 42))
    def test_float32_y_matches_float64_support(self, seed):
        X, y64 = _build_linear_regression_data(seed)
        y32 = y64.astype(np.float32)

        sel64 = _make_mrmr(random_seed=seed)
        _fit_quiet(sel64, X.copy(), y64)
        sup64 = list(sel64.get_feature_names_out())

        sel32 = _make_mrmr(random_seed=seed)
        _fit_quiet(sel32, X.copy(), y32)
        sup32 = list(sel32.get_feature_names_out())

        assert sup64 == sup32, (
            f"float32 y diverged from float64 y on the same seed: "
            f"f64 support={sup64}, f32 support={sup32}, seed={seed}"
        )
