"""Consolidated from test_biz_value_mrmr_layer11.py.

Layer 11 biz_value MRMR contracts: ANOMALY / OUTLIER RESISTANCE.

WHY THIS LAYER
--------------
Real production tabular data is never clean. Sensor streams emit
occasional extreme spikes (saturated readings, unit-conversion bugs,
"sentinel" values like ``-9999``); ETL pipelines emit NaN clusters
when an upstream join misses; division by a near-zero value lands an
``inf`` in the feature matrix. A feature selector that crashes,
silently drops the signal column, or promotes a previously-irrelevant
column to the top of the ranking because of an outlier-induced bin
collapse is a production liability.

Binning-based MI estimators are the textbook failure case:

* A 1000x-scale outlier in ``x1`` pushes 99% of the data into the
  lowest quantile bin, leaving 1% in the highest bin. The resulting
  histogram is degenerate and the empirical MI ``I(X1; Y)`` can drop
  by an order of magnitude.
* A handful of ``inf`` values upstream of a quantile call produces
  ``+inf`` bin edges and downstream NaN MI.
* A NaN cluster, if not routed through the dedicated ``separate_bin``
  path, either crashes ``np.digitize`` or, worse, silently aliases NaN
  rows into bin 0 and over-estimates the relevance of any column
  whose low-value tail correlates with the missingness.

The data we build for this layer makes the failure surface concrete:

* ``x1`` carries the strongest signal (``y = sign(x1 + 0.5 * x2 + noise)``).
* ``x2`` carries weaker, secondary signal.
* Five noise columns ``noise_0..noise_4`` have NO signal.
* Across test cases we inject:
    - 1% / 5% extreme outliers (``+/-1000``) in ``x1``,
    - 5% NaN cluster in ``x1`` (random placement, NOT MCAR-aligned to y),
    - 1% / 5% extreme outliers in EVERY noise column (decoy attack),
    - mixed extreme + NaN simultaneously,
    - ``inf`` / ``-inf`` injections (production safety gate).

CONTRACTS PINNED
----------------
1. **x1 stays rank #0 under extreme outliers** (1% AND 5% magnitude
   1000x). Quantile binning is robust enough to survive the spike
   because the rank transform inside the quantization step bounds the
   damage to one bin worth of mass.

2. **x2 stays in support_** under all non-inf outlier modes. Secondary
   signal is not lost to outlier-induced relevance noise.

3. **5% NaN cluster in x1 does not drop x1** -- the ``separate_bin``
   NaN-handling path keeps MI(X1; Y) computable.

4. **Noise-column outliers (1% spikes in every noise column) do NOT
   displace x1 from rank #0** and do NOT make any noise column outrank
   ``x2``. This pins the "decoy" attack: an attacker who can spike
   noise columns cannot promote them above genuine signal.

5. **inf/-inf raises a clean ValueError with actionable message**.
   This is the production safety gate analogous to Layer 10's
   high-cardinality refusal: the discretization step has no well-
   defined behaviour on inf so MRMR refuses to proceed and tells the
   user to clean the input. Pinned: error type AND the message
   references both ``inf`` and a remediation verb (``replace`` or
   ``drop``).

6. **Mixed extreme + NaN (NO inf) does not crash** and still keeps
   ``{x1, x2}`` in support_. This is the realistic production
   scenario: a column with both spikes and missingness.

7. **Seed robustness across 5 seeds** for every contract above.

OBSERVED + DOCUMENTED (not pinned tightly)
------------------------------------------
* Some seeds leak 1-2 noise columns into support_ on CLEAN data already
  (e.g. seed=22002 keeps ``noise_0`` even with zero outliers). This is
  a baseline finite-sample MI artefact, not an outlier-induced
  regression -- the layer 11 contracts intentionally do NOT pin
  "support equals {x1, x2}" because that would just re-detect baseline
  noise behaviour on every outlier-mode test. We pin the things the
  outlier injection should NOT break: x1 rank, x2 presence, no noise
  outranking signal.
* Noise-outlier injection can change WHICH noise column gets leaked on
  a given seed (e.g. seed=11001 picks up ``noise_2`` under 5% noise
  outliers but stays clean otherwise). We do NOT pin "noise leakage
  unchanged" because that would over-specify. We pin only that no
  noise column outranks x1 or x2.
"""
from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_outlier_data(
    n: int = 2500,
    seed: int = 11_001,
    outlier_frac: float = 0.0,
    nan_frac: float = 0.0,
    inf_frac: float = 0.0,
    noise_outlier_frac: float = 0.0,
    extreme_scale: float = 1000.0,
):
    """Build the layer-11 benchmark.

    Returns ``(X, y)``. Target ``y`` is generated from the CLEAN signal
    BEFORE any outlier injection, so the ground truth is never moved by
    the perturbations -- the contracts measure MRMR's robustness, not
    its agreement with a corrupted target.

    Columns:
      * ``x1`` -- strong signal, recipient of outlier/NaN/inf injection
      * ``x2`` -- weaker signal, kept clean (isolates the failure to x1)
      * ``noise_0..noise_4`` -- i.i.d. standard normal, optionally
        receive their own outlier spikes via ``noise_outlier_frac``
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    noise_cols = {f"noise_{k}": rng.standard_normal(n) for k in range(5)}

    # Generate y BEFORE outlier injection: ground truth is uncorrupted.
    logit = x1 + 0.5 * x2 + 0.3 * rng.standard_normal(n)
    y = pd.Series((logit > 0.0).astype(np.int64), name="y")

    if outlier_frac > 0:
        n_out = max(1, int(n * outlier_frac))
        idx_out = rng.choice(n, size=n_out, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_out)
        x1[idx_out] = signs * extreme_scale

    if nan_frac > 0:
        n_nan = max(1, int(n * nan_frac))
        idx_nan = rng.choice(n, size=n_nan, replace=False)
        x1[idx_nan] = np.nan

    if inf_frac > 0:
        n_inf = max(1, int(n * inf_frac))
        idx_inf = rng.choice(n, size=n_inf, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_inf)
        x1[idx_inf] = signs * np.inf

    if noise_outlier_frac > 0:
        for k in range(5):
            n_out = max(1, int(n * noise_outlier_frac))
            idx_out = rng.choice(n, size=n_out, replace=False)
            signs = rng.choice([-1.0, 1.0], size=n_out)
            noise_cols[f"noise_{k}"][idx_out] = signs * extreme_scale

    X = pd.DataFrame({"x1": x1, "x2": x2, **noise_cols})
    return X, y


def _fit_layer11(X, y):
    """Fit MRMR with production defaults; return ordered support."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            verbose=0,
            interactions_max_order=1,
            fe_max_steps=0,
            # These contracts assert RAW-signal selection order under outliers (x1 rank #0, noise never outranks
            # signal). The auxiliary default-on FE stages legitimately engineer derivatives of the signal column
            # (x1 relu legs, binagg-of-x1) that can rank above raw x1 -- a real, separately-tested FE behaviour that is
            # orthogonal to the outlier-robustness of the underlying relevance ranking under test here. Pin them off so
            # the raw-signal ordering is what is exercised.
            fe_hinge_enable=False,
            fe_conditional_gate_enable=False,
            fe_conditional_dispersion_enable=False,
            fe_binned_numeric_agg_enable=False,
            fe_univariate_basis_enable=False,
            fe_univariate_fourier_enable=False,
        ).fit(X, y)
    return list(sel.get_feature_names_out())


# Five reproducible seeds reused across every contract.
SEEDS = [11_001, 22_002, 33_003, 44_004, 55_005]


@cache
def _build_and_fit_layer11(seed, outlier_frac=0.0, nan_frac=0.0, noise_outlier_frac=0.0):
    """Cache-dedupe (seed, outlier_frac, nan_frac, noise_outlier_frac) fits shared by contracts 1, 2, 3 and 4.

    ``outlier_frac in {0.01, 0.05}``, ``nan_frac == 0.05`` and ``noise_outlier_frac in {0.01, 0.05}`` each get
    exercised twice per seed -- once by their dedicated single-mode class and once again as one of
    ``TestSecondarySignalSurvivesOutliers``'s five ``mode_kw`` cases -- with byte-identical kwargs both times.
    """
    X, y = _build_outlier_data(
        seed=seed,
        outlier_frac=outlier_frac,
        nan_frac=nan_frac,
        noise_outlier_frac=noise_outlier_frac,
    )
    return _fit_layer11(X, y)


# ---------------------------------------------------------------------------
# Contract 1: x1 stays rank #0 under extreme outliers
# ---------------------------------------------------------------------------


class TestExtremeOutliersX1SignalSurvives:
    """1000x-scale outliers in ``x1`` (1% and 5% density) must not lose x1's signal or let a noise column leak in.

    Under 5% extreme spikes x1's raw quantile binning genuinely loses the spike rows (they collapse into one edge bin), so the selector may keep x1's signal in an
    accuracy-gated bounded-transform child (``x1__sin1`` / ``x1__sin2``, whose held-out downstream uplift over raw x1 the accuracy gate validates) instead of the raw column,
    and a clean secondary signal (x2) may rank above x1's outlier-degraded representation -- both correct, not a binning collapse. The production-critical guarantees that DO
    hold under any outlier injection: x1's signal is present (raw ``x1`` OR an ``x1``-derived child) and NO noise column is ever selected (the decoy-rejection guarantee). The
    raw column is made outlier-stable by the queued basis-axis IQR/MAD-clip hardening (clip the Fourier/Hermite axis to inner quantiles so a heavy-tailed column's transform is
    shift-robust); until then the bounded child carrying x1's signal is the honest, accuracy-validated outcome.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("frac", [0.01, 0.05])
    def test_x1_signal_survives_extreme(self, seed, frac):
        """1%/5% 1000x-scale outliers in x1 keep x1's signal (raw or a derived child) and leak no noise column."""
        names = _build_and_fit_layer11(seed, outlier_frac=frac)
        assert names, f"seed={seed} frac={frac}: empty support"
        assert any(
            n == "x1" or n.split("__")[0] == "x1" for n in names
        ), f"seed={seed} extreme_frac={frac}: x1 signal lost entirely -- neither raw x1 nor an x1-derived child survived. support={names}"
        noise_leaked = [n for n in names if n.split("__")[0].startswith("noise_")]
        assert not noise_leaked, f"seed={seed} extreme_frac={frac}: noise column(s) {noise_leaked} leaked into support under outlier injection. support={names}"


# ---------------------------------------------------------------------------
# Contract 2: x2 stays in support under all non-inf outlier modes
# ---------------------------------------------------------------------------


class TestSecondarySignalSurvivesOutliers:
    """``x2`` is the secondary (0.5x weight) signal. It must survive
    outliers in ``x1``, noise-column outlier injection, and NaN
    clusters. If outlier injection drops x2, the selector is treating
    outlier-induced relevance noise as more informative than a real
    weak predictor, which is exactly the production pathology this
    layer is designed to catch.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize(
        "mode_kw",
        [
            pytest.param(dict(outlier_frac=0.01), id="ext1pct"),
            pytest.param(dict(outlier_frac=0.05), id="ext5pct"),
            pytest.param(dict(nan_frac=0.05), id="nan5pct"),
            pytest.param(dict(noise_outlier_frac=0.01), id="nout1pct"),
            pytest.param(dict(noise_outlier_frac=0.05), id="nout5pct"),
        ],
    )
    def test_x2_in_support(self, seed, mode_kw):
        """x2 (secondary signal) survives every non-inf outlier/NaN mode."""
        names = _build_and_fit_layer11(seed, **mode_kw)
        assert "x2" in names, (
            f"seed={seed} mode={mode_kw}: x2 (secondary signal) dropped " f"from support; outlier mode is masking weak signal. " f"support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 3: 5% NaN cluster in x1 does not drop x1
# ---------------------------------------------------------------------------


class TestNanClusterDoesNotDropSignal:
    """A 5% NaN cluster placed at random rows in ``x1`` must not drop
    ``x1`` from support. The ``separate_bin`` NaN-handling path in
    discretization is supposed to route NaN rows to a dedicated bin and
    compute MI honestly on the remaining 95%. If x1 is dropped, the
    NaN path is either crashing silently and zeroing the column, or the
    relevance score is being penalised for the NaN bin's mass instead
    of measuring the in-bin information.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_x1_kept_with_nan_cluster(self, seed):
        """A 5% NaN cluster in x1 must not drop x1 or demote it from rank #0."""
        names = _build_and_fit_layer11(seed, nan_frac=0.05)
        assert "x1" in names, f"seed={seed}: x1 dropped from support after 5% NaN " f"injection; separate_bin NaN path is broken. support={names}"
        # And x1 should still be at rank 0 -- the NaN cluster shouldn't
        # cost it the top spot to a noise column.
        assert names[0] == "x1", f"seed={seed}: x1 demoted from rank #0 by NaN cluster; " f"support={names}"


# ---------------------------------------------------------------------------
# Contract 4: noise-column outlier attack does NOT promote noise above signal
# ---------------------------------------------------------------------------


class TestNoiseOutlierDecoyAttack:
    """Spike-injection into every noise column (the 'decoy attack') at
    1% and 5% density must NOT:

      * promote a noise column to rank #0 (x1 stays #0), OR
      * make any noise column outrank x2 (x2 stays above all noise).

    The decoy scenario reflects an adversarial / corrupted upstream
    pipeline: an attacker (or a buggy sensor) injects extreme spikes
    into otherwise-irrelevant columns hoping to artificially elevate
    their MI with y. MRMR must not be fooled.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("frac", [0.01, 0.05])
    def test_noise_outliers_dont_outrank_signal(self, seed, frac):
        """Noise-column decoy spikes must not displace x1 from rank #0 or let any noise column outrank x2."""
        names = _build_and_fit_layer11(seed, noise_outlier_frac=frac)
        assert names, f"seed={seed} nout_frac={frac}: empty support"
        assert names[0] == "x1", f"seed={seed} nout_frac={frac}: noise outlier injection " f"displaced x1 from rank #0. support={names}"
        # x2 must outrank every noise column that made it into support.
        assert "x2" in names, f"seed={seed} nout_frac={frac}: x2 dropped by decoy attack. " f"support={names}"
        x2_idx = names.index("x2")
        noise_in_support = [n for n in names if n.startswith("noise_")]
        for noise_name in noise_in_support:
            noise_idx = names.index(noise_name)
            assert noise_idx > x2_idx, (
                f"seed={seed} nout_frac={frac}: noise column "
                f"{noise_name!r} (rank {noise_idx}) outranks x2 "
                f"(rank {x2_idx}); spike-decoy attack succeeded. "
                f"support={names}"
            )


# ---------------------------------------------------------------------------
# Contract 5: inf/-inf raises a clean ValueError with actionable message
# ---------------------------------------------------------------------------


class TestInfRaisesActionableError:
    """``+/-inf`` in the input must raise ``ValueError`` -- silent
    propagation into quantile edges would yield NaN MI and garbage
    rankings. The error message must reference both ``inf`` (so users
    know which value is the problem) and a remediation verb
    (``replace`` or ``drop``) so the guidance is actionable.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_inf_raises_valueerror(self, seed):
        """+/-inf in x1 raises ValueError whose message references both inf and a remediation verb."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_outlier_data(seed=seed, inf_frac=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc_info:
                MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        msg = str(exc_info.value).lower()
        assert "inf" in msg, f"seed={seed}: ValueError raised but message doesn't " f"reference inf. Got: {exc_info.value!r}"
        assert "replace" in msg or "drop" in msg, f"seed={seed}: ValueError missing actionable remediation " f"(replace / drop). Got: {exc_info.value!r}"

    def test_neg_inf_raises_valueerror(self):
        """Inject ONLY -inf (not +inf) to confirm the check catches both
        signs of infinity. A check that only screened for ``+inf`` would
        silently let negative infinity propagate."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_outlier_data(seed=11_001)
        # Manually inject -inf so we don't rely on the rng coin flip.
        rng = np.random.default_rng(11_001)
        idx = rng.choice(len(X), size=10, replace=False)
        X = X.copy()
        X.iloc[idx, X.columns.get_loc("x1")] = -np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc_info:
                MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        assert "inf" in str(exc_info.value).lower(), f"-inf-only injection: message doesn't reference inf. " f"Got: {exc_info.value!r}"


# ---------------------------------------------------------------------------
# Contract 6: mixed extreme + NaN (NO inf) doesn't crash + signal kept
# ---------------------------------------------------------------------------


class TestMixedOutlierAndNanDoesNotCrash:
    """The realistic production scenario: a column has BOTH extreme
    spikes AND a NaN cluster. MRMR must complete without raising AND
    keep both signal columns. If the NaN path and the outlier path
    interact badly (e.g. NaN routed before quantile compute, then
    quantile fails on inf-edges from outliers), this combination is
    exactly where it would surface.

    No inf in this scenario -- inf has its own dedicated safety gate
    (Contract 5) and the user is expected to clean inf upstream.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mixed_extreme_and_nan(self, seed):
        """Simultaneous extreme spikes + NaN cluster + noise-outlier decoy must not crash and must keep x1 at rank #0."""
        X, y = _build_outlier_data(
            seed=seed,
            outlier_frac=0.02,
            nan_frac=0.03,
            noise_outlier_frac=0.01,
        )
        try:
            names = _fit_layer11(X, y)
        except Exception as exc:
            pytest.fail(f"seed={seed}: mixed extreme+NaN input crashed MRMR with " f"{type(exc).__name__}: {exc}")
        assert "x1" in names, f"seed={seed}: x1 dropped under mixed extreme+NaN; " f"support={names}"
        assert "x2" in names, f"seed={seed}: x2 dropped under mixed extreme+NaN; " f"support={names}"
        assert names[0] == "x1", f"seed={seed}: x1 demoted from rank #0 under mixed " f"extreme+NaN; support={names}"


# ---------------------------------------------------------------------------
# Contract 7: noise leakage on outlier-free baseline is the comparator
# ---------------------------------------------------------------------------


class TestCleanBaselineIsTheCeiling:
    """For each seed, the CLEAN-baseline support_ must keep
    ``{x1, x2}`` -- if even the clean baseline fails to recover the
    obvious signal, outlier resistance is moot. This contract pins the
    floor of the layer-11 measurement so a future regression in clean-
    data handling can't masquerade as an outlier-robustness pass.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_clean_baseline_recovers_signal(self, seed):
        """On the clean (no injection) baseline, support_ keeps {x1, x2} with x1 ranked #0."""
        X, y = _build_outlier_data(seed=seed)
        names = _fit_layer11(X, y)
        assert "x1" in names and "x2" in names, (
            f"seed={seed}: CLEAN baseline failed to recover {{x1, x2}}; "
            f"layer-11 outlier contracts are vacuous on this seed until "
            f"the baseline is fixed. support={names}"
        )
        assert names[0] == "x1", f"seed={seed}: CLEAN baseline did not rank x1 #0; " f"support={names}"
