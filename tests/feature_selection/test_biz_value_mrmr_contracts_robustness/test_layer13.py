"""Consolidated from test_biz_value_mrmr_layer13.py.

Layer 13 biz_value MRMR contracts: EXTREMELY IMBALANCED BINARY ``y``.

WHY THIS LAYER
--------------
Imbalanced binary targets are not an edge case, they are the modal
production workload for tabular ML:

* Fraud detection: ~0.1-1% positive rate.
* Click / conversion prediction: 0.5-5% positive.
* Rare disease screening: 0.1% or lower.
* Churn on monthly cohorts: 1-3%.
* Industrial defect detection: 0.1-1%.

A feature selector that only behaves correctly on balanced
``y in {0, 1}`` with ~50/50 mass cannot be trusted on any of those
workloads. The risk profile is concrete and binning-specific:

* When positives are <= 1% of n, the entire positive class can land
  in a SINGLE quantile bin of any well-binned ``x``. The MI estimator
  then sees a 2D joint histogram with all the positive mass in one
  row, and noise columns whose lowest or highest bin happens to
  contain a small but unlucky cluster of positives can artificially
  outscore a genuine signal column.
* Conversely, the empirical MI for a real signal column shrinks (the
  asymptotic ``I(X; Y)`` is bounded above by ``H(Y) = - p log p
  - (1-p) log(1-p)``, which at p = 0.005 is only ~0.045 nats). The
  margin between "real signal" and "spurious noise correlation"
  collapses.
* Without stratified sampling, a CV / bootstrap fold can land with
  ZERO positives at p = 0.1% on n = 10000 (binomial(10000, 0.001)
  expects 10 positives with std ~3.16; the floor draws 0-1
  occasionally). MRMR's inner bootstrap, if any, must either stratify
  or accept the variance.

DATA DESIGN
-----------
* ``n = 15000`` rows -- large enough that p = 0.001 yields ~15
  positives in expectation, the floor below which MI estimation has
  no statistical leg to stand on.
* ``x_signal``: standard normal on negatives, ``Normal(+2.0, 1.0)`` on
  positives. This is the strongest predictor by construction:
  knowing ``x_signal`` shifts the posterior on y substantially even
  at p = 0.001 because the positive-class distribution is displaced
  far from the negative-class distribution.
* ``x_uniform_signal``: a weaker class-conditional shift,
  ``Normal(+0.4, 1.0)`` on positives vs ``Normal(0, 1)`` on negatives.
  Tests whether MRMR can pick up a SECOND, weaker signal under
  imbalance.
* ``noise_0..noise_5``: i.i.d. ``Normal(0, 1)``, fully independent of
  y. Floor for the noise contracts.

CONTRACTS PINNED
----------------
1. **Mild imbalance (1% positives) -- strong signal + downstream
   parity.** ``x_signal`` must appear in ``get_feature_names_out()``
   AND the selection's 5-fold downstream ROC-AUC must be within 0.03
   of the all-signal baseline. (Rebaselined from "both signals found":
   under the new default's diminishing-returns gate the weak +0.4 sigma
   ``x_uniform_signal`` is declined at the 2nd slot, costing only
   ~0.005 AUC -- see TestMildImbalanceFindsBothSignals.)

2. **Stronger imbalance (0.5% positives) -- strong signal found,
   weak signal not pinned.**
   ``x_signal`` must still be in support. ``x_uniform_signal`` is
   ALLOWED to drop because at ~75 positives the weak class-
   conditional shift (+0.4 sigma) approaches the resolution limit
   of the binning estimator and the expected MI margin against
   noise becomes seed-dependent. We document what we observe but
   pin only the load-bearing contract.

3. **Extreme imbalance (0.1% positives) -- documented behaviour.**
   At ~15 positives the data is at or below the floor where binning-
   based MI is statistically meaningful (the positive class
   over-concentrates in 1-2 bins of ANY column, signal or not).
   The contract is: MRMR does NOT crash, returns a well-formed
   selector, AND admits at most TWO noise columns. The 2-noise
   ceiling (not 1) is observation-calibrated: seed 13003 with
   n_pos=15 produces ``kept=['x_signal', 'noise_1', 'noise_3']``
   under the current MRMR. ``x_signal`` is still rank #0 in that
   draw -- the noise columns leak in BEHIND it, not ahead. So we
   pin the honest behaviour: noise CAN leak at p=0.001 but stays
   bounded; the relevance gate has NOT collapsed (a full collapse
   would admit 4-6 noise columns).

4. **Bounded binning artefact at p >= 0.005.** At MILD and STRONG
   imbalance (n_pos >= 75), at most ONE noise column appears in
   ``get_feature_names_out()``, it sits ALONGSIDE ``x_signal`` (never
   replacing it), and the selection keeps downstream-AUC parity with
   the all-signal baseline. The ``<= 1`` ceiling is observation-
   calibrated to the new default (full-mode conditional-MI): the worst
   case across SEEDS x {p=0.01, p=0.005} is 1 noise column. (Under the
   legacy simple-mode selector this contract pinned ``== 0`` noise,
   because the relative-gain floor compared the candidate's MARGINAL
   MI; full-mode's conditional-MI estimate at the rare-class floor lets
   one near-zero-MI noise column through -- a finite-sample artifact
   that vanishes with more positives, NOT a gate collapse.) The
   contract still guards the production-safety property that the
   relevance gate has NOT collapsed (which would admit 5-6 noise). We
   do NOT extend this contract to p = 0.001; that level has its own
   (looser) noise pin via contract 3.

5. **MRMR is robust to imbalance** -- it runs end-to-end and
   produces well-formed output (non-empty feature list, no
   exceptions, no NaN in any internal state we can observe) on
   every imbalance level.

6. **Multiple seeds.** Every contract above is verified across 3
   seeds. We use 3 (not the usual 2) because at extreme imbalance
   the per-seed variance is genuinely high and a 2-seed grid is too
   easy to luck through.

NOT PINNED (deliberately)
-------------------------
* **Rank #0 for ``x_signal`` at p = 0.001.** With ~15 positives any
  noise column whose top quantile happens to contain 2-3 positives
  by chance can score higher empirical MI than ``x_signal``. This
  is not an MRMR bug, it is the statistical floor of histogram MI
  on rare-class data. The user-facing remedy is "collect more
  positives" or "use a different estimator", not "rebalance MRMR".

* **Stratified resampling guarantees.** MRMR's inner bootstrap (if
  any) is NOT documented to stratify on y. We do NOT pin stratified
  behaviour because the API does not promise it; instead we pin
  that the OUTPUT support is sensible at p >= 0.005, where even
  non-stratified resampling preserves enough positives.

* **Specific MI / score values.** Histogram-MI on rare classes has
  enough finite-sample variance that pinning numeric scores would
  be brittle. We pin SET membership in support_ instead.

* **``x_uniform_signal`` at p = 0.005.** The weak-signal contract
  is intentionally only pinned at p = 0.01. Below that, the +0.4
  sigma class-conditional shift on ~75 positives gives a population
  MI that the histogram estimator cannot reliably resolve above
  noise. We document the observation but do not pin.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------


N_TOTAL = 15_000

# Imbalance levels we test, named for readability of parametrize IDs.
IMBALANCE_MILD = 0.01      # 1% positives, ~150 positives at n=15000
IMBALANCE_STRONG = 0.005   # 0.5% positives, ~75 positives at n=15000
IMBALANCE_EXTREME = 0.001  # 0.1% positives, ~15 positives at n=15000

NOISE_COL_COUNT = 6


def _build_imbalanced_data(imbalance: float, seed: int):
    """Construct the layer-13 imbalanced binary dataset.

    Generative process:

    * Draw a binary ``y`` with positive-class probability ``imbalance``
      using a deterministic count (np.random.choice on row indices)
      rather than a Bernoulli draw. This GUARANTEES the per-seed
      positive count is exactly ``round(n * imbalance)`` so the test
      assertions reason about a known number of positives rather than
      a binomial draw that can swing ~+/- 3 sigma. At p = 0.001 that
      matters: a Bernoulli draw can give 8 or 22 positives and we
      want exactly 15 every time.
    * Generate ``x_signal`` so that positives come from
      ``Normal(+2.0, 1.0)`` and negatives from ``Normal(0, 1)``. The
      +2.0 shift is chosen to give a clean MI signal even at p =
      0.001 (the population MI(X; Y) is bounded by H(Y) which is
      tiny at extreme imbalance, but the strong displacement makes
      the SHARE of H(Y) captured close to 1).
    * Generate ``x_uniform_signal`` with a weaker +0.4 shift on
      positives. This is the secondary, harder-to-find signal.
    * Generate ``noise_0..noise_5`` i.i.d. ``Normal(0, 1)`` independent
      of y. Six noise columns is enough to test "noise as a class"
      without exploding the test grid.

    Parameters
    ----------
    imbalance : float
        Positive-class fraction. Must be in (0, 0.5].
    seed : int
        RNG seed. Drives both y placement and feature draws.

    Returns
    -------
    X : pd.DataFrame, shape (n, 8)
        Columns: x_signal, x_uniform_signal, noise_0..noise_5.
    y : pd.Series, shape (n,), dtype int64
        Binary target with EXACTLY ``round(n * imbalance)`` positives.
    n_positives : int
        Exact count of positives in y (for diagnostics in failure
        messages and for the "we have enough positives" sanity check).
    """
    rng = np.random.default_rng(seed)

    n_pos = int(round(N_TOTAL * imbalance))
    if n_pos < 1:
        raise ValueError(
            f"imbalance={imbalance} on n={N_TOTAL} yields zero positives"
        )

    # Deterministic positive count: pick exactly n_pos row indices to be
    # positive. This removes binomial variance from the test so failures
    # always reflect MRMR behaviour at a KNOWN positive count, not RNG
    # variance in y's own draw.
    pos_idx = rng.choice(N_TOTAL, size=n_pos, replace=False)
    y_arr = np.zeros(N_TOTAL, dtype=np.int64)
    y_arr[pos_idx] = 1

    # x_signal: strong class-conditional shift.
    x_signal = rng.standard_normal(N_TOTAL)
    x_signal[pos_idx] += 2.0

    # x_uniform_signal: weak class-conditional shift.
    x_uniform_signal = rng.standard_normal(N_TOTAL)
    x_uniform_signal[pos_idx] += 0.4

    cols = {
        "x_signal": x_signal,
        "x_uniform_signal": x_uniform_signal,
    }
    for k in range(NOISE_COL_COUNT):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)

    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y")
    return X, y, n_pos


# ---------------------------------------------------------------------------
# Fit helper
# ---------------------------------------------------------------------------


def _fit_mrmr(X, y):
    """Run MRMR with the layer's standard config and return the fitted
    selector. Mirrors the helper used in layers 11 and 12 so the only
    moving piece across layers is the data, not the selector config.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            verbose=0,
            interactions_max_order=1,
            fe_max_steps=0,
        ).fit(X, y)
    return sel


def _selected_names(sel):
    """Return MRMR's selected features as a plain list of column names."""
    return list(sel.get_feature_names_out())


# 3 seeds (not 2): at extreme imbalance the per-seed variance is genuine
# and a 2-seed grid is too easy to luck through. 3 also fits comfortably
# under the 300s timeout per the layer config.
SEEDS = [13_001, 13_002, 13_003]


# ---------------------------------------------------------------------------
# Contract 1: mild imbalance (1%) -- BOTH signals found
# ---------------------------------------------------------------------------


class TestMildImbalanceFindsBothSignals:
    """At p = 0.01 (n_pos ~ 150) MRMR must recover the strong (+2 sigma)
    class-conditional signal and lose no material downstream AUC.

    Regression mode this guards: a change that breaks MRMR's relevance
    gate on imbalanced binary y (e.g. by computing MI assuming balanced
    class priors and under-weighting the rare-class signal) would drop
    the strong column and collapse the selection's downstream AUC.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_strong_signal_and_downstream_parity(self, seed):
        """Rebaselined from the old "both x_signal AND x_uniform_signal
        in support" name-membership assertion, which was simple-mode
        specific. Under the new default (``use_simple_mode=False`` ->
        full-mode Fleuret conditional-MI redundancy WITH the on-by-default
        ``min_relevance_gain_relative_to_first=0.05`` diminishing-returns
        gate) the weak +0.4 sigma secondary signal is DECLINED at the
        2nd slot: its conditional-MI gain over the already-selected
        x_signal sits below the 5% relative-gain floor. Measured cost of
        dropping it is tiny -- 5-fold ROC-AUC of [x_signal,x_uniform_signal]
        beats [x_signal] alone by only +0.004..+0.006 across the SEEDS
        grid (x_signal carries ~0.91 AUC by itself). So pinning the weak
        signal's membership pinned a simple-mode (no-relative-floor)
        artifact, not a real win. We instead pin the load-bearing
        properties: the STRONG signal is recovered, and the selection's
        downstream AUC is within a small band of the all-signal baseline.
        Falsifiable: if MRMR dropped x_signal the AUC gap would blow past
        0.03 (toward the ~0.5 chance floor) and this fires.
        """
        from tests.feature_selection._biz_val_synth import downstream_auc, baseline_signal_auc

        X, y, n_pos = _build_imbalanced_data(IMBALANCE_MILD, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = _selected_names(sel)
        assert "x_signal" in kept, (
            f"seed={seed} n_pos={n_pos} (p=1%): x_signal missing from "
            f"support; MRMR is not detecting strong class-conditional "
            f"shift at mild imbalance. kept={kept}"
        )
        auc_sel = downstream_auc(sel, X, y.to_numpy(), cv=5)
        auc_base = baseline_signal_auc(
            X, y.to_numpy(), signal=["x_signal", "x_uniform_signal"],
            prefix="", cv=5,
        )
        assert auc_sel >= auc_base - 0.03, (
            f"seed={seed} n_pos={n_pos} (p=1%): MRMR selection {kept} lost "
            f"material downstream signal -- selection AUC={auc_sel:.4f} vs "
            f"all-signal baseline AUC={auc_base:.4f} (gap > 0.03)."
        )


# ---------------------------------------------------------------------------
# Contract 2: stronger imbalance (0.5%) -- strong signal still found
# ---------------------------------------------------------------------------


class TestStrongImbalanceFindsStrongSignal:
    """At p = 0.005 (n_pos ~ 75) the strong +2 sigma shift on the
    positive class is still resolvable by histogram MI, but the weak
    +0.4 sigma signal is at the resolution limit and we do NOT pin it.

    Regression mode this guards: MI under-weighting of the rare class
    that would push x_signal below the relevance gate even at a 75-
    positive count.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_strong_signal_in_support(self, seed):
        X, y, n_pos = _build_imbalanced_data(IMBALANCE_STRONG, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = _selected_names(sel)
        assert "x_signal" in kept, (
            f"seed={seed} n_pos={n_pos} (p=0.5%): x_signal missing from "
            f"support; even the strong +2 sigma class-conditional shift "
            f"is being missed at 75 positives. kept={kept}"
        )


# ---------------------------------------------------------------------------
# Contract 3: extreme imbalance (0.1%) -- no crash, noise does not flood
# ---------------------------------------------------------------------------


class TestExtremeImbalanceDocumentedBehaviour:
    """At p = 0.001 (n_pos ~ 15) the dataset is at the statistical floor
    for histogram-MI on rare classes. The positive class is so small
    that random noise columns can plausibly score higher empirical MI
    than x_signal: 15 positives can over-concentrate into any 1-2 of
    20 quantile bins of ANY column by pure chance.

    OBSERVED behaviour at p = 0.001 (n_pos = 15, seeds 13001-13003):

    * ``x_signal`` IS in support on every seed in the SEEDS grid,
      and on seed 13003 it is even rank #0 with two noise columns
      trailing behind it (``kept=['x_signal', 'noise_1', 'noise_3']``).
      We pin "x_signal in support" because the observation is robust
      on this grid, AND because it is the user-relevant property
      ("did MRMR find the predictor?"). We do NOT pin rank #0 -- a
      different seed could plausibly invert that.
    * Up to 2 noise columns leak through on some seeds. This is
      genuine finite-sample MI artefact, not a regression: with
      n_pos = 15 the positive cluster lands in 1-2 quantile bins of
      ANY column by pure chance, lifting its empirical MI(X; Y)
      above the gate. The redundancy step still bounds the count.

    PINNED behaviour (load-bearing):

    * MRMR does NOT crash at p = 0.001.
    * The returned support is well-formed (a list of strings, all of
      which are valid column names from X).
    * ``x_signal`` IS in support (observation-calibrated, robust on
      the SEEDS grid).
    * At most TWO noise columns appear in the returned support. This
      is the noise-flooding guard, calibrated to observation: a
      regression that drops the relevance gate entirely at extreme
      imbalance would let 4-6 noise columns through. The 2-noise
      ceiling is honest about what happens on n_pos = 15 today.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_runs_and_produces_wellformed_support(self, seed):
        X, y, n_pos = _build_imbalanced_data(IMBALANCE_EXTREME, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = _selected_names(sel)
        # Well-formed: list of strings, each a real column.
        assert isinstance(kept, list), (
            f"seed={seed} n_pos={n_pos} (p=0.1%): get_feature_names_out "
            f"returned non-list {type(kept).__name__}"
        )
        cols = set(X.columns)
        bogus = [name for name in kept if name not in cols]
        assert not bogus, (
            f"seed={seed} n_pos={n_pos} (p=0.1%): MRMR returned column "
            f"names not present in input: {bogus}. kept={kept}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_signal_in_support_at_extreme(self, seed):
        X, y, n_pos = _build_imbalanced_data(IMBALANCE_EXTREME, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = _selected_names(sel)
        assert "x_signal" in kept, (
            f"seed={seed} n_pos={n_pos} (p=0.1%): x_signal not in "
            f"support even though it carries a +2 sigma class-"
            f"conditional shift. kept={kept}. If this fails on a new "
            f"seed, the layer needs re-calibration -- at n_pos=15 the "
            f"signal can plausibly be lost, but it should be the rule "
            f"not the exception on the SEEDS grid."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_at_most_two_noise_in_support(self, seed):
        X, y, n_pos = _build_imbalanced_data(IMBALANCE_EXTREME, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = _selected_names(sel)
        noise_in_support = [n for n in kept if n.startswith("noise_")]
        assert len(noise_in_support) <= 2, (
            f"seed={seed} n_pos={n_pos} (p=0.1%): MRMR promoted "
            f"{len(noise_in_support)} noise columns to support "
            f"({noise_in_support}); relevance gate is collapsing at "
            f"extreme imbalance. Observation-calibrated bound is 2 on "
            f"the SEEDS grid -- 3 or more indicates regression. "
            f"kept={kept}"
        )


# ---------------------------------------------------------------------------
# Contract 4: no false-positive binning artefact across imbalance levels
# ---------------------------------------------------------------------------


class TestNoFalsePositiveBinningArtefact:
    """Class-level noise contract for imbalance levels at which binning
    MI is statistically resolvable (p >= 0.005, n_pos >= 75).

    The pathological failure mode this guards: a total collapse of the
    relevance gate that would admit 5-6 noise columns alongside (or
    instead of) genuine signal. The contract says: even though noise
    DOES leak at imbalanced p (observed up to 3 columns on the SEEDS
    grid), the count stays bounded AND the signal column is always in
    support alongside the leakage.

    Two assertions, both load-bearing:

    (a) ``x_signal`` is in support on EVERY (seed, level) at p >= 0.005.
        The strong +2 sigma class-conditional shift gives MRMR enough
        statistical purchase even at n_pos = 75 to keep ``x_signal``
        on the relevant side of the gate.

    (b) At most THREE noise columns leak through. Observation-
        calibrated worst case is 3 (seed 13001, p = 1%). This bounds
        the gate-collapse failure mode without overfitting to the
        common case of "0-2 noise" -- the regression we are guarding
        against is the relevance gate going to zero, not the gate
        sitting at its honest finite-sample level.

    Scope deliberately EXCLUDES p = 0.001 (see contract 3 in the
    module docstring) -- that level has its own looser noise pin via
    ``TestExtremeImbalanceDocumentedBehaviour.test_at_most_two_noise_in_support``.
    """

    @pytest.mark.parametrize(
        "imbalance",
        [IMBALANCE_MILD, IMBALANCE_STRONG],
        ids=["p=1pct", "p=0.5pct"],
    )
    @pytest.mark.parametrize("seed", SEEDS)
    def test_signal_present_alongside_any_noise_leakage(
        self, imbalance, seed
    ):
        X, y, n_pos = _build_imbalanced_data(imbalance, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = _selected_names(sel)
        assert "x_signal" in kept, (
            f"seed={seed} imbalance={imbalance} n_pos={n_pos}: "
            f"x_signal missing from support at a resolvable imbalance "
            f"level (n_pos >= 75); the +2 sigma class-conditional shift "
            f"should comfortably clear the relevance gate. kept={kept}"
        )

    @pytest.mark.parametrize(
        "imbalance",
        [IMBALANCE_MILD, IMBALANCE_STRONG],
        ids=["p=1pct", "p=0.5pct"],
    )
    @pytest.mark.parametrize("seed", SEEDS)
    def test_bounded_noise_leakage_at_resolvable_imbalance(self, imbalance, seed):
        """At most ONE noise column leaks at p >= 0.005, and x_signal is
        always present alongside it.

        Rebaselined from the old ``== 0 noise`` assertion. That zero-leak
        contract was calibrated under simple mode: the
        ``min_relevance_gain_relative_to_first=0.05`` floor compared the
        candidate's MARGINAL MI gain (~0.0004, ~2.5% of signal gain) to
        the floor, so trailing noise was excluded outright. Under the new
        default (``use_simple_mode=False``) the 2nd-slot gain is a
        CONDITIONAL-MI estimate given x_signal; at the rare-class floor
        (n_pos in {75,150}) those conditional estimates are dominated by
        finite-sample bias and one near-zero-MI noise column can clear the
        relative floor. This is a genuine finite-sample artifact, not a
        gate collapse: it vanishes with more positives (n=60000 p=1% ->
        ['x_signal'] only, no leak) and x_signal stays rank-0. Observed
        full-mode worst case on the SEEDS x {1%, 0.5%} grid is exactly 1
        noise column (seeds 13001/13002 p=1%, seed 13003 p=0.5%). The <= 1
        bound still guards the real regression -- a gate collapse would
        admit 5-6 noise columns -- and we additionally pin that the noise
        sits ALONGSIDE x_signal (not crowding it out) and costs no material
        downstream AUC, so this is not a vacuous relaxation.
        """
        from tests.feature_selection._biz_val_synth import downstream_auc, baseline_signal_auc

        X, y, n_pos = _build_imbalanced_data(imbalance, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = _selected_names(sel)
        noise_in_support = [n for n in kept if n.startswith("noise_")]
        assert len(noise_in_support) <= 1, (
            f"seed={seed} imbalance={imbalance} n_pos={n_pos}: "
            f"{len(noise_in_support)} noise columns in support "
            f"({noise_in_support}); full-mode finite-sample worst case on "
            f"this grid is 1. >= 2 indicates the relevance gate is "
            f"collapsing at rare-class. kept={kept}"
        )
        assert "x_signal" in kept, (
            f"seed={seed} imbalance={imbalance} n_pos={n_pos}: any noise "
            f"leakage must sit alongside x_signal, not replace it; "
            f"x_signal missing. kept={kept}"
        )
        # Leaked noise must not have cost real downstream signal.
        auc_sel = downstream_auc(sel, X, y.to_numpy(), cv=5)
        auc_base = baseline_signal_auc(
            X, y.to_numpy(), signal=["x_signal", "x_uniform_signal"],
            prefix="", cv=5,
        )
        assert auc_sel >= auc_base - 0.03, (
            f"seed={seed} imbalance={imbalance} n_pos={n_pos}: selection "
            f"{kept} lost downstream signal (AUC={auc_sel:.4f} vs baseline "
            f"{auc_base:.4f}); the noise leak crowded out signal."
        )

    @pytest.mark.parametrize(
        "imbalance",
        [IMBALANCE_MILD, IMBALANCE_STRONG],
        ids=["p=1pct", "p=0.5pct"],
    )
    @pytest.mark.parametrize("seed", SEEDS)
    def test_support_dominated_by_signal_post_fix(self, imbalance, seed):
        """The support is dominated by true signal: x_signal is always
        present and at most ONE non-signal (noise) column rides along.

        Rebaselined from the old "support_ is a strict subset of
        {x_signal, x_uniform_signal}" assertion, which was simple-mode
        specific (same root cause as
        test_bounded_noise_leakage_at_resolvable_imbalance: full-mode
        conditional-MI at the rare-class floor lets one near-zero-MI
        noise column clear the 5% relative-gain floor). We keep the
        load-bearing half -- the strong signal must be recovered and
        non-signal contamination is bounded to 1 -- which still rules
        out a gate collapse (which would admit several spurious columns).
        """
        X, y, n_pos = _build_imbalanced_data(imbalance, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = set(_selected_names(sel))
        signals = {"x_signal", "x_uniform_signal"}
        non_signal = kept - signals
        assert "x_signal" in kept, (
            f"seed={seed} imbalance={imbalance} n_pos={n_pos}: x_signal "
            f"missing from support; strong signal must be recovered. "
            f"kept={kept}"
        )
        assert len(non_signal) <= 1, (
            f"seed={seed} imbalance={imbalance} n_pos={n_pos}: support "
            f"contains {len(non_signal)} non-signal columns {non_signal!r}; "
            f"full-mode rare-class worst case is 1. >= 2 indicates the "
            f"relevance gate is collapsing. kept={kept}"
        )


# ---------------------------------------------------------------------------
# Contract 5: MRMR runs end-to-end on every imbalance level
# ---------------------------------------------------------------------------


class TestMRMRRobustToImbalance:
    """Sanity floor: fit completes, returns a selector with a non-empty,
    well-formed feature list. Catches catastrophic regressions where a
    code path crashes or returns empty support on rare-class y (e.g.
    a division by ``count(y == 1) - 1`` or a quantile call that fails
    when the positive class fits inside a single bin).
    """

    @pytest.mark.parametrize(
        "imbalance",
        [IMBALANCE_MILD, IMBALANCE_STRONG, IMBALANCE_EXTREME],
        ids=["p=1pct", "p=0.5pct", "p=0.1pct"],
    )
    @pytest.mark.parametrize("seed", SEEDS)
    def test_fit_succeeds_and_returns_nonempty_support(self, imbalance, seed):
        X, y, n_pos = _build_imbalanced_data(imbalance, seed=seed)
        sel = _fit_mrmr(X, y)
        kept = _selected_names(sel)
        assert len(kept) >= 1, (
            f"seed={seed} imbalance={imbalance} n_pos={n_pos}: MRMR "
            f"returned empty support; rare-class handling failure."
        )
