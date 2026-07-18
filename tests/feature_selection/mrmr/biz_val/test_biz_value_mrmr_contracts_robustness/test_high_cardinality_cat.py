"""Consolidated from test_biz_value_mrmr_layer10.py.

Layer 10 biz_value MRMR contracts: HIGH-CARDINALITY CATEGORICAL.

WHY THIS LAYER
--------------
High-cardinality categoricals (``user_id``, ``item_id``, ``zip_code``,
session ID, hashed device fingerprint) are present in virtually every
production tabular pipeline. They are famously dangerous for MI-based
selectors because raw mutual information
``I(X; Y) <= min(H(X), H(Y))`` grows with ``H(X)``: a 1000-level column
where most levels appear 1-3 times has near-maximal entropy AND - on
finite samples - an artifactually high empirical MI with any target
just because each (level, y_value) pair is rare and the plug-in MI
estimator over-counts these singletons as "informative".

A naive selector either:

* Picks the ID column as the top feature (the classic "user_id
  hijacks the model" pattern), pushing genuine signal out of the
  short-list budget;
* Hands the ID column to downstream model + collinear engineered
  features that the cat-FE pair search materialises from it (worst
  case: O(N^2) pair candidates where N = number of unique IDs);
* Or quietly trims signal columns it deems redundant with the
  high-MI ID, since MRMR's redundancy term sees them as "explained
  by user_id".

The data we build for this layer makes the hijack risk concrete:

* ``user_id`` -- 1200 levels, each level appears 1-3 times, NOT
  predictive of y. (Empirically: ``sklearn.mutual_info_score(user_id,
  y) = 0.328`` vs ``mutual_info_score(region, y) = 0.056`` - the
  hijacker has ~5.9x higher raw MI than the real signal despite
  being pure noise.)
* ``region`` -- 50 levels, with TRUE signal (a small set of "hot"
  levels predicts y).
* 2 informative numeric features + 8 noise numeric features.

True informative set: ``{region, num_signal_1, num_signal_2}``.
Anything that promotes ``user_id`` above these is the hijack we want
to detect.

CONTRACTS PINNED
----------------
Two complementary modes, because MRMR ships TWO production defaults:

A. Default config (``cat_fe_config=None`` -> ``CatFEConfig(enable=True)``):
   the cat-FE step refuses to run on a column with ``nbins > sqrt(n)*2``
   and raises ``ValueError`` with a "drop the column or reconsider
   whether it's truly categorical" message. This is a STRONG safety
   contract - we pin both that the error is raised and that the
   message references high cardinality so the user has actionable
   guidance.

B. cat-FE disabled (``CatFEConfig(enable=False)``):
   MRMR proceeds, and we pin its empirical behaviour:

   1. ``num_signal_1`` and ``num_signal_2`` are always selected
      (numeric signal must survive the high-card distractor).
   2. ``region`` is always selected (medium-card categorical with
      real signal is NOT crowded out by the high-card distractor).
   3. ``user_id`` does NOT rank #1 on the majority of seeds (4/5 in
      our sweep), confirming that even on raw MI, MRMR's redundancy
      term plus screening prevents the worst-case hijack on most
      datasets.
   4. The OBSERVED + DOCUMENTED limitation: on seed=101 the high-card
      ``user_id`` DOES leak into ``support_``. We pin this as a known
      behaviour (test asserts it on that specific seed) rather than
      xfail, because:
        * the leak is a finite-sample MI artefact, not a code bug;
        * the default ``cat_fe_config`` already refuses to run on
          such data (contract A above), so production users hit the
          safety gate before the leak occurs;
        * users who opt out of cat-FE inherit the cardinality-bias
          risk of any MI-based selector and need to know about it.

   5. Switching ``mi_normalization='su'`` (Symmetric Uncertainty,
      Witten-Frank-Hall 2011) demotes ``user_id`` from rank #0 to
      rank #2 on seed=101, but does NOT remove it. SU helps with
      ordering but is not a full bias correction; pinning this avoids
      the temptation to oversell SU as a fix.

REFERENCES
----------
* The ``cat_interactions._select_candidate_indices`` guard at
  ``nbins > sqrt(n) * 2`` is the production-side defence.
* The MI-normalisation knob is documented in
  ``MRMR.__init__`` around the ``mi_normalization`` parameter.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest


@cache
def _build_highcard_data(n: int = 2500, seed: int = 101):
    """Build the layer-10 benchmark.

    Returns ``(X, y)`` where:
    * ``user_id`` is a category dtype with ~1200 levels (hijack risk).
    * ``region`` is a category dtype with 50 levels, with REAL signal
      (a small "hot" subset predicts y).
    * ``num_signal_1`` / ``num_signal_2`` carry direct signal.
    * ``noise_0..noise_7`` are i.i.d. standard normal (no signal).
    """
    rng = np.random.default_rng(seed)

    # high-cardinality user_id: 1200 levels, each appears 1-3 times.
    n_levels = 1200
    reps = rng.integers(1, 4, size=n_levels)
    pool = np.repeat(np.arange(n_levels), reps)
    rng.shuffle(pool)
    user_vals = pool[:n]
    if len(user_vals) < n:
        # pad if reps undershot n (extreme RNG draws).
        pad = rng.integers(0, n_levels, size=n - len(user_vals))
        user_vals = np.concatenate([user_vals, pad])
    user_id = pd.Series([f"u_{i}" for i in user_vals[:n]], name="user_id").astype("category")

    # medium-cardinality region: 50 levels, 6 of them carry signal.
    region_vals = rng.integers(0, 50, size=n)
    hot_levels = {3, 7, 11, 19, 27, 41}
    region_signal = np.array([1.5 if v in hot_levels else 0.0 for v in region_vals])
    region = pd.Series([f"r_{v}" for v in region_vals], name="region").astype("category")

    # numeric signal + noise.
    num_signal_1 = rng.standard_normal(n)
    num_signal_2 = rng.standard_normal(n)
    noise = {f"noise_{k}": rng.standard_normal(n) for k in range(8)}

    cols = {
        "user_id": user_id,
        "region": region,
        "num_signal_1": num_signal_1,
        "num_signal_2": num_signal_2,
    }
    cols.update(noise)
    X = pd.DataFrame(cols)

    # y depends ONLY on region (hot levels) + num_signal_1 + num_signal_2.
    logit = region_signal + 0.9 * num_signal_1 + 0.7 * num_signal_2 + 0.3 * rng.standard_normal(n)
    y = pd.Series((logit > 0.0).astype(np.int64), name="y")
    return X, y


def _fit_layer10(X, y, *, mi_normalization: str = "none"):
    """Common MRMR fit with cat-FE disabled (contract B path)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig

    cfg = CatFEConfig(enable=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            verbose=0,
            interactions_max_order=1,
            fe_max_steps=0,
            cat_fe_config=cfg,
            mi_normalization=mi_normalization,
        ).fit(X, y)
    return list(sel.get_feature_names_out())


@cache
def _build_and_fit_layer10(seed: int) -> tuple:
    """Cached ``(X, y, names)`` for the default ``mi_normalization="none"`` fit, per seed.

    9 test methods across 5 classes re-derive different contracts from the SAME deterministic
    (data, fit) pair for a given seed -- up to 7x redundant identical MRMR.fit() calls per seed
    before this cache. Nothing downstream mutates X/y/names in place.
    """
    X, y = _build_highcard_data(seed=seed)
    names = _fit_layer10(X, y)
    return X, y, names


# ---------------------------------------------------------------------------
# Contract A: default config REFUSES high-card categoricals
# ---------------------------------------------------------------------------


class TestHighCardDefaultRefuses:
    """The production default ``on_high_cardinality="skip"`` (set in
    ``cat_fe_state``, commit 31910ce2 "high-card-cat crash" coverage-gap
    fix) handles a 1200-level cat column WITHOUT crashing: cat-FE skips it
    and it flows through as an ordinary column the relevance screen drops,
    so a raw frame carrying an id / hash / free-text column no longer
    hard-fails the whole fit. The earlier ``raise`` default is preserved as
    an opt-in via ``cat_fe_config`` for callers who want the strict gate."""

    def test_default_skips_highcard_without_crashing(self):
        """``MRMR()`` with the default ``cat_fe_config=None`` must NOT crash
        on a 1200-level categorical column. The high-card column is skipped
        by cat-FE and dropped by the relevance screen (not in support), and
        the genuine numeric signals are still selected -- no garbage MI on a
        column that violates the int16 ceiling, achieved by skipping rather
        than by aborting the fit.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_highcard_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        support = list(sel.get_feature_names_out())
        # The 1200-level ``user_id`` must not survive into support (it carries no
        # generalising signal and would violate the int16 binning ceiling).
        assert not any("user_id" in str(c) for c in support), f"1200-level high-card column leaked into support: {support}"
        # The genuine numeric signals must still be recovered.
        assert any("num_signal" in str(c) for c in support), f"high-card handling dropped the genuine signals too: {support}"


# ---------------------------------------------------------------------------
# Contract B: cat-FE disabled - numeric + region signals survive
# ---------------------------------------------------------------------------


class TestHighCardCatFEDisabledNumericSignalsSurvive:
    """With the cat-FE safety gate opted out (``CatFEConfig(enable=False)``),
    the two informative numeric features must still appear in ``support_``
    - the high-card distractor doesn't crowd them out."""

    def test_num_signal_1_in_support(self):
        """num_signal_1 (true numeric signal) survives the high-card user_id distractor."""
        _X, _y, names = _build_and_fit_layer10(101)
        assert "num_signal_1" in names, f"num_signal_1 (true signal) missing from support; the high-card user_id hijacked the budget. support={names}"

    def test_num_signal_2_in_support(self):
        """num_signal_2 (true numeric signal) survives the high-card user_id distractor."""
        _X, _y, names = _build_and_fit_layer10(101)
        assert "num_signal_2" in names, f"num_signal_2 (true signal) missing from support; support={names}"


class TestHighCardCatFEDisabledRegionSurvives:
    """``region`` is the medium-card categorical with real signal. It
    must survive the high-card distractor - if MRMR's redundancy term
    is wrongly treating region as 'explained by user_id' it'll get
    dropped, and the model loses the only real categorical predictor."""

    def test_region_in_support(self):
        """region (medium-card categorical with real signal) is not crowded out."""
        _X, _y, names = _build_and_fit_layer10(101)
        assert (
            "region" in names
        ), f"region (medium-card cat with real signal) dropped; the high-card user_id likely crowded it out via false redundancy. support={names}"


class TestHighCardCatFEDisabledSeedRobustness:
    """Across 5 seeds, the 3 truly-informative features
    ({num_signal_1, num_signal_2, region}) must all survive. This is
    the strong stability claim: hijacking is a finite-sample MI
    artefact and can shuffle the ordering, but real-signal columns
    must be kept on every seed."""

    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_true_signals_kept_across_seeds(self, seed):
        """Contract B.1/B.2: all 3 true-signal columns survive on every seed."""
        _X, _y, names = _build_and_fit_layer10(seed)
        true_signals = {"num_signal_1", "num_signal_2", "region"}
        missing = true_signals - set(names)
        assert not missing, f"seed={seed}: true-signal column(s) {missing} missing from support; support={names}"

    @pytest.mark.parametrize("seed", [202, 303, 404, 505])
    def test_user_id_not_top_pick_majority_seeds(self, seed):
        """On 4 of 5 seeds, ``user_id`` must NOT be MRMR's top pick.
        Even with raw MI heavily biased toward high cardinality, the
        screening + redundancy pipeline keeps the genuine numeric
        signals at the top for the majority of seeds.

        Seed 101 is excluded from this contract because it documents
        the failure mode (see TestHighCardCatFEDisabledKnownHijack
        below).
        """
        _X, _y, names = _build_and_fit_layer10(seed)
        assert names, f"seed={seed}: empty support"
        assert names[0] != "user_id", f"seed={seed}: user_id ranked #1 -- the cardinality-bias hijack escaped onto a normally-clean seed; support={names}"


# ---------------------------------------------------------------------------
# Documented limitation: known hijack on seed=101 + partial SU mitigation
# ---------------------------------------------------------------------------


class TestHighCardHijackFullyResolved:
    """2026-05-30: high-card hijack FULLY RESOLVED by the screening-
    level cardinality-bias pre-screen (cells > 0.5*n filter) +
    Miller-Madow MM-bias gate.

    Layered defenses (any one of which kills user_id on its own):

    1. PRE-SCREEN (cells > 0.5*n): user_id has 1200 levels x 2-level y
       = 2400 joint cells at n=2500. 2400 > 1250 (0.5*n) so user_id is
       refused at the candidate-generation step BEFORE any MI is even
       computed. This is the load-bearing fix - principled hard limit:
       a contingency table with more cells than half the samples has
       <2 expected occupancy per cell, making plug-in MI dominated by
       finite-sample artefact. Matches the cat-FE safety-gate
       convention (cat_interactions.py:167 refuses nbins > 2*sqrt(n)).

    2. MM-CORRECTION AT GATE (defense in depth): even if pre-screen
       were disabled, the Miller-Madow subtraction would still demote
       user_id to corrected-MI ~0.088 vs num_signal_1's corrected ~0.185,
       so user_id would not anchor at rank #0.

    3. RELATIVE-GAIN FLOOR: 5% of max(corrected gains so far) catches
       trailing noise/bias residual.

    A regression in any layer would surface: user_id reappearing in
    support_ on ANY seed = pre-screen filter is broken or has been
    threshold-relaxed past the 0.5*n cell budget.
    """

    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_user_id_never_in_support(self, seed):
        """The cells > 0.5*n pre-screen keeps user_id out of support_ on every seed."""
        _X, _y, names = _build_and_fit_layer10(seed)
        assert "user_id" not in names, (
            f"seed={seed}: user_id leaked into support_ despite the "
            f"cells > 0.5*n pre-screen filter. n=2500, user_id has "
            f"1200 levels x 2 y-classes = 2400 cells > 1250 - should be "
            f"refused at candidate generation. support={names}"
        )

    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_numeric_signals_survive(self, seed):
        """Negative-control: pre-screen + MM-correction must NOT cut
        the real numeric signals. Both num_signal_1 and num_signal_2
        survive across all seeds (their effective binned cardinality
        is in {3, 5, 10, 20} - well within the cell budget).
        """
        _X, _y, names = _build_and_fit_layer10(seed)
        kept = {"num_signal_1", "num_signal_2"} & set(names)
        assert kept == {"num_signal_1", "num_signal_2"}, (
            f"seed={seed}: pre-screen/MM-correction cut a real numeric signal. support={names}; missing={ {'num_signal_1', 'num_signal_2'} - kept }"
        )

    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_region_medium_card_signal_survives(self, seed):
        """region has 50 levels x 2 y-classes = 100 cells << 1250 (the
        0.5*n budget at n=2500), so the cardinality pre-screen does NOT
        refuse it. region carries genuine signal (logit weight 1.5 on
        hot levels) and must end up in support_.
        """
        _X, _y, names = _build_and_fit_layer10(seed)
        assert "region" in names, (
            f"seed={seed}: region (medium-card signal, 100 cells) "
            f"missing from support_; cardinality pre-screen may have "
            f"become over-aggressive. support={names}"
        )


# ---------------------------------------------------------------------------
# Cardinality penalty visibility: at least one numeric signal outranks user_id
# ---------------------------------------------------------------------------


class TestHighCardCardinalityPenaltyVisible:
    """The cardinality-bias mitigation must be VISIBLE in the ranking:
    on the majority of seeds, at least one true-signal numeric column
    must rank above ``user_id`` (when user_id is kept at all). This
    pins that MRMR isn't just regurgitating raw-MI order; the screening
    + redundancy logic provides SOME defence.

    Seed 101 is excluded: that's the documented hijack case (see
    ``TestHighCardCatFEDisabledKnownHijack``) where user_id ranks #0
    and outranks every signal. Including it here would be redundant
    with the hijack pin AND would force the contract to weaken to
    something that doesn't actually catch regressions on the clean
    seeds.
    """

    @pytest.mark.parametrize("seed", [202, 303, 404, 505])
    def test_at_least_one_signal_outranks_user_id(self, seed):
        """Contract: when user_id survives at all, at least one true signal outranks it."""
        _X, _y, names = _build_and_fit_layer10(seed)
        if "user_id" not in names:
            # user_id correctly dropped - strictly better than the
            # "outranks" contract; nothing to assert.
            return
        user_idx = names.index("user_id")
        signal_ranks = [names.index(s) for s in ("num_signal_1", "num_signal_2", "region") if s in names]
        assert any(r < user_idx for r in signal_ranks), (
            f"seed={seed}: user_id (rank {user_idx}) outranks ALL true "
            f"signals; MRMR is following raw-MI order with zero "
            f"cardinality-bias mitigation. support={names}"
        )
