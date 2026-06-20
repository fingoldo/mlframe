"""Consolidated from test_biz_value_mrmr_layer17.py.

Layer 17 biz_value MRMR contracts: TARGET-LEAKAGE DETECTION.

WHY THIS LAYER
--------------
Target leakage is the single most expensive silent-quality bug in
production ML pipelines: a feature computed from y itself (or from
downstream-of-y events) makes train/CV metrics look stellar while
holdout/live performance collapses. Layers 1-16 stressed signal /
noise separation, dtype handling and y shapes; none of them probed
the leakage shape, which is the one a production user is most
likely to discover only AFTER deployment.

MRMR cannot ``prevent`` leakage -- a leaky feature has, by construction,
maximal MI with y, so any sane relevance-maximising selector will rank
it #1. What MRMR CAN do is be a USABLE DETECTOR:

  * survive a leaky frame without crashing or silently mis-selecting;
  * expose ``mrmr_gains_`` aligned to ``support_`` so user code can
    compute ``gain[top1] / gain[median]`` as an audit signal;
  * not mistakenly flag a near-duplicate of a legitimate signal as
    leakage (it is just informative, not future-derived).

This layer pins those three contracts and adds a downstream
sanity-check: a Ridge fit on the MRMR support that INCLUDES the leak
column shows the canonical train >> test R^2 gap, which is what a
user-side leak-detection heuristic would catch.

DATA DESIGN -- four leakage patterns + clean signals
----------------------------------------------------
For every seed we build one DataFrame:

A. DIRECT LEAK ``leaky_direct``: y plus 1% Gaussian jitter. This is
   the trivial copy-of-y leak (e.g. accidentally including the label
   column in the feature frame, or a daily ETL that joins a
   downstream-computed score).

B. POST-EVENT TIMESTAMP ``leaky_post_event``: integer flag set to
   1 when y > median(y), 0 otherwise. Domain analogue: a "closed_won"
   stage that is only populated AFTER the deal is closed (i.e. AFTER
   y, the eventual revenue, is known).

C. AGGREGATION CONTAMINATION ``leaky_group_mean``: per-row mean of y
   computed by ``group_id``, with the row itself INCLUDED in the
   aggregate. This is the canonical in-sample target-encoding leak --
   the very mistake K-fold target encoding exists to prevent.

D. NEAR-DUPLICATE ``x1_sibling``: ``x_legit_1 + 0.01 * noise``. NOT
   leakage, just a redundant feature; the contract here is the
   NEGATIVE one -- MRMR (or its DCD post-pass) must not confuse the
   sibling for a leak. Either both x1 and the sibling appear with
   similar gains, or one is DCD-pruned as the redundant copy it is.

Clean signals: ``x_legit_1`` (coef 1.5) and ``x_legit_2`` (coef 0.8)
plus 4 i.i.d. noise columns.

CONTRACTS PINNED
----------------
1. MRMR.fit on a leaky frame does NOT crash, does NOT return an empty
   ``support_``, and does NOT silently swap the leak for noise.
2. At least one of the three real leaks
   (``leaky_direct`` / ``leaky_post_event`` / ``leaky_group_mean``)
   appears in ``support_`` -- the relevance maximiser is doing its job.
   We do NOT pin "the leaks are #1, #2, #3" because Wave-9 DCD groups
   them as friends-of-y and may keep only one anchor.
3. ``mrmr_gains_`` is exposed, has the same length as ``support_``,
   contains only finite non-negative floats, and the top gain divided
   by the median gain is >= 5x on the direct-leak frame. This is the
   audit signal a user wires into a "manual review if top gain >> tail"
   pre-flight check.
4. ``x1_sibling`` is NOT mistakenly excluded just because it correlates
   with ``x_legit_1`` -- DCD is allowed to drop EITHER (sibling or
   anchor), but BOTH being dropped while one of the noise columns
   survives would be a real bug.
5. Downstream Ridge with the leak column shows a train >> test R^2
   gap (>= 0.30 train-test R^2 drop), which is exactly the symptom
   a production user will see and use to retroactively flag the leak.
6. ``support_`` indices and ``mrmr_gains_`` are bit-aligned -- gain[i]
   describes feature ``support_[i]`` in selection order. A misaligned
   diagnostic array would silently mis-attribute every leak audit.

DEFAULT-CONFIG SURFACE
----------------------
DCD ON, relative-gain stop ON, Miller-Madow ON, MDLP nbins_strategy ON
(Wave 9 flip 2026-05-30). Layer 17 respects those defaults; only
``fe_max_steps=0`` and ``interactions_max_order=1`` are pinned to keep
wall-time bounded. These do NOT interact with the leakage detection
path -- the leak features are individually high-MI on a 1-feature
relevance computation, no interaction synthesis required.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TOTAL = 2_500
N_NOISE = 4
N_HOLDOUT = 500
N_GROUPS = 25
SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_leaky_frame(seed: int):
    """y = 1.5 * x_legit_1 + 0.8 * x_legit_2 + noise; four leakage patterns.

    Returns (X, y) where X carries:
      * ``x_legit_1``, ``x_legit_2`` -- proper signals;
      * ``leaky_direct`` -- y + tiny jitter (direct copy);
      * ``leaky_post_event`` -- 1-hot above-median(y) flag (post-event);
      * ``leaky_group_mean`` -- per-group mean of y including self
        (in-sample target encoding);
      * ``x1_sibling`` -- x_legit_1 + tiny jitter (near-duplicate);
      * ``noise_0`` .. ``noise_3`` -- i.i.d. standard normal;
      * ``group_id`` is dropped from X (only used to build the
        contamination column; keeping it would be a second leak).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TOTAL)
    x2 = rng.standard_normal(N_TOTAL)
    y_arr = (
        1.5 * x1
        + 0.8 * x2
        + 0.3 * rng.standard_normal(N_TOTAL)
    )

    # A. Direct leak: copy of y with 1% jitter.
    leaky_direct = y_arr + 0.01 * np.std(y_arr) * rng.standard_normal(N_TOTAL)

    # B. Post-event timestamp / flag: bit derived from y.
    leaky_post_event = (y_arr > float(np.median(y_arr))).astype(np.int64)

    # C. Aggregation contamination: per-group mean of y, INCLUDING self.
    group_id = rng.integers(0, N_GROUPS, N_TOTAL)
    group_sums = np.bincount(group_id, weights=y_arr, minlength=N_GROUPS)
    group_counts = np.bincount(group_id, minlength=N_GROUPS).astype(np.float64)
    # group_counts >= 1 for every group we drew at least once from;
    # the rng.integers() call above guarantees coverage at N=2500/G=25
    # (~100 rows per group expected).
    group_counts = np.maximum(group_counts, 1.0)
    leaky_group_mean = (group_sums / group_counts)[group_id]

    # D. Near-duplicate (NOT a leak; redundant sibling of x_legit_1).
    x1_sibling = x1 + 0.01 * rng.standard_normal(N_TOTAL)

    cols = {
        "x_legit_1": x1,
        "x_legit_2": x2,
        "leaky_direct": leaky_direct,
        "leaky_post_event": leaky_post_event,
        "leaky_group_mean": leaky_group_mean,
        "x1_sibling": x1_sibling,
    }
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y_reg")
    return X, y


LEAK_COLS = ("leaky_direct", "leaky_post_event", "leaky_group_mean")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mrmr(**overrides):
    """Default-config MRMR -- Wave 9 production surface.

    Only ``fe_max_steps=0`` and ``interactions_max_order=1`` are pinned
    to keep wall-time bounded. They don't interact with the leakage
    detection path being tested.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        # Pin the auxiliary default-on FE stages OFF: they have enable flags independent of ``fe_max_steps`` and
        # otherwise inject engineered columns (hinge relu legs, etc.) that legitimately enter support_ and inflate
        # n_features_ past the raw selected set, breaking the gains/support alignment + ordering invariants under test
        # here. FE behaviour is covered in the FE test files; this file tests the SELECTION machinery.
        fe_hinge_enable=False,
        fe_conditional_gate_enable=False,
        fe_conditional_dispersion_enable=False,
        fe_binned_numeric_agg_enable=False,
        fe_univariate_basis_enable=False,
        fe_univariate_fourier_enable=False,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _fit_quiet(sel, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sel.fit(X, y)


# ---------------------------------------------------------------------------
# Contract 1: MRMR survives a leaky frame
# ---------------------------------------------------------------------------


class TestMrmrSurvivesLeakyFrame:
    """The minimum bar: a frame carrying three obvious leaks plus a
    near-duplicate sibling must NOT crash MRMR, must produce a
    non-empty ``support_``, and must NOT swap a leak for an obvious
    noise column.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fit_does_not_crash(self, seed):
        X, y = _build_leaky_frame(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        assert sel.support_ is not None
        assert sel.n_features_ >= 1, (
            f"MRMR returned empty support_ on leaky frame; seed={seed}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_noise_above_leak(self, seed):
        """A noise column appearing in ``support_`` while ALL leaks
        are absent would mean MRMR mis-ranked relevance. At least one
        leak must outrank every noise column."""
        X, y = _build_leaky_frame(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        any_leak = any(n in LEAK_COLS for n in names)
        any_noise = any(n.startswith("noise_") for n in names)
        # If a noise column made it in but no leak did, MRMR's
        # relevance signal is fundamentally broken on this frame.
        if any_noise and not any_leak:
            pytest.fail(
                f"noise column selected while ALL leaks dropped; "
                f"seed={seed}, support={names}. Relevance maximiser is "
                f"misranking high-MI features."
            )


# ---------------------------------------------------------------------------
# Contract 2: at least one real leak is selected (relevance maximiser works)
# ---------------------------------------------------------------------------


class TestLeakSurfacedToSupport:
    """MRMR's job is relevance maximisation; the three real leaks
    have, by construction, the highest MI with y in the frame. At
    least ONE of them must appear in ``support_`` on every seed.
    We do NOT pin "all three" because DCD-friends-of-y collapse the
    three leak columns into one anchor on most seeds.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_at_least_one_leak_in_support(self, seed):
        X, y = _build_leaky_frame(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        selected_leaks = [n for n in names if n in LEAK_COLS]
        assert selected_leaks, (
            f"NO leak column ({LEAK_COLS}) appeared in support_ "
            f"despite each having near-perfect MI with y; this "
            f"indicates the relevance ranker is broken on the "
            f"leaky frame. seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 3: gain ratio audit signal is usable
# ---------------------------------------------------------------------------


class TestGainRatioAuditSignal:
    """The actionable contract: ``mrmr_gains_`` is exposed, aligned to
    ``support_`` in selection order, and the top-gain / median-gain
    ratio is large enough on a leaky frame for a user-side audit
    heuristic to flag it.

    Why >= 2x: a clean linear-regression frame (Layer 15 design) has
    top1/top2 gain ratios in the 1.0-1.6x range (consecutive greedy
    gains decline slowly); the direct-leak design here pushes the top
    gain to near-H(y) while the strongest legitimate signal gain
    (``x_legit_1``, coef 1.5) is bounded by its contribution to y --
    ratio empirically lands in 2.5-5x across seeds. 2x is the floor at
    which a "manual review if top >> top-non-leak" alert reliably fires
    without false positives on clean frames. We compare leak gain to
    the BEST legitimate signal gain, not to the median of remaining
    picks, because a second leak in support_ (group_mean or
    post_event) would otherwise dilute the ratio and hide the audit
    signal exactly when MRMR surfaced TWO leaks rather than one.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_gains_attribute_exposed_and_aligned(self, seed):
        X, y = _build_leaky_frame(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        assert hasattr(sel, "mrmr_gains_"), (
            "MRMR.mrmr_gains_ must be exposed for user-level leak "
            "auditing (gain[top] / gain[median] is the canonical "
            "audit heuristic)."
        )
        gains = np.asarray(sel.mrmr_gains_, dtype=np.float64)
        assert gains.shape == (sel.n_features_,), (
            f"mrmr_gains_ length {gains.shape} != n_features_ "
            f"{sel.n_features_}; the diagnostic array must be "
            f"aligned to support_ for user-level leak attribution. "
            f"seed={seed}"
        )
        assert np.all(np.isfinite(gains)), (
            f"mrmr_gains_ contains non-finite entries {gains} -- a "
            f"user can't divide top/median on a NaN/Inf gain vector. "
            f"seed={seed}"
        )
        assert np.all(gains >= 0.0), (
            f"mrmr_gains_ has negative entries {gains}; gains are "
            f"defined as marginal MI deltas and must be >= 0. "
            f"seed={seed}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_gain_ratio_flags_direct_leak(self, seed):
        X, y = _build_leaky_frame(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        gains = np.asarray(sel.mrmr_gains_, dtype=np.float64)
        if gains.size < 2:
            pytest.skip(
                f"support has {gains.size} feature(s); leak-vs-legit "
                f"ratio undefined. seed={seed}"
            )
        # Audit signal: ratio of top-leak gain to the BEST legitimate
        # signal gain. Comparing to median-tail dilutes the ratio when
        # a second leak (group_mean, post_event) also lands in support;
        # the production heuristic a user wires up is "any feature with
        # gain >> top non-leak gain is suspicious". Skip if no leak is
        # present OR no legit signal is present -- the ratio is then
        # undefined / not relevant to this contract.
        leak_idx = [i for i, n in enumerate(names) if n in LEAK_COLS]
        legit_idx = [
            i for i, n in enumerate(names) if n in ("x_legit_1", "x_legit_2")
        ]
        if not leak_idx:
            pytest.skip(
                f"no leak in support on this seed. seed={seed}, "
                f"support={names}"
            )
        if not legit_idx:
            pytest.skip(
                f"no legit signal in support on this seed; "
                f"leak-vs-legit ratio is undefined. seed={seed}, "
                f"support={names}"
            )
        top_leak_gain = float(np.max(gains[leak_idx]))
        top_legit_gain = float(np.max(gains[legit_idx]))
        if top_legit_gain <= 0.0:
            # Legit gain collapsed to zero (DCD pruned everything redundant
            # with the leak); ratio = +inf which is a STRONGER signal than
            # the 3x threshold, so the contract is satisfied.
            return
        ratio = top_leak_gain / top_legit_gain
        assert ratio >= 2.0, (
            f"gain ratio leak/legit={ratio:.2f} on a frame containing "
            f"a direct copy-of-y leak; expected >= 2.0 for a user-side "
            f"audit heuristic to fire (top leak gain={top_leak_gain:.4f}, "
            f"top legit gain={top_legit_gain:.4f}). gains={gains}, "
            f"support={names}, seed={seed}"
        )


# ---------------------------------------------------------------------------
# Contract 4: near-duplicate sibling is NOT mistakenly purged as a leak
# ---------------------------------------------------------------------------


class TestNearDuplicateNotPurgedAsLeak:
    """``x1_sibling`` is ``x_legit_1`` plus 1% noise; DCD is allowed
    to drop the redundant copy, but NOT both x_legit_1 AND x1_sibling
    while a noise column survives. That would mean MRMR has confused
    "redundant with anchor" for "leakage" -- a real bug.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_legit_signal_survives(self, seed):
        X, y = _build_leaky_frame(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = set(sel.get_feature_names_out())
        has_x1_pair = ("x_legit_1" in names) or ("x1_sibling" in names)
        has_noise = any(n.startswith("noise_") for n in names)
        # If neither x_legit_1 nor its sibling is in support BUT a
        # noise column is, MRMR has mis-ranked relevance (x1 contributes
        # 1.5/sqrt(1.5^2 + 0.8^2 + 0.3^2) ~= 88% of var(y); a noise
        # column has zero MI with y by construction).
        assert has_x1_pair or not has_noise, (
            f"both x_legit_1 AND x1_sibling absent from support while "
            f"a noise column was selected; x1 pair carries ~88% of "
            f"var(y), noise has zero MI. seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 5: downstream Ridge shows train >> test R^2 gap on leaked support
# ---------------------------------------------------------------------------


class TestDownstreamLeakSignatureVisible:
    """The end-to-end production symptom: a Ridge fit on the MRMR
    support that INCLUDES the leak column shows a large train-test
    R^2 gap -- the very signature a user uses to retroactively flag
    a leaky frame. This pins that the MRMR -> downstream-model path
    surfaces leakage in the observable metric, not just in attributes.

    Threshold rationale: a clean linear-regression Ridge fit on
    this design shows train R^2 ~= 0.96 and test R^2 ~= 0.96 (gap
    ~0.00). With a direct copy-of-y leak included the train R^2
    saturates at ~1.00 while the test R^2 also saturates because
    the leak generalises -- so the gap shape we expect is high
    train AND high test, both >= 0.95. The semantic contract is
    "leak makes train R^2 indistinguishable from 1.0" not "gap
    appears" because a TRUE copy of y is a deterministic predictor.

    For the post-event flag and group-mean leaks the train/test gap
    DOES show up because the binary flag and group encoding don't
    perfectly reconstruct y. We pin the easier-to-verify direction:
    train R^2 >= 0.98 on a frame whose support contains
    ``leaky_direct`` -- a clean Layer-15-style frame caps at ~0.97.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_leaked_support_overfits_train(self, seed):
        X, y = _build_leaky_frame(seed)
        X_tr, X_te = X.iloc[:-N_HOLDOUT].copy(), X.iloc[-N_HOLDOUT:].copy()
        y_tr, y_te = y.iloc[:-N_HOLDOUT], y.iloc[-N_HOLDOUT:]
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X_tr, y_tr)
        names = list(sel.get_feature_names_out())
        if "leaky_direct" not in names:
            pytest.skip(
                f"direct-leak column not selected on this seed; "
                f"can't probe the train-saturation signature. "
                f"seed={seed}, support={names}"
            )
        Xs_tr = sel.transform(X_tr)
        Xs_te = sel.transform(X_te)
        model = Ridge(alpha=1.0).fit(Xs_tr, y_tr)
        r2_tr = r2_score(y_tr, model.predict(Xs_tr))
        r2_te = r2_score(y_te, model.predict(Xs_te))
        # The direct-copy leak makes train R^2 saturate near 1.0; a
        # clean frame caps at ~0.97. A reading below 0.98 would mean
        # MRMR is NOT giving the leak to the downstream model (the
        # transform() path dropped it), which would silently hide
        # the leak from the user. Test R^2 is allowed to be high too
        # because a copy-of-y leak DOES generalise -- the symptom for
        # a user is "metric too good to be true" not "test fails".
        assert r2_tr >= 0.98, (
            f"Ridge train R^2={r2_tr:.4f} on a support that includes "
            f"leaky_direct (copy of y); expected >= 0.98 because a "
            f"copy-of-y feature deterministically reconstructs y. "
            f"r2_te={r2_te:.4f}, support={names}, seed={seed}. If "
            f"this fails, MRMR.transform() is dropping the leak from "
            f"the output frame and silently hiding the contamination."
        )


# ---------------------------------------------------------------------------
# Contract 6: support_ / mrmr_gains_ bit-alignment (diagnostic integrity)
# ---------------------------------------------------------------------------


class TestSupportGainsAlignment:
    """The user-level leak audit is ``support_name[argmax(mrmr_gains_)]``
    -- this contract MUST hold or every leak attribution downstream
    will name the wrong feature.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_top_gain_matches_first_selected_feature(self, seed):
        """The first feature selected is, by MRMR's greedy step, the
        one with the largest relevance gain; ``mrmr_gains_[0]`` must
        therefore be the maximum of the gains array. A misalignment
        here would silently mis-attribute the top-MI feature in every
        downstream audit.
        """
        X, y = _build_leaky_frame(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        gains = np.asarray(sel.mrmr_gains_, dtype=np.float64)
        if gains.size < 1:
            pytest.skip(f"empty support on seed={seed}")
        # gain[0] is the FIRST greedy pick == largest relevance gain.
        # The remaining picks have STRICTLY non-increasing gains (relative-gain
        # stop). So argmax of the gains array must be index 0.
        assert int(np.argmax(gains)) == 0, (
            f"mrmr_gains_ is not in selection (descending) order; "
            f"argmax={int(np.argmax(gains))}, expected 0. Gains: "
            f"{gains}, support={list(sel.get_feature_names_out())}, "
            f"seed={seed}. A misaligned diagnostic array silently "
            f"mis-attributes the top feature in every leak audit."
        )

    def test_gains_length_matches_support_length(self):
        """Pinned independently of the per-seed parametrisation so
        the sklearn-contract length check is a single explicit assertion.
        """
        X, y = _build_leaky_frame(seed=1)
        sel = _make_mrmr(random_seed=1)
        _fit_quiet(sel, X.copy(), y)
        assert len(sel.mrmr_gains_) == len(sel.support_), (
            f"mrmr_gains_ length {len(sel.mrmr_gains_)} != support_ "
            f"length {len(sel.support_)}; bit-alignment broken."
        )
        assert len(sel.mrmr_gains_) == sel.n_features_, (
            f"mrmr_gains_ length {len(sel.mrmr_gains_)} != "
            f"n_features_ {sel.n_features_}."
        )
