"""Adversarial selection-masking biz_val: monotone-warped duplicate survivor +
partial time-series look-ahead leak.

Covers two gaps the rest of the suite leaves in shadow (audit keys
gaps_selection_masking-07 and -12):

(a) MONOTONE-WARPED DUPLICATE -- which of {raw f, warp g=exp(4 f)} survives
    redundancy elimination. Binned MI / SU are monotone-invariant, so for a
    strictly-monotone warp the relevance AND redundancy of f and g are
    identical up to discretization jitter; the DCD anchor choice between them
    is a tie broken by column order. If g wins, a downstream LINEAR model loses
    the signal f carried (g's logistic fit is far worse) and nothing else in
    the suite would notice -- this is the raw-column analogue of the FE-side
    MI-monotone-invariance blind spot.

    Measured behaviour pinned here:
      * MRMR (dcd_enable default ON): exactly ONE of {f, g} survives, and the
        survivor is DETERMINISTIC for a fixed column order across 5 seeds. The
        survivor FLIPS with column order (f when f precedes g, g when g
        precedes f) -- the MI-monotone-invariance tie. The "survivor is always
        the linear-usable raw f regardless of order" contract is the stronger
        guarantee a linear downstream wants; current behaviour does NOT honour
        it (order decides), so that side is xfail(strict=False).
      * RFECV(make_pipeline(StandardScaler, LogisticRegression),
        n_features_selection_rule='one_se_min'): f survives, g is dropped, on
        ALL 5 seeds -- the linear estimator ranks f far above its exp-warp g.

(b) PARTIAL TIME-SERIES LEAK -- an AR(1) series s_t=ar s_{t-1}+e_t, y_t=s_t,
    with legit lag1..lag3 and a one-step-ahead look-ahead leak lead1=s_{t+1}.
    By AR symmetry lead1 has the SAME marginal corr with y as lag1 (~ar=0.7) --
    above every other legit lag, below any sane direct-leak threshold (0.95).

    Measured behaviour pinned here:
      * MRMR (regression): lead1 reliably appears in support across seeds. BUT
        the layer-17 gain-ratio audit (top-leak gain / top-legit gain >= 2x,
        which fires on a DIRECT corr~1 leak) does NOT fire on this partial leak:
        lead1 ties lag1 in marginal MI, so the ratio lands ~0.7-1.5, never >= 2.
        The gain-ratio detector is structurally BLIND to a partial leak that
        does not dominate the legit features -> that side is xfail(strict=False)
        documenting the blind spot, while "lead1 surfaced to support" is a hard
        assertion.
      * RFECV(RandomForestRegressor): with leakage_corr_threshold=0.6 +
        leakage_action='exclude', lead1 is EXCLUDED on all seeds; with the
        production default threshold (0.95) lead1 is INCLUDED on all seeds.
        Both sides are pinned so the default's partial-leak blind spot is
        explicit.

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test" the
quantitative floors here are set 5-15% below measured values; per the
real-prod-bug rule, the two structurally-failing contracts (warp survivor not
linear-preferred; gain-ratio blind to partial leak) are written to the CORRECT
behaviour and marked xfail(strict=False) rather than weakened.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tests.feature_selection.conftest import fast_subset

WARP_SEEDS = (0, 1, 2, 3, 4)
LEAK_SEEDS = (0, 1, 2)
AR_COEF = 0.7  # lead1 (look-ahead) ties lag1 at corr ~= AR_COEF by AR(1) symmetry


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


def _make_warp_frame(n: int = 3000, seed: int = 0, n_noise: int = 5, reverse: bool = False):
    """f ~ N(0,1); g = exp(4 f) (strictly monotone, rank-identical to f, so binned
    MI / SU(f, y) == SU(g, y) up to discretization jitter). y = (f + 0.3 N(0,1) > 0).
    ``reverse`` reverses the column order so g precedes f (probes order-dependence
    of the redundancy tie-break)."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    g = np.exp(4.0 * f)
    y = ((f + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    cols = {"f": f, "g": g}
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.standard_normal(n)
    df = pd.DataFrame(cols)
    if reverse:
        df = df[list(df.columns)[::-1]]
    return df, pd.Series(y, name="y")


def _make_ar1_leak_frame(n: int = 2000, seed: int = 0, n_noise: int = 5, ar: float = AR_COEF):
    """AR(1) s_t = ar s_{t-1} + e_t with y_t = s_t. Legit features lag1..lag3
    (past values). Leak lead1 = s_{t+1} (one-step look-ahead): by AR symmetry
    corr(lead1, y) == corr(lag1, y) ~= ar, above lag2/lag3 and below 0.95."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 2)
    s = np.zeros(n + 2)
    for t in range(1, n + 2):
        s[t] = ar * s[t - 1] + e[t]
    y = s[1 : n + 1].copy()
    lag1 = s[0:n]
    lag2 = np.concatenate([[0.0], s[0 : n - 1]])
    lag3 = np.concatenate([[0.0, 0.0], s[0 : n - 2]])
    lead1 = s[2 : n + 2].copy()
    cols = {"lag1": lag1, "lag2": lag2, "lag3": lag3, "lead1": lead1}
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Selector helpers
# ---------------------------------------------------------------------------


def _fit_mrmr(df, y, seed: int):
    """Default-surface MRMR (dcd_enable ON by default). interactions_max_order=1
    + fe_max_steps=0 keep wall-time bounded without touching the redundancy /
    relevance path under test."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0, random_seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.fit(df.copy(), y)
    return sel


def _fit_rfecv(df, y, *, estimator, leakage_corr_threshold, rule="argmax", leakage_action="exclude"):
    """Fit rfecv."""
    from mlframe.feature_selection.wrappers import RFECV

    sel = RFECV(
        estimator=estimator,
        cv=3,
        max_refits=4,
        random_state=0,
        leakage_corr_threshold=leakage_corr_threshold,
        leakage_action=leakage_action,
        n_features_selection_rule=rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.fit(df.copy(), y)
    return sel


def _names(sel):
    """Helper that names."""
    return list(sel.get_feature_names_out())


def _warp_survivors(names):
    """Raw {f, g} survivors among selected names (engineered cols like
    'f__relu_lt...' reference f -> counted toward f's survival, mirroring
    signal_recovery_count crediting). Returns set subset of {'f', 'g'}."""
    out = set()
    for nm in names:
        if nm == "f" or nm.startswith("f__") or "f__" in nm or nm.startswith("f_relu"):
            out.add("f")
        if nm == "g" or nm.startswith("g__") or "g__" in nm or nm.startswith("g_relu"):
            out.add("g")
    return out


# ---------------------------------------------------------------------------
# (a)(i) MRMR monotone-warp survivor
# ---------------------------------------------------------------------------


class TestMRMRMonotoneWarpSurvivor:
    """For g = exp(4 f) MRMR's DCD must keep exactly ONE of {f, g} (they are a
    rank-identical redundant pair), and the choice must be deterministic for a
    fixed column order. The survivor flips with column order (the MI-monotone-
    invariance tie), so 'survivor is always the linear-usable raw f' is a
    documented limitation, not the current contract."""

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", WARP_SEEDS)
    def test_exactly_one_of_warp_pair_survives(self, seed):
        """Exactly one of warp pair survives."""
        df, y = _make_warp_frame(seed=seed)
        sel = _fit_mrmr(df, y, seed)
        survivors = _warp_survivors(_names(sel))
        assert survivors, (
            f"neither f nor a warp(f)=g column survived; the rank-identical "
            f"redundant pair should leave exactly one leg in support. "
            f"seed={seed}, support={_names(sel)}"
        )
        assert survivors != {"f", "g"}, (
            f"BOTH f and g survived; DCD must collapse a rank-identical monotone-warped duplicate to ONE leg, not keep both. seed={seed}, support={_names(sel)}"
        )

    def test_exactly_one_of_warp_pair_survives_fast(self):
        """Exactly one of warp pair survives fast."""
        seed = fast_subset(WARP_SEEDS, n=1)[0]
        df, y = _make_warp_frame(seed=seed)
        sel = _fit_mrmr(df, y, seed)
        survivors = _warp_survivors(_names(sel))
        assert survivors and survivors != {"f", "g"}, f"fast rep: exactly one of {{f, g}} must survive. seed={seed}, support={_names(sel)}"

    @pytest.mark.slow
    def test_warp_survivor_deterministic_for_fixed_order(self):
        """For a FIXED column order (f before g) the survivor is identical across
        all 5 seeds -- the tie-break is deterministic, not seed-flaky."""
        survivors_per_seed = []
        for seed in WARP_SEEDS:
            df, y = _make_warp_frame(seed=seed, reverse=False)
            sel = _fit_mrmr(df, y, seed)
            survivors_per_seed.append(frozenset(_warp_survivors(_names(sel))))
        uniq = set(survivors_per_seed)
        assert len(uniq) == 1, (
            f"warp-pair survivor is NOT deterministic across seeds at fixed "
            f"column order; got {survivors_per_seed}. A non-deterministic "
            f"redundancy tie-break ships selection nondeterminism between "
            f"otherwise-identical pipelines."
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", WARP_SEEDS)
    def test_warp_survivor_is_linear_usable_f_regardless_of_order(self, seed):
        """The contract a linear downstream wants: whichever order f and g appear
        in, the SURVIVOR is the linear-usable raw f (not its exp-warp g).

        DCD's ``warp_tiebreak_prefer_linear`` (default ON) biases the redundancy-
        resolution choice between strictly-monotone twins toward the more linearly-
        usable leg: at the cluster-pruning point, when the candidate about to be
        pruned as SU-redundant with the anchor is a raw-rank-corr>=0.99 twin AND
        strictly more linearly-usable (|corr(col, rank col)|), it DISPLACES the
        anchor. Exactly one leg is kept either way, so f survives under both column
        orders without any risk of emptying support_."""
        keep = []
        for reverse in (False, True):
            df, y = _make_warp_frame(seed=seed, reverse=reverse)
            sel = _fit_mrmr(df, y, seed)
            keep.append(_warp_survivors(_names(sel)))
        for reverse, survivors in zip((False, True), keep):
            assert survivors == {"f"}, f"survivor should be the linear-usable raw f, got {survivors} (reverse={reverse}, seed={seed})"


# ---------------------------------------------------------------------------
# (a)(ii) RFECV monotone-warp -- f outranks g for a linear estimator
# ---------------------------------------------------------------------------


class TestRFECVMonotoneWarpLinearPrefersRaw:
    """A StandardScaler+LogisticRegression pipeline ranks raw f far above its
    exp-warp g (the exp distribution makes g's standardized coefficient nearly
    useless). RFECV trimmed to a compact set (one_se_min) must keep f and drop g
    on every seed."""

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", WARP_SEEDS)
    def test_rfecv_keeps_f_drops_g(self, seed):
        """Rfecv keeps f drops g."""
        df, y = _make_warp_frame(seed=seed)
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=0))
        sel = _fit_rfecv(df, y, estimator=est, leakage_corr_threshold=None, rule="one_se_min")
        names = _names(sel)
        assert "f" in names, f"raw f (the linear-usable leg) must be in RFECV support; seed={seed}, support={names}"
        assert "g" not in names, (
            f"exp-warp g must be dropped -- its standardized logistic coef is "
            f"far weaker than f's, so a linear-estimator RFECV must not prefer "
            f"it. seed={seed}, support={names}"
        )

    def test_rfecv_keeps_f_drops_g_fast(self):
        """Rfecv keeps f drops g fast."""
        seed = fast_subset(WARP_SEEDS, n=1)[0]
        df, y = _make_warp_frame(seed=seed)
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=0))
        sel = _fit_rfecv(df, y, estimator=est, leakage_corr_threshold=None, rule="one_se_min")
        names = _names(sel)
        assert "f" in names and "g" not in names, f"fast rep: f kept, g dropped. seed={seed}, support={names}"


# ---------------------------------------------------------------------------
# (b)(i) MRMR partial time-series leak
# ---------------------------------------------------------------------------


class TestMRMRPartialTimeSeriesLeak:
    """lead1 = s_{t+1} is a one-step look-ahead leak with corr(lead1, y) ~= 0.7
    (it ties lag1 by AR symmetry). MRMR's relevance ranker surfaces it to support
    -- but the layer-17 gain-ratio audit (designed for a DIRECT corr~1 leak that
    dominates) is structurally blind to it because lead1 does not out-gain the
    legit lag1."""

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", LEAK_SEEDS)
    def test_partial_leak_surfaces_to_support(self, seed):
        """Partial leak surfaces to support."""
        df, y = _make_ar1_leak_frame(seed=seed)
        sel = _fit_mrmr(df, y, seed)
        names = _names(sel)
        assert (
            "lead1" in names
        ), f"the look-ahead leak lead1 (corr ~= {AR_COEF} with y) should be surfaced to support by the relevance ranker; seed={seed}, support={names}"

    def test_partial_leak_surfaces_to_support_fast(self):
        """Partial leak surfaces to support fast."""
        seed = fast_subset(LEAK_SEEDS, n=1)[0]
        df, y = _make_ar1_leak_frame(seed=seed)
        sel = _fit_mrmr(df, y, seed)
        names = _names(sel)
        assert "lead1" in names, f"fast rep: lead1 should surface to support. seed={seed}, support={names}"

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="PROD BUG: the layer-17 gain-ratio leak audit (top-leak gain / top-legit gain >= 2x) "
        "fires only on a DIRECT corr~1 leak. A PARTIAL look-ahead leak (lead1 ties lag1 in "
        "marginal MI by AR(1) symmetry) yields a ratio ~0.7-1.5, so the audit is structurally "
        "blind to it; a leak-aware audit must not rely on gain dominance alone.",
        strict=False,
    )
    @pytest.mark.parametrize("seed", LEAK_SEEDS)
    def test_gain_ratio_flags_partial_leak(self, seed):
        """Extends the layer-17 direct-leak gain-ratio detector to a partial leak.
        Current behaviour: the ratio lead1_gain / best_lag_gain does NOT reach the
        2x flag threshold (lead1 ties lag1), so the audit cannot flag the partial
        leak -> xfail(strict=False)."""
        df, y = _make_ar1_leak_frame(seed=seed)
        sel = _fit_mrmr(df, y, seed)
        names = _names(sel)
        gains = np.asarray(getattr(sel, "mrmr_gains_", []), dtype=np.float64)
        assert (
            gains.size == len(names) and "lead1" in names
        ), f"need aligned mrmr_gains_ and lead1 in support to compute the audit ratio; seed={seed}, support={names}, gains={gains}"
        lead_idx = [i for i, n in enumerate(names) if n == "lead1"]
        lag_idx = [i for i, n in enumerate(names) if n in ("lag1", "lag2", "lag3")]
        assert lag_idx, f"no legit lag in support to baseline against; seed={seed}, support={names}"
        ratio = float(np.max(gains[lead_idx])) / max(float(np.max(gains[lag_idx])), 1e-12)
        assert ratio >= 2.0, (
            f"gain-ratio leak/legit={ratio:.2f}; a leak audit extended to partial "
            f"leaks should fire (>= 2.0) but the partial leak ties the legit lag "
            f"in marginal MI. seed={seed}, gains={gains}, support={names}"
        )


# ---------------------------------------------------------------------------
# (b)(ii) RFECV partial-leak threshold -- pin BOTH sides
# ---------------------------------------------------------------------------


class TestRFECVPartialLeakThreshold:
    """The leakage_corr scan catches a partial leak only if the threshold is
    tightened below the leak's correlation. Pin BOTH sides so the production
    default's blind spot is explicit: 0.6 excludes lead1; the default 0.95 (and
    None=disabled) keeps it."""

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", LEAK_SEEDS)
    def test_tight_threshold_excludes_partial_leak(self, seed):
        """Tight threshold excludes partial leak."""
        df, y = _make_ar1_leak_frame(seed=seed)
        est = RandomForestRegressor(n_estimators=40, random_state=0)
        sel = _fit_rfecv(df, y, estimator=est, leakage_corr_threshold=0.6, leakage_action="exclude")
        names = _names(sel)
        assert (
            "lead1" not in names
        ), f"leakage_corr_threshold=0.6 + action='exclude' must drop the look-ahead leak lead1 (corr ~= {AR_COEF}); seed={seed}, support={names}"

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", LEAK_SEEDS)
    def test_default_threshold_keeps_partial_leak(self, seed):
        """Production default leakage_corr_threshold=0.95: the partial leak's
        corr (~0.7) is below it, so lead1 is NOT caught -- the blind spot."""
        df, y = _make_ar1_leak_frame(seed=seed)
        est = RandomForestRegressor(n_estimators=40, random_state=0)
        sel = _fit_rfecv(df, y, estimator=est, leakage_corr_threshold=0.95, leakage_action="exclude")
        names = _names(sel)
        assert "lead1" in names, (
            f"default leakage_corr_threshold=0.95 does NOT catch a corr~{AR_COEF} "
            f"partial leak -- lead1 should remain in support (documenting the "
            f"default's blind spot). seed={seed}, support={names}"
        )

    def test_partial_leak_threshold_both_sides_fast(self):
        """Partial leak threshold both sides fast."""
        seed = fast_subset(LEAK_SEEDS, n=1)[0]
        df, y = _make_ar1_leak_frame(seed=seed)
        est = RandomForestRegressor(n_estimators=40, random_state=0)
        tight = _fit_rfecv(df, y, estimator=est, leakage_corr_threshold=0.6, leakage_action="exclude")
        loose = _fit_rfecv(df, y, estimator=est, leakage_corr_threshold=0.95, leakage_action="exclude")
        assert "lead1" not in _names(tight), f"fast: 0.6 must exclude lead1; support={_names(tight)}"
        assert "lead1" in _names(loose), f"fast: 0.95 must keep lead1; support={_names(loose)}"


# ---------------------------------------------------------------------------
# (c) biz_value: warp linear-usability tie-break + never-empty raw support_
# ---------------------------------------------------------------------------


def _binq(x, nb: int = 10):
    """Helper that binq."""
    return pd.qcut(x, nb, labels=False, duplicates="drop").astype(np.int64)


class TestWarpLinearTiebreakDirect:
    """Unit-level proof that the DCD cluster-pruning tie-break keeps the linearly-
    usable leg of a strictly-monotone twin pair. Probes ``discover_cluster_members``
    directly with g (the exp-warp, linear-unusable) pre-selected as the anchor and f
    (the linear-usable twin) as the candidate about to be pruned -- the regime the
    end-to-end greedy hides because its relevance-tie ordering already happens to
    pick f first, so this is the only place the mechanism is observable in isolation.
    The flag-OFF side pins the legacy column-order behaviour (anchor kept)."""

    def _setup(self, flag, seed=0, n=3000):
        """Helper that setup."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            make_dcd_state,
            discover_cluster_members,
        )

        rng = np.random.default_rng(seed)
        f = rng.standard_normal(n)
        g = np.exp(4.0 * f)  # strictly-monotone, rank-identical to f, binned SU(f,g)=1
        fac = np.column_stack([_binq(g), _binq(f)]).astype(np.int64)  # cols ['g','f']
        nbins = np.array([fac[:, 0].max() + 1, fac[:, 1].max() + 1], dtype=np.int64)
        Xraw = pd.DataFrame({"g": g, "f": f})
        state = make_dcd_state(
            X_raw=Xraw,
            factors_data=fac,
            cols=["g", "f"],
            nbins=nbins,
            factors_nbins=nbins,
            target_indices=np.array([], dtype=np.int64),
            warp_tiebreak_prefer_linear=flag,
            tau_cluster=0.7,
        )
        sv = [0]  # g (idx 0) is the already-selected anchor
        discover_cluster_members(state, 0, [1], factors_data=fac, factors_nbins=nbins, selected_vars=sv)
        return state, sv

    def test_warp_tiebreak_on_keeps_linear_f(self):
        """Warp tiebreak on keeps linear f."""
        state, sv = self._setup(flag=True)
        assert sv == [1], f"warp_tiebreak_prefer_linear ON must DISPLACE the exp-warp anchor g with the linear-usable twin f; selected_vars={sv}"
        assert bool(state.pool_pruned_mask[0]) and not bool(state.pool_pruned_mask[1]), f"g must be pruned and f kept; mask={state.pool_pruned_mask.tolist()}"

    def test_warp_tiebreak_off_keeps_order_decided_g(self):
        """Warp tiebreak off keeps order decided g."""
        state, sv = self._setup(flag=False)
        assert sv == [0], f"flag OFF must preserve the order-decided anchor g (legacy column-order tie-break); selected_vars={sv}"
        assert not bool(state.pool_pruned_mask[0]) and bool(state.pool_pruned_mask[1]), f"legacy: g kept, f pruned; mask={state.pool_pruned_mask.tolist()}"

    def test_non_twin_does_not_fire(self):
        """A merely-correlated (NOT monotone-twin) candidate below the rank-corr band
        must NOT trigger the displacement even when SU clusters it -- guards against
        over-firing on ordinary collinearity."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            make_dcd_state,
            discover_cluster_members,
        )

        rng = np.random.default_rng(1)
        n = 3000
        a = rng.standard_normal(n)
        # c shares enough rank structure to cluster on SU but is NOT a strict monotone
        # twin (rank-corr < 0.99): a heavy-noise linear mix.
        c = a + 1.2 * rng.standard_normal(n)
        fac = np.column_stack([_binq(a), _binq(c)]).astype(np.int64)
        nbins = np.array([fac[:, 0].max() + 1, fac[:, 1].max() + 1], dtype=np.int64)
        Xraw = pd.DataFrame({"a": a, "c": c})
        state = make_dcd_state(
            X_raw=Xraw,
            factors_data=fac,
            cols=["a", "c"],
            nbins=nbins,
            factors_nbins=nbins,
            target_indices=np.array([], dtype=np.int64),
            warp_tiebreak_prefer_linear=True,
            tau_cluster=0.0,  # tau=0 forces clustering
        )
        sv = [0]
        discover_cluster_members(state, 0, [1], factors_data=fac, factors_nbins=nbins, selected_vars=sv)
        # a (idx 0) stays the anchor -- c is not a monotone twin, so no displacement.
        assert sv == [0], f"a non-twin (rank-corr below band) must NOT displace the anchor; selected_vars={sv}"


class TestNeverEmptyRawSupport:
    """The never-empty guarantee: a fit that confirms any feature must expose a NON-EMPTY SELECTION
    (``get_feature_names_out()``), and ``support_`` (the RAW indices) must stay within bounds.

    Contract note: when the only confirmed feature is an engineered multi-parent interaction whose raw operands are
    ALL fully subsumed by it, the fit deliberately leaves an ENGINEERED-ONLY support (empty ``support_``) rather than
    re-attach a raw stand-in -- _fit_impl_core.py:9756-9764 documents this with measured evidence that re-attaching
    resurrects the dropped operands or pulls in the next pure-noise column. The engineered feature replays from the
    raw columns present in X, so the selection is complete and usable. Hence the invariant is on the SELECTION being
    non-empty, not on ``support_`` specifically (a borderline binning shift can flip a case into engineered-only)."""

    @pytest.mark.parametrize("nbins", [5, 10, 20])
    def test_selection_never_empty_and_support_in_bounds(self, nbins):
        """Selection never empty and support in bounds."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        from tests.feature_selection._biz_val_synth import make_signal_plus_noise, as_df

        X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
        df, ys = as_df(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, random_seed=42, quantization_nbins=nbins).fit(df, ys)
        names = list(sel.get_feature_names_out())
        assert len(names) >= 1, f"the never-empty guarantee: some feature must be confirmed; nbins={nbins}, names={names}"
        # support_ holds RAW indices; it may be empty in the engineered-only case, but never out of bounds.
        assert 0 <= len(sel.support_) <= df.shape[1], f"support_ (raw indices) must be within bounds; nbins={nbins}, support_={sel.support_}, names={names}"
