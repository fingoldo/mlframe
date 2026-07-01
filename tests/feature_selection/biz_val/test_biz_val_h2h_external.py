"""Head-to-head business-value proofs for the flagship MRMR selector against the
EXTERNAL baselines a practitioner reaches for first -- ``SelectKBest(mutual_info)``,
sklearn ``RFE``, random-K, all-features, and (when pip-installed) ``mrmr_classif``.

Pins two findings from the fs-tests audit (2026-06-10):

* bizvalue_value_proofs-01 -- there was no CI assertion anywhere that any mlframe
  selector beats, or even ties at lower cost, an external baseline a user reaches
  for first. Implemented honestly:
    - a HARD, structural property on the canonical mRMR terrain (a dominant
      correlated cluster): ``SelectKBest(mutual_info)`` is provably redundancy-
      trapped -- its whole top-K budget goes to near-duplicate copies of one latent
      factor -- and MRMR ties-or-beats it at a MATCHED feature budget;
    - a documented within-epsilon tie where MRMR has no structural edge.
* bizvalue_value_proofs-03 -- the quality-vs-K frontier. The MI-descending ranking's
  downstream quality PLATEAUS while it spends K on redundant copies of the loudest
  cluster, then only improves once K is large enough to escape the cluster. The
  ASCII frontier table is the showcase artifact.

Downstream metric: 5-fold LogisticRegression roc_auc. Every comparative claim is
asserted on a MAJORITY of seeds (selectors are high-variance -- a single-seed win
does not count, per the project rule).

Calibration (dev runs, ``audit/fs_tests_audit_2026_06_10/_calib*.py``, seeds 0-5):
  - SelectKBest(MI, k=4) top-4 was 4/4 from the dominant cluster on 6/6 seeds.
  - MI downstream AUC plateaued ~0.805 for K=1..6 then jumped +0.057..+0.106 at K=8
    on 6/6 seeds.
  - MRMR full-support AUC at its own (matched) K met-or-beat SelectKBest on 6/6 seeds
    (often by +0.04..+0.13); margins below are set well under the measured median.
  - MRMR top-4 beat the random-K floor by >= 0.03 on 5/6 seeds.

The MRMR config is the deliberately-patient diversifying one: DCD OFF (DCD collapses
the dominant cluster to nothing on this fixture), no relevance-gain floor, generous
``max_consec_unconfirmed`` so the greedy search explores past the dominant cluster's
redundant copies into the weaker clusters + independent signals; FE off
(``fe_max_steps=0``) so ``support_`` is a clean raw-column ranking.
``run_additional_rfecv_minutes=False`` keeps it a pure filter.

PROD BUG surfaced (see ``test_xfail_...`` below): on ``make_signal_plus_noise`` MRMR
returns an EMPTY ``support_`` (0-width ``transform``) on a majority of seeds despite
``min_features_fallback=1``, whose docstring guarantees ``support_`` is never empty.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tests.feature_selection._biz_val_synth import as_df, make_signal_plus_noise
from tests.feature_selection.conftest import fast_subset, is_fast_mode


# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------



pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)

def _mrmr_kwargs(seed: int, fe: bool = False) -> dict:
    kw = dict(
        random_seed=seed,
        verbose=0,
        cv=3,
        run_additional_rfecv_minutes=False,
        use_simple_mode=False,
        dcd_enable=False,
        min_relevance_gain=0.0,
        min_relevance_gain_relative_to_first=0.0,
        max_consec_unconfirmed=60,
        min_features_fallback=4,
        full_npermutations=3 if is_fast_mode() else 5,
    )
    if not fe:
        kw["fe_max_steps"] = 0
    return kw


SEEDS = (0, 1, 2, 3, 4, 5)
K_GRID = (1, 2, 3, 4, 5, 6, 8)


def _seeds():
    return fast_subset(SEEDS, n=1)


def _k_grid():
    return [2, 4, 8] if is_fast_mode() else list(K_GRID)


def _n():
    return 1200 if is_fast_mode() else 2000


def _majority(n_seeds: int) -> int:
    return (n_seeds // 2) + 1


def _auc_on_cols(X: pd.DataFrame, y: pd.Series, cols, cv: int = 5) -> float:
    """5-fold LogReg roc_auc on a chosen RAW column subset. ``nan`` if empty."""
    cols = list(cols)
    if not cols:
        return float("nan")
    return float(
        cross_val_score(
            LogisticRegression(max_iter=400), X[cols], y, cv=cv, scoring="roc_auc"
        ).mean()
    )


def _fit_mrmr(X: pd.DataFrame, y: pd.Series, seed: int, fe: bool = False):
    from mlframe.feature_selection.filters.mrmr import MRMR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(**_mrmr_kwargs(seed, fe=fe)).fit(X, y)


def _mrmr_selection_order(X: pd.DataFrame, y: pd.Series, seed: int) -> list[str]:
    """MRMR support as RAW column names IN SELECTION ORDER.

    ``MRMR.support_`` is the greedy selection-order integer index array
    (``_mrmr_fit_impl`` appends each pick), so a length-K prefix is exactly
    "MRMR's first K choices".
    """
    sel = _fit_mrmr(X, y, seed, fe=False)
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)]


def _mi_descending_order(X: pd.DataFrame, y: pd.Series, seed: int) -> list[str]:
    mi = mutual_info_classif(X.values, y.values, random_state=seed)
    return [X.columns[i] for i in np.argsort(mi)[::-1]]


def _skb_cols(X: pd.DataFrame, y: pd.Series, k: int, seed: int) -> list[str]:
    """SelectKBest(mutual_info_classif) top-k as column names (= MI-descending top-k)."""
    return _mi_descending_order(X, y, seed)[:k]


def _rfe_cols(X: pd.DataFrame, y: pd.Series, k: int, seed: int) -> list[str]:
    rfe = RFE(
        LogisticRegression(max_iter=400, random_state=seed), n_features_to_select=k
    ).fit(X.values, y.values)
    return [X.columns[i] for i in np.flatnonzero(rfe.get_support())]


def _random_k_mean_auc(X: pd.DataFrame, y: pd.Series, k: int, seed: int,
                        draws: int = 5) -> float:
    rng = np.random.default_rng(7000 + seed)
    aucs = [
        _auc_on_cols(X, y, list(rng.choice(X.columns, size=k, replace=False)))
        for _ in range(draws)
    ]
    return float(np.mean(aucs))


def _clusters_in(cols) -> set:
    return {c.split("_")[0] for c in cols if c.startswith("clu")}


# ---------------------------------------------------------------------------
# Fixture: dominant correlated cluster (the canonical mRMR / redundancy terrain)
# ---------------------------------------------------------------------------


def make_dominant_cluster_fixture(n: int = 2000, seed: int = 0):
    """One DOMINANT correlated cluster (6 very-tight copies, largest y-weight) +
    two weaker clusters (3 copies each) + 2 independent signals + 20 pure noise.

    ``make_correlated_redundant`` builds only a SINGLE correlated cluster, so this
    multi-cluster fixture is constructed inline (mirrors the TestClusterRecall
    pattern in test_biz_value_mrmr_quality_metrics.py). The point: a marginal-MI
    top-K selector ranks all 6 dominant-cluster copies above everything else and
    spends its whole budget on near-duplicates of one latent factor, leaving the
    weaker clusters + the independent signals (complementary predictive info)
    unrepresented -- capping its downstream AUC. MRMR's redundancy gate rejects the
    copies and diversifies.
    """
    rng = np.random.default_rng(seed)
    latents = [rng.standard_normal(n) for _ in range(3)]
    copies = (6, 3, 3)
    tightness = (0.05, 0.25, 0.25)
    cols: dict[str, np.ndarray] = {}
    for c, lat in enumerate(latents):
        for k in range(copies[c]):
            cols[f"clu{c}_m{k}"] = lat + tightness[c] * rng.standard_normal(n)
    indep0 = rng.standard_normal(n)
    indep1 = rng.standard_normal(n)
    cols["indep0"] = indep0
    cols["indep1"] = indep1
    for j in range(20):
        cols[f"noise{j}"] = rng.standard_normal(n)
    score = (
        2.2 * latents[0]
        + 1.3 * latents[1]
        + 1.3 * latents[2]
        + 1.1 * indep0
        + 1.1 * indep1
        + 0.3 * rng.standard_normal(n)
    )
    y = (score > np.median(score)).astype(np.int64)
    return pd.DataFrame(cols), pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# bizvalue_value_proofs-01 : the baseline (SelectKBest-MI) is redundancy-trapped
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_biz_val_selectkbest_mi_is_redundancy_trapped():
    """The external baseline's structural weakness, asserted directly: on the
    dominant-cluster fixture, ``SelectKBest(mutual_info, k=4)`` spends >= 3 of its 4
    picks on copies of the SAME dominant cluster (clu0), leaving the other two
    clusters + both independent signals out of budget. Measured: 4/4 from clu0 on
    6/6 dev seeds. This is exactly the redundancy that motivates mRMR.
    """
    n = _n()
    trapped = 0
    rows = []
    for seed in _seeds():
        X, y = make_dominant_cluster_fixture(n=n, seed=seed)
        skb = _skb_cols(X, y, 4, seed)
        n_clu0 = sum(1 for c in skb if c.startswith("clu0_"))
        rows.append((seed, skb, n_clu0))
        if n_clu0 >= 3:
            trapped += 1

    print("\nSelectKBest(MI) top-4 redundancy trap:")
    for seed, skb, n_clu0 in rows:
        print(f"  seed={seed} top4={skb} dominant-cluster picks={n_clu0}/4")

    need = _majority(len(_seeds()))
    assert trapped >= need, (
        f"SelectKBest(MI) top-4 should be dominated (>=3/4) by the dominant cluster "
        f"on a majority of seeds; got {trapped}/{len(_seeds())} (need {need}). rows={rows}"
    )


@pytest.mark.slow
def test_biz_val_mrmr_ties_or_beats_selectkbest_at_matched_k():
    """At a MATCHED feature budget (SelectKBest given exactly MRMR's support size),
    MRMR's diversified raw selection ties-or-beats ``SelectKBest(mutual_info)`` on a
    majority of seeds -- it does not waste its budget on the dominant cluster's
    copies. Measured: MRMR met-or-beat SelectKBest on 6/6 dev seeds (deltas up to
    +0.13); floor set at -0.03 (a documented within-epsilon win/tie, never a
    fabricated margin).
    """
    n = _n()
    margin = 0.03
    wins = 0
    valid = 0
    rows = []
    for seed in _seeds():
        X, y = make_dominant_cluster_fixture(n=n, seed=seed)
        order = _mrmr_selection_order(X, y, seed)
        km = len(order)
        if km == 0:
            rows.append((seed, 0, float("nan"), float("nan")))
            continue
        a_mrmr = _auc_on_cols(X, y, order)
        a_skb = _auc_on_cols(X, y, _skb_cols(X, y, km, seed))
        rows.append((seed, km, a_mrmr, a_skb))
        valid += 1
        if not np.isnan(a_mrmr) and a_mrmr >= a_skb - margin:
            wins += 1

    print("\nMRMR vs SelectKBest at matched K:")
    print("  seed   K   auc_mrmr   auc_skb    delta")
    for seed, km, a_m, a_s in rows:
        d = a_m - a_s if not np.isnan(a_m) else float("nan")
        print(f"  {seed:>4}  {km:>2}   {a_m:>8.4f}   {a_s:>7.4f}   {d:>+7.4f}")

    assert valid >= 1, f"MRMR produced no usable selection on any seed; rows={rows}"
    need = _majority(valid)
    assert wins >= need, (
        f"MRMR should tie-or-beat SelectKBest within {margin} at matched K on a "
        f"majority of seeds; got {wins}/{valid} (need {need}). rows={rows}"
    )


@pytest.mark.slow
def test_biz_val_mrmr_top4_beats_random_and_undercuts_allfeatures():
    """Sanity-floor roster at K=4 on the dominant-cluster fixture: MRMR's top-4 must
    (i) clear the random-K floor (mean of 5 random 4-subsets) by >= 0.03 on a
    majority of seeds, and (ii) stay AT-OR-BELOW the all-features oracle (a 4-feature
    subset cannot beat the full ~0.99 model -- a guard that the AUC helper isn't
    leaking). RFE(LogReg) is recorded for the showcase table only: it is a wrapper,
    not a marginal-MI filter, so it is NOT cluster-trapped and we make no claim
    against it (honest: RFE wins this particular terrain).

    Measured: random-K mean AUC ~0.65-0.81, MRMR top-4 ~0.80-0.96, all-features
    ~0.994, RFE ~0.95.
    """
    n = _n()
    rand_margin = 0.03
    beats_rand = 0
    valid = 0
    rows = []
    for seed in _seeds():
        X, y = make_dominant_cluster_fixture(n=n, seed=seed)
        order = _mrmr_selection_order(X, y, seed)
        a_mrmr4 = _auc_on_cols(X, y, order[:4])
        a_rand = _random_k_mean_auc(X, y, 4, seed)
        a_all = _auc_on_cols(X, y, list(X.columns))
        a_rfe = _auc_on_cols(X, y, _rfe_cols(X, y, 4, seed))
        rows.append((seed, a_mrmr4, a_rand, a_rfe, a_all))
        if np.isnan(a_mrmr4):
            continue
        valid += 1
        assert a_mrmr4 <= a_all + 1e-9, (
            f"seed={seed}: 4-feature MRMR AUC {a_mrmr4:.4f} exceeded the all-features "
            f"oracle {a_all:.4f} -- AUC helper leak?"
        )
        if a_mrmr4 >= a_rand + rand_margin:
            beats_rand += 1

    print("\nK=4 roster (dominant-cluster fixture):")
    print("  seed   auc_mrmr4   auc_rand   auc_rfe   auc_all")
    for seed, a_m, a_r, a_f, a_a in rows:
        print(f"  {seed:>4}   {a_m:>9.4f}   {a_r:>8.4f}   {a_f:>7.4f}   {a_a:>7.4f}")

    assert valid >= 1, f"MRMR produced no usable top-4 on any seed; rows={rows}"
    need = _majority(valid)
    assert beats_rand >= need, (
        f"MRMR top-4 should beat the random-K floor by >= {rand_margin} on a "
        f"majority of seeds; got {beats_rand}/{valid} (need {need}). rows={rows}"
    )


# ---------------------------------------------------------------------------
# bizvalue_value_proofs-03 : quality-vs-K frontier (K-efficiency of mRMR)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_biz_val_quality_vs_k_frontier():
    """The quality-vs-K frontier. For each K in the grid, downstream 5-fold LogReg
    AUC of the ``mutual_info_classif``-descending top-K prefix on the
    dominant-cluster fixture. The MI frontier PLATEAUS for small K (every added
    feature is a copy of the dominant cluster, no new signal) and only JUMPS once K
    is large enough to escape the cluster into another latent factor.

    Pinned on a MAJORITY of seeds:
      (i)  plateau: ``auc_mi[6] - auc_mi[1] <= 0.02`` (no real gain inside the trap);
      (ii) escape jump: ``auc_mi[8] - auc_mi[6] >= 0.03`` (the gain only arrives once
           the budget finally reaches a second factor) -- the K-budget inefficiency
           of a redundancy-blind filter, which is precisely the mRMR pitch.

    Also asserts MRMR's K=4 prefix reaches the MI frontier's eventual K=8 level
    (within 0.05) on a majority of seeds -- half the budget, same quality -- where
    MRMR produced >= 4 features. Prints the full ASCII frontier table (showcase).

    Measured: MI ~0.805 flat for K=1..6 then ~0.866-0.92 at K=8 on 6/6 seeds.
    """
    n = _n()
    ks = _k_grid()
    k_lo = 4 if 4 in ks else ks[len(ks) // 2]
    k_hi = max(ks)
    k_mid = 6 if 6 in ks else ks[-2]
    k_first = ks[0]

    plateau_margin = 0.02
    jump_margin = 0.03
    eff_margin = 0.05

    plateau_jump = 0
    eff = 0
    eff_valid = 0
    table = []
    for seed in _seeds():
        X, y = make_dominant_cluster_fixture(n=n, seed=seed)
        m_order = _mrmr_selection_order(X, y, seed)
        i_order = _mi_descending_order(X, y, seed)
        auc_m = {k: _auc_on_cols(X, y, m_order[:k]) for k in ks}
        auc_i = {k: _auc_on_cols(X, y, i_order[:k]) for k in ks}
        table.append((seed, auc_m, auc_i))

        plateau = auc_i[k_mid] - auc_i[k_first]
        jump = auc_i[k_hi] - auc_i[k_mid]
        if plateau <= plateau_margin and jump >= jump_margin:
            plateau_jump += 1

        a_m_lo = auc_m[k_lo]
        a_i_hi = auc_i[k_hi]
        if not np.isnan(a_m_lo):
            eff_valid += 1
            if a_m_lo >= a_i_hi - eff_margin:
                eff += 1

    # ASCII showcase table.
    print("\nquality-vs-K frontier (dominant-cluster fixture), downstream LogReg AUC:")
    print("  seed  method " + "".join(f"  K={k:<3}" for k in ks))
    for seed, auc_m, auc_i in table:
        m_row = "  {:>4}  mrmr   ".format(seed) + "".join(
            f"  {auc_m[k]:>5.3f}" if not np.isnan(auc_m[k]) else "  ----." for k in ks
        )
        i_row = "  {:>4}  mi     ".format(seed) + "".join(
            f"  {auc_i[k]:>5.3f}" if not np.isnan(auc_i[k]) else "  ----." for k in ks
        )
        print(m_row)
        print(i_row)

    need = _majority(len(_seeds()))
    assert plateau_jump >= need, (
        f"MI frontier should plateau (auc_mi[{k_mid}]-auc_mi[{k_first}] <= "
        f"{plateau_margin}) then jump (auc_mi[{k_hi}]-auc_mi[{k_mid}] >= {jump_margin}) "
        f"on a majority of seeds; got {plateau_jump}/{len(_seeds())} (need {need}). "
        f"table={table}"
    )
    assert eff_valid >= 1, f"MRMR produced no K={k_lo} prefix on any seed; table={table}"
    need_eff = _majority(eff_valid)
    assert eff >= need_eff, (
        f"K-efficiency: auc_mrmr[K={k_lo}] >= auc_mi[K={k_hi}] - {eff_margin} should "
        f"hold on a majority of seeds; got {eff}/{eff_valid} (need {need_eff}). "
        f"table={table}"
    )


# ---------------------------------------------------------------------------
# Honest control on linear, no-redundancy data -- surfaces a PROD BUG
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_biz_val_h2h_honest_tie_on_linear_no_redundancy():
    """CONTROL leg: on ``make_signal_plus_noise`` (linear additive signal, zero
    redundancy) MRMR has no structural edge over a plain MI filter, so the CORRECT
    expected behaviour is a documented within-epsilon TIE -- MRMR's selected set
    (raw + engineered, via ``transform``) should reach within 0.02 AUC of
    ``SelectKBest`` on a majority of seeds.

    This previously xfailed because MRMR returned an EMPTY ``support_`` (0-width
    transform) on a majority of seeds, violating the ``min_features_fallback>=1``
    non-empty guarantee. That defect was fixed (raw-redundancy drop no longer empties
    the selection via an un-replayable nested subsumer -- verified 0/6 empties), so
    the within-epsilon tie now holds and this is a passing value proof."""
    n = 1200 if is_fast_mode() else 1500
    margin = 0.02
    ties = 0
    rows = []
    for seed in _seeds():
        Xn, yn, sig = make_signal_plus_noise(n=n, p_signal=3, p_noise=16, seed=seed)
        X, y = as_df(Xn, yn)
        a_skb = _auc_on_cols(X, y, _skb_cols(X, y, 3, seed))
        sel = _fit_mrmr(X, y, seed, fe=True)
        xt = sel.transform(X)
        a_mrmr = (
            float(
                cross_val_score(
                    LogisticRegression(max_iter=400), xt, y, cv=5, scoring="roc_auc"
                ).mean()
            )
            if getattr(xt, "shape", (0, 0))[1] > 0
            else float("nan")
        )
        rows.append((seed, getattr(xt, "shape", (0, 0))[1], a_mrmr, a_skb))
        if not np.isnan(a_mrmr) and a_mrmr >= a_skb - margin:
            ties += 1

    print("\nhonest-tie control (signal_plus_noise, linear, no redundancy):")
    print("  seed  n_out   auc_mrmr   auc_skb     delta")
    for seed, nout, a_m, a_s in rows:
        d = a_m - a_s if not np.isnan(a_m) else float("nan")
        am = f"{a_m:>8.4f}" if not np.isnan(a_m) else "   EMPTY"
        print(f"  {seed:>4}  {nout:>5}   {am}   {a_s:>7.4f}   {d:>+7.4f}")

    need = _majority(len(_seeds()))
    assert ties >= need, (
        f"MRMR should produce a non-empty selection that ties SelectKBest within "
        f"{margin} on a majority of seeds; got {ties}/{len(_seeds())} (need {need}). "
        f"EMPTY-support collapse is the surfaced prod bug. rows={rows}"
    )


# ---------------------------------------------------------------------------
# pip mrmr head-to-head -- skipped today (not installed), live the day it is
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_biz_val_h2h_vs_pip_mrmr_when_installed():
    """When the pip ``mrmr`` (mrmr_selection) package is installed, our MRMR's top-4
    must stay within 0.05 AUC of ``mrmr_classif(X, y, K=4)`` on the dominant-cluster
    fixture (both are mRMR-family -- a documented near-tie, not a win claim).
    Skipped today: ``mrmr`` is not installed on this box (verified in the audit).
    """
    mrmr_pkg = pytest.importorskip("mrmr")
    n = _n()
    margin = 0.05
    ok = 0
    valid = 0
    for seed in _seeds():
        X, y = make_dominant_cluster_fixture(n=n, seed=seed)
        order = _mrmr_selection_order(X, y, seed)
        if len(order) < 4:
            continue
        a_ours = _auc_on_cols(X, y, order[:4])
        pip_cols = mrmr_pkg.mrmr_classif(X=X, y=y, K=4, show_progress=False)
        a_pip = _auc_on_cols(X, y, list(pip_cols))
        if np.isnan(a_ours) or np.isnan(a_pip):
            continue
        valid += 1
        if abs(a_ours - a_pip) <= margin:
            ok += 1
    assert valid >= 1, "no seed produced a comparable >=4-feature MRMR selection"
    need = _majority(valid)
    assert ok >= need, (
        f"mlframe MRMR and pip mrmr_classif should be within {margin} AUC on a "
        f"majority of seeds; got {ok}/{valid} (need {need})."
    )
