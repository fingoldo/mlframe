"""High-dimensional p >> n biz_value contract for the practical selector families.

The FS biz_value suite is overwhelmingly n >> p. The p >> n regime (few samples, a SEA of
features) is where univariate-relevance selectors overfit to spurious correlations: with p >> n
SOME pure-noise column WILL correlate with y by chance, and a selector with no multiple-comparison
control returns a huge spurious set. This is the regime real genomics / text-FS users hit, and it
was uncovered.

Datasets: n in {40, 60, 80}, p in {150, 300}, a handful (p_signal=4) of truly informative columns
embedded in an overwhelming sea of iid N(0,1) noise. Target is a linear-additive function of the
informative columns only.

For each PRACTICAL selector (MRMR, RFECV -- the heavy SHAP/Boruta/Hybrid members are excluded: they
are ~10-12 s each AND not the relevant univariate-relevance family) three things are asserted:

  (a) COMPLETENESS -- fit completes without crashing / singular-matrix error;
  (b) RECOVERY     -- informative columns are recovered above the chance rate;
  (c) FP-CONTROL   -- the selected set does NOT blow up to ~all p features (multiple-comparison
                      inflation is controlled).

Two real, MEASURED selector weaknesses surface here and are pinned as ``xfail(strict=False)`` rather
than silenced with a weakened assertion (per CLAUDE.md "real selector weakness -> xfail"):

  * MRMR recovers FP-control beautifully (it returns a tight 1-5 feature set on p up to 300) but its
    3-permutation confirmation gate loses POWER at n=40-80, so recovery of the 4 informative columns
    is below the chance bar on a majority of seeds -> RECOVERY xfail.
  * RFECV recovers all 4 informative columns every time but with the argmax rule (no plateau) it
    selects ALL p features on p >> n -> FP-CONTROL xfail (the same RFECV argmax FP gap that
    ``test_biz_val_null_fdr_and_stability.py`` documents on the pure-null fixture, here under signal).

All floors are calibrated from a measured dev run (CPU, seeds {0,1,2}) and pinned 5-15% below the
measured value. Seeds are fixed; majority-of-seeds is used everywhere (selectors are high-variance).
CPU-only -- no GPU path. n<=80, p<=300, MRMR fe disabled -> every test fits in well under 50 s.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection._selector_factories import SELECTOR_SPECS, selected_names
from tests.feature_selection.conftest import is_fast_mode

# Practical univariate-relevance / wrapper selectors for the p >> n regime. The heavy SHAP-based
# members (ShapProxiedFS, BorutaShap, HybridSelector) are intentionally excluded: each is ~10-12 s
# per fit (would blow the per-test budget across the (n,p) grid) and they are not the family this
# regime stresses (multiple-comparison control on univariate relevance).
_PRACTICAL = ["MRMR", "RFECV"]

_P_SIGNAL = 4
_NPS = [(40, 150), (60, 150), (60, 300), (80, 300)]
_SEEDS = [0] if is_fast_mode() else [0, 1, 2]


def make_p_gt_n(n: int, p: int, p_signal: int = _P_SIGNAL, seed: int = 0):
    """Linear-additive binary target on ``p_signal`` informative cols + (p - p_signal) iid N(0,1) noise.

    ``y = 1[ sum(X[:, :p_signal]) + 0.3*eps > median ]``. The informative columns are indices
    ``0..p_signal-1``; everything else is pure noise. With p >> n some noise column correlates with y
    by chance -- that is the multiple-comparison trap this fixture exists to probe.

    Returns ``(X_df, y_ser, signal_indices)``.
    """
    rng = np.random.default_rng(seed)
    X_sig = rng.standard_normal((n, p_signal))
    X_noise = rng.standard_normal((n, p - p_signal))
    X = np.column_stack([X_sig, X_noise])
    score = X_sig.sum(axis=1) + 0.3 * rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    cols = [f"x{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y"), list(range(p_signal))


def _recovered_and_total(sel, signal):
    """``(n_signal_recovered, n_total_selected)`` for a fitted selector, crediting engineered names."""
    names = list(selected_names(sel))
    sig = set(int(i) for i in signal)
    refs: set = set()
    for nm in names:
        refs |= set(int(m) for m in re.findall(r"x(\d+)", nm))
    return len(refs & sig), len(names)


def _fit(spec, n, p, seed):
    sel = spec.make("binary")
    X, y, signal = make_p_gt_n(n, p, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.fit(X, y)
    return sel, signal


# ---------------------------------------------------------------------------
# (a) COMPLETENESS -- every practical selector fits on p >> n without crashing.
# A singular-matrix / empty-screen / shape error here is a hard regression.
# ---------------------------------------------------------------------------


def _spec_param(name):
    spec = SELECTOR_SPECS[name]
    marks = [pytest.mark.slow] if spec.slow else []
    return pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize("name", [_spec_param(n) for n in _PRACTICAL])
@pytest.mark.parametrize("n,p", _NPS)
def test_biz_val_p_gt_n_fit_completes(name, n, p):
    """Fit completes on p >> n and yields a usable, in-range support. No singular-matrix / empty /
    shape crash, and the selected count is in ``[0, p]``. This is the floor every later assertion
    builds on -- a selector that CRASHES on p >> n fails here, not silently in a recovery metric."""
    if SELECTOR_SPECS[name].needs_shap:
        pytest.importorskip("shap")
    sel, signal = _fit(SELECTOR_SPECS[name], n, p, seed=_SEEDS[0])
    _, total = _recovered_and_total(sel, signal)
    assert 0 <= total <= p, f"{name}: selected count {total} out of range [0,{p}] at n={n} p={p}"
    # transform must round-trip to the selected width (proves the fitted support is coherent)
    Xt = sel.transform(make_p_gt_n(n, p, seed=_SEEDS[0])[0])
    assert Xt.shape[0] == n and Xt.shape[1] >= 0


# ---------------------------------------------------------------------------
# (b) RECOVERY -- informative columns recovered above the chance rate.
# Chance recovery of >=1 of 4 informative cols when picking K of p at random is ~K*4/p; for the
# tight-K selectors (MRMR picks ~1-5 of 300) that chance bar is well under 1, so "recover >=1 on a
# majority of seeds, >=2 cumulatively" is a real above-chance signal.
#
# MEASURED (CPU, seeds {0,1,2}, fe disabled):
#   MRMR  n=40 p=150 recovered(of4)=[1,0,0] total=[1,5,4]
#   MRMR  n=60 p=150 recovered(of4)=[3,1,0] total=[3,1,2]
#   MRMR  n=60 p=300 recovered(of4)=[0,1,0] total=[1,4,2]
#   MRMR  n=80 p=300 recovered(of4)=[0,1,2] total=[1,1,3]
#   RFECV all cells recovered(of4)=[4,4,4]  total=[p,p,p]  (recovers all, selects all)
# -> RFECV recovery is a clean PASS; MRMR recovery is below the bar on a majority of seeds at the
#    smallest n (the 3-perm confirmation gate loses power) -> RECOVERY xfail for MRMR.
# ---------------------------------------------------------------------------

_MRMR_RECOVERY_XFAIL = (
    "FS GAP: MRMR's 3-permutation confirmation gate loses POWER at n=40-80 in the p>>n regime -- "
    "recovery of the 4 informative columns falls below the above-chance bar on a majority of seeds "
    "(measured e.g. n=40,p=150 recovered=[1,0,0]; n=60,p=300 recovered=[0,1,0]). The selector errs "
    "toward UNDER-selection here (excellent FP-control, weak power); a higher full_npermutations or "
    "an n-aware gate would restore power. Not a crash -- a documented power gap."
)


@pytest.mark.parametrize("name", [_spec_param(n) for n in _PRACTICAL])
def test_biz_val_p_gt_n_recovers_signal_above_chance(name):
    """Across the (n,p) grid the selector recovers the informative columns above chance: at least one
    of the 4 informative columns on a MAJORITY of (cell,seed) trials. RFECV clears this; MRMR's
    power collapse at small n is pinned as an xfail (its under-selection trades recovery for FP-control)."""
    if SELECTOR_SPECS[name].needs_shap:
        pytest.importorskip("shap")
    spec = SELECTOR_SPECS[name]
    hits, trials, detail = 0, 0, []
    for (n, p) in _NPS:
        for seed in _SEEDS:
            sel, signal = _fit(spec, n, p, seed)
            rec, total = _recovered_and_total(sel, signal)
            trials += 1
            if rec >= 1:
                hits += 1
            detail.append((n, p, seed, rec, total))
    msg = f"{name}: recovered>=1 on {hits}/{trials} trials; detail (n,p,seed,rec,total)={detail}"
    if name == "MRMR" and hits <= trials // 2:
        pytest.xfail(_MRMR_RECOVERY_XFAIL + f" | {msg}")
    assert hits > trials // 2, msg


# ---------------------------------------------------------------------------
# (c) FP-CONTROL -- the selected set does not blow up to ~all p features.
# A multiple-comparison-controlling selector keeps the total selected well below p even when SOME
# noise column spuriously correlates with y. MEASURED: MRMR total in {1..5} of p (TIGHT control);
# RFECV total == p on every cell (argmax rule, no plateau -> selects ALL features).
# -> MRMR FP-control PASS; RFECV FP-control xfail (selects all p).
# ---------------------------------------------------------------------------

_RFECV_FP_XFAIL = (
    "FS GAP: RFECV with the argmax selection rule (no plateau / 1-SE tie-break) selects ALL p features "
    "on p>>n (measured total==p on every (n,p) cell). With p>>n the CV-score curve is non-decreasing "
    "out to the full set, so argmax picks all p -- zero multiple-comparison control. The pure-null "
    "sibling (test_biz_val_null_fdr_and_stability.py) documents the same argmax FP gap; the real fix "
    "needs an outer-loop / plateau-aware rule, not a weakened assertion."
)

# FP-control ceiling: a controlled selector keeps the set small. p//3 is a generous bound (it would
# still catch a selector that admits a third of all noise) while leaving MRMR (~1-5 of 300) ample room.
_FP_CEILING = lambda p: max(20, p // 3)


@pytest.mark.parametrize("name", [_spec_param(n) for n in _PRACTICAL])
def test_biz_val_p_gt_n_false_positive_rate_bounded(name):
    """The selector controls multiple-comparison inflation: the total selected stays under a generous
    ceiling (``max(20, p/3)``) on a MAJORITY of (cell,seed) trials -- it must NOT return ~all p
    features. MRMR's tight 1-5 set clears this with room; RFECV's select-all argmax behaviour is
    pinned as an xfail documenting the FP-control gap."""
    if SELECTOR_SPECS[name].needs_shap:
        pytest.importorskip("shap")
    spec = SELECTOR_SPECS[name]
    bounded, trials, detail = 0, 0, []
    for (n, p) in _NPS:
        for seed in _SEEDS:
            sel, signal = _fit(spec, n, p, seed)
            rec, total = _recovered_and_total(sel, signal)
            trials += 1
            if total <= _FP_CEILING(p):
                bounded += 1
            detail.append((n, p, seed, total, _FP_CEILING(p)))
    msg = f"{name}: bounded on {bounded}/{trials} trials; detail (n,p,seed,total,ceiling)={detail}"
    if name == "RFECV" and bounded <= trials // 2:
        pytest.xfail(_RFECV_FP_XFAIL + f" | {msg}")
    assert bounded > trials // 2, msg


# ---------------------------------------------------------------------------
# Joint recovery + FP-control PASS for the selector that controls both, on a single representative
# cell -- the positive contract a future fix to the xfail'd legs should also clear. MRMR's FP-control
# is rock-solid, so this pins it as a hard, non-xfail floor (and serves as the fast-mode representative).
# ---------------------------------------------------------------------------


def test_biz_val_p_gt_n_mrmr_fp_control_hard_floor():
    """MRMR keeps a TIGHT selected set across the whole p>>n grid (multiple-comparison control is its
    strength even where its recovery power is weak): total selected <= max(20, p/3) on EVERY (cell,seed).
    This is the non-xfail counterpart to RFECV's select-all gap -- a real, hard, quantitative win."""
    spec = SELECTOR_SPECS["MRMR"]
    overs = []
    for (n, p) in _NPS:
        for seed in _SEEDS:
            sel, signal = _fit(spec, n, p, seed)
            _, total = _recovered_and_total(sel, signal)
            if total > _FP_CEILING(p):
                overs.append((n, p, seed, total, _FP_CEILING(p)))
    assert not overs, f"MRMR FP-control breached at p>>n (should keep a tight set): overs={overs}"
