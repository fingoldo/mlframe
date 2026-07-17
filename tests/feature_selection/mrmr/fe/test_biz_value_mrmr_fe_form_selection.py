"""Regression guard for the FE pair-search winner-selection bug (2026-06-01).

Bug: when a raw pair produced a group of near-equivalent "leading" engineered
forms AND there were >2 numeric columns, the search re-ranked the leaders by an
EXTERNAL-VALIDATION score -- the MI of each candidate RECOMBINED WITH AN
UNRELATED THIRD FACTOR -- and selected the winner by THAT score alone. The
primary objective, ``MI(engineered_feature ; y)``, was discarded at the final
pick. Concretely, on ``y = a**2/b + log(c)*sin(d)`` the search returned
``add(log(c),reciproc(d))`` (target MI ~0.25) over the TRUE ``mul(log(c),sin(d))``
(target MI ~0.32) because the former happened to recombine better with an
external factor.

Fix: ``_select_single_best`` selects by PRIMARY target MI, with the
external-validation MI demoted to a TIE-BREAK among target-MI-equal leaders.

Two layers of coverage:
  * unit: ``_select_single_best`` honours primary > secondary > name ordering.
  * integration: on the canonical fixture the engineered (c,d) feature carries
    near-maximal target MI (i.e. the search no longer returns a clearly
    suboptimal form).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs import _select_single_best
from mlframe.feature_selection.filters.feature_engineering import get_new_feature_name

# config = (((idx_a, unary_a), (idx_b, unary_b)), binary_name, i)
_COLS = ["c", "d"]
_CFG_TRUE = (((0, "log"), (1, "sin")), "mul", 0)  # mul(log(c),sin(d))
_CFG_ALT = (((0, "log"), (1, "reciproc")), "add", 1)  # add(log(c),reciproc(d))
_CFG_Z = (((0, "identity"), (1, "sqr")), "add", 2)  # add(c,sqr(d)) -- name sorts late


def test_select_single_best_primary_target_mi_beats_secondary():
    """PRIMARY (target MI) must win even when SECONDARY (external-validation MI)
    strongly favours the other candidate. This is the exact regression: the true
    form has higher target MI but lower external-validation MI."""
    primary = {_CFG_TRUE: 0.32, _CFG_ALT: 0.25}
    secondary = {_CFG_TRUE: 0.10, _CFG_ALT: 0.99}  # would pick _CFG_ALT if used as primary
    winner = _select_single_best(primary, _COLS, secondary=secondary)
    assert winner == _CFG_TRUE, f"expected the max-target-MI form {get_new_feature_name(_CFG_TRUE, _COLS)}, got {get_new_feature_name(winner, _COLS)}"


def test_select_single_best_secondary_breaks_target_mi_ties():
    """When target MI is EXACTLY tied, the external-validation MI breaks the tie."""
    primary = {_CFG_TRUE: 0.30, _CFG_ALT: 0.30}
    secondary = {_CFG_TRUE: 0.10, _CFG_ALT: 0.99}
    winner = _select_single_best(primary, _COLS, secondary=secondary)
    assert winner == _CFG_ALT, "secondary score must break a primary tie"


def test_select_single_best_name_tiebreak_when_all_equal():
    """Full tie on primary AND secondary -> deterministic lexicographically-smallest
    engineered name. 'add(c,sqr(d))' < 'mul(log(c),sin(d))' so _CFG_Z wins."""
    primary = {_CFG_TRUE: 0.30, _CFG_Z: 0.30}
    winner = _select_single_best(primary, _COLS)  # no secondary
    names = {c: get_new_feature_name(c, _COLS) for c in primary}
    expected = min(primary, key=lambda c: names[c])
    assert winner == expected, f"name tie-break failed: {names}"


def test_select_single_best_usability_breaks_mi_ties():
    """LINEAR-USABILITY tie-break (2026-06-16): among forms with EQUAL target MI,
    prefer the one with the higher |corr(continuous y)| -- a linearly-usable leg.
    The canonical case is a bilinear product ``y=1.5*a*b`` where ``mul(a,b)`` (|corr|
    ~0.76), ``log(a)+log(b)`` (~0.61) and ``1/(a**2*b**2)`` (~0.004) all tie in
    binned MI but only ``mul`` lets a linear model recover the magnitude. Pre-fix the
    MI-tie was broken by extval-MI then NAME, so a useless inverse form could win and
    cap the downstream linear R2 (test_suite_fe_linear_recovery bilinear: 0.884<0.90)."""
    primary = {_CFG_TRUE: 0.30, _CFG_ALT: 0.30, _CFG_Z: 0.30}
    usability = {_CFG_TRUE: 0.76, _CFG_ALT: 0.61, _CFG_Z: 0.004}
    winner = _select_single_best(primary, _COLS, usability=usability)
    assert winner == _CFG_TRUE, (
        f"usability tie-break must pick the most linearly-usable MI-tied form "
        f"{get_new_feature_name(_CFG_TRUE, _COLS)}, got {get_new_feature_name(winner, _COLS)}"
    )


def test_select_single_best_usability_never_overrides_higher_mi():
    """Usability is a TIE-BREAK only: a higher-MI form still wins even when a lower-MI
    form is more linearly usable -- the MI-primary contract is preserved (no regression
    to the rank objective the tree list depends on)."""
    primary = {_CFG_TRUE: 0.32, _CFG_ALT: 0.25}
    usability = {_CFG_TRUE: 0.10, _CFG_ALT: 0.99}  # ALT more usable but lower MI
    winner = _select_single_best(primary, _COLS, usability=usability)
    assert winner == _CFG_TRUE, "usability must not override a genuinely higher target MI"


def test_select_single_best_usability_ranks_above_secondary():
    """On an MI tie, linear usability is decisive ABOVE the external-validation
    secondary -- the project prioritises linear usability over external generalisation
    when the rank objective (MI) cannot separate the forms."""
    primary = {_CFG_TRUE: 0.30, _CFG_ALT: 0.30}
    usability = {_CFG_TRUE: 0.80, _CFG_ALT: 0.10}
    secondary = {_CFG_TRUE: 0.10, _CFG_ALT: 0.99}  # secondary favours ALT
    winner = _select_single_best(primary, _COLS, secondary=secondary, usability=usability)
    assert winner == _CFG_TRUE, "usability must outrank the external-validation secondary on an MI tie"


def test_select_single_best_empty_returns_none():
    assert _select_single_best({}, _COLS) is None


def test_select_single_best_no_secondary_is_pure_max_primary():
    """Without a secondary dict, selection is pure max-primary (back-compat)."""
    primary = {_CFG_TRUE: 0.20, _CFG_ALT: 0.40}
    assert _select_single_best(primary, _COLS) == _CFG_ALT


# ---------------------------------------------------------------------------
# Quantised exact-MI tiebreaker (2026-06-26): the CPU and GPU scoring backends
# compute MI separately and can differ by fp reduction order (~1e-12). The
# exact-MI tiebreaker leg is SNAPPED to a 1e-7 grid so that sub-grid jitter
# never flips the pick between backends, while genuine within-band MI gaps
# still order correctly. These pin the contract in ``_fe_mi_contract``.
# ---------------------------------------------------------------------------
def test_quantize_mi_tiebreak_collapses_subgrid_preserves_genuine():
    from mlframe.feature_selection.filters._fe_mi_contract import quantize_mi_tiebreak

    # sub-grid jitter (< 1e-7) -> identical key
    assert quantize_mi_tiebreak(0.30000000) == quantize_mi_tiebreak(0.30000001)
    # genuine gap (>> 1e-7) -> distinct, order preserved
    assert quantize_mi_tiebreak(0.1180) > quantize_mi_tiebreak(0.1167)
    # quantum<=0 -> identity (no quantisation)
    assert quantize_mi_tiebreak(0.123456789, quantum=0.0) == 0.123456789


def test_select_single_best_tiebreak_is_backend_stable_under_fp_jitter():
    """Two band+usability-tied forms whose EXACT MI differs only by sub-grid fp
    jitter (~1e-8, the cross-backend reduction-order scale) must select the SAME
    form regardless of which backend's value is marginally higher -- i.e. the pick
    falls through to the deterministic name key, not the jitter. Pre-quantisation
    (raw ``float(kv[1])`` leg) this FLIPPED between the two jitter orderings."""
    band = 0.004  # > the jitter, so both forms are band-tied -> exact-MI leg decides
    # _CFG_ALT name 'add(log(c),reciproc(d))' < _CFG_TRUE 'mul(log(c),sin(d))' -> ALT wins on a name tie.
    cpu_view = {_CFG_TRUE: 0.30000001, _CFG_ALT: 0.30000000}  # TRUE marginally higher (one backend)
    gpu_view = {_CFG_TRUE: 0.30000000, _CFG_ALT: 0.30000001}  # ALT marginally higher (other backend)
    w_cpu = _select_single_best(cpu_view, _COLS, mi_band=band)
    w_gpu = _select_single_best(gpu_view, _COLS, mi_band=band)
    assert w_cpu == w_gpu == _CFG_ALT, (
        f"sub-grid jitter must not flip the pick: cpu={get_new_feature_name(w_cpu, _COLS)} "
        f"gpu={get_new_feature_name(w_gpu, _COLS)} (both must be the name-key winner _CFG_ALT)"
    )


def test_select_single_best_exact_mi_leg_still_resolves_genuine_within_band_gap():
    """Quantisation must NOT defang the exact-MI leg for a GENUINE within-band gap:
    the F2 'mixed' case (0.1180 vs 0.1167, gap 1.3e-3 >> the 1e-7 grid) is band-tied
    (band 0.004) and usability-tied, so the higher EXACT MI must still win."""
    band = 0.004
    primary = {_CFG_TRUE: 0.1180, _CFG_ALT: 0.1167}
    winner = _select_single_best(primary, _COLS, mi_band=band)
    assert winner == _CFG_TRUE, (
        "the exact-MI leg must still pick the higher within-band MI form; quantisation only collapses sub-grid (1e-7) jitter, not a 1.3e-3 genuine gap"
    )


# ---------------------------------------------------------------------------
# Integration: the search recovers a near-optimal (c,d) form on the canonical
# fixture (uniform[0,1] inputs, the regime the user reported).
# ---------------------------------------------------------------------------

# n reduced from 100_000 -> 30_000 (2026-06-13): at 30k the clean (c,d) interaction is already
# recovered as the canonical combined form (the standalone-vs-combined emission separates here),
# and 30k completes well within a generous per-test timeout. The 100k fit overran the suite's
# global --timeout=60 and made this test structurally un-runnable (it timed out, not asserted).
N = 30_000
NB = 10


def _binned(arr):
    from mlframe.feature_selection.filters.discretization import discretize_array

    arr = np.nan_to_num(np.asarray(arr, float), nan=0.0, posinf=0.0, neginf=0.0)
    return discretize_array(arr=arr, n_bins=NB, method="quantile", dtype=np.int32)


def _mi(xb, yb):
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes, merge_vars

    fd = np.column_stack([xb, yb]).astype(np.int32)
    fn = np.array([NB, NB], dtype=np.int64)
    cx, fx, _ = merge_vars(fd, (0,), None, fn, dtype=np.int32)
    cy, fy, _ = merge_vars(fd, (1,), None, fn, dtype=np.int32)
    return float(compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32))


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_search_recovers_near_optimal_cd_form(seed):
    """The engineered (c,d) feature MRMR returns must carry near-maximal target MI.

    Pins the fix: the pre-fix search returned add(log(c),reciproc(d)) at MI~0.25
    while the true mul(log(c),sin(d)) sits at MI~0.32. We assert the chosen (c,d)
    engineered column's MI clears 0.90 * the true-form MI -- i.e. the search no
    longer locks onto a clearly suboptimal representation.

    Runs with conditional_gate DEFAULT-ON (2026-06-13). The clean (c,d) interaction is
    recovered either as a standalone form or embedded in the canonical combined form
    ``add(abs(div(sqr(a),abs(b))),mul(log(c),sin(d)))`` -- both carry bare ``c`` AND ``d``.
    The regressed gate junk ``mul(reciproc(d),neg(gate_mask__c__b...))`` does NOT (its ``c``
    is hidden inside the gate column name, so it is not a bare token), so the locator below
    still distinguishes the recovered form from the broken one.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(seed)
    n = N
    a = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    d = rng.random(n)
    e = rng.random(n)
    f = rng.random(n)
    y = a**2 / b + f / 5.0 + np.log(c) * np.sin(d)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})

    fs = MRMR(verbose=0)
    fs.fit(df, pd.Series(y, name="y"))

    yb = _binned(y)
    true_mi = _mi(_binned(np.log(c) * np.sin(d)), yb)

    names = list(fs.get_feature_names_out())
    Xt = np.asarray(fs.transform(df))
    # locate engineered columns carrying the (c,d) interaction: mention both c AND d as BARE
    # tokens. The clean standalone form mul(log(c),sin(d)) and the combined form
    # add(...,mul(log(c),sin(d))) both qualify; the regressed gate junk does not (its c is
    # buried in gate_mask__c__b, not a bare token). The earlier ``not {a,b}`` exclusion rejected
    # the (equally clean) combined form and is dropped -- it conflated "captures the (c,d)
    # interaction" with "emits a STANDALONE (c,d) column", which the search does not guarantee.
    import re

    def bare(nm):
        return set(re.findall(r"(?<![A-Za-z_])([a-e])(?![A-Za-z_])", nm))

    cd_cols = [i for i, nm in enumerate(names) if "(" in nm and {"c", "d"} <= bare(nm)]
    assert cd_cols, f"no (c,d) engineered feature found in {names}"
    best_cd_mi = max(_mi(_binned(Xt[:, i]), yb) for i in cd_cols)
    assert best_cd_mi >= 0.90 * true_mi, (
        f"[seed={seed}] chosen (c,d) form MI={best_cd_mi:.4f} is far below the true mul(log(c),sin(d)) MI={true_mi:.4f}; search picked a suboptimal form"
    )
