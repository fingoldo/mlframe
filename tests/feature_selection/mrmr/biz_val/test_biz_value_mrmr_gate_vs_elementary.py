"""Regression guard: conditional_gate (default-ON) must COMPETE with, not SUPPRESS, the
clean elementary pair form (2026-06-13).

Bug: enabling ``fe_conditional_gate_enable`` (now default-ON) regressed the (c,d) recovery on
CASE1 ``y = a**2/b + f/5 + log(c)*sin(d)``. A high-MI ``gate_mask`` column built FROM raw ``c``
is selected by the greedy screen AHEAD of raw ``c``; raw ``c`` is then dropped from
``selected_vars`` (redundant given the gate) and re-enters the FE pool only via the synergy
bootstrap, which tags it ``synergy_added`` and forces the elementary (c,d) pair onto the STRICTER
``fe_synergy_min_prevalence`` bar. The clean ``mul(log(c),sin(d))`` (target MI ~0.31) misses that
bar by a hair and is NOT emitted; the search returns a linearly-useless gate composite
(``mul(reciproc(d),neg(gate_mask...))``) / a lower-MI escalation form (~0.26) instead.

Fix: the raw source operands of a SELECTED gate_mask / row_argmax feature are reclassified from
synergy-bootstrap to REGULARLY-selected, so their elementary pair competes on the lenient
prevalence bar. The gate column itself stays selected + pairable, so a genuinely warped
interaction the elementary library cannot express is still captured via the gate (CASE2).

Pins:
  * CASE1 (gate ON, default): the recovered (c,d) engineered feature carries near-true MI
    (>= 0.90 * the true mul(log(c),sin(d)) MI) -- the clean elementary form WINS.
  * CASE2 ``y = 0.2*a**2/b + f/5 + log(c*2)*sin(d/3)`` (gate ON): the (c,d) interaction is STILL
    captured (some engineered feature touching c & d carries materially more MI than raw c / raw d
    alone) -- the gate composite still wins where no clean elementary form matches the warp.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

NB = 10
N = 30_000  # tractable; the standalone (c,d) form already separates from the combined a,b,c,d form here.


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


def _bare(nm):
    return set(re.findall(r"(?<![A-Za-z_])([a-e])(?![A-Za-z_])", nm))


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_case1_clean_elementary_cd_form_recovered_with_gate_on():
    """CASE1: with conditional_gate ON (default) the clean (c,d) elementary form competes and wins."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(0)
    n = N
    a = rng.random(n); b = rng.random(n); c = rng.random(n)
    d = rng.random(n); e = rng.random(n); f = rng.random(n)
    y = a**2 / b + f / 5.0 + np.log(c) * np.sin(d)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})

    fs = MRMR(verbose=0)  # fe_conditional_gate_enable defaults to True
    fs.fit(df, pd.Series(y, name="y"))

    yb = _binned(y)
    true_mi = _mi(_binned(np.log(c) * np.sin(d)), yb)
    names = list(fs.get_feature_names_out())
    Xt = np.asarray(fs.transform(df))
    # any engineered feature touching c AND d (the standalone (c,d) form OR the combined
    # a,b,c,d additive form both embed mul(log(c),sin(d))).
    cd = [i for i, nm in enumerate(names) if "(" in nm and {"c", "d"} <= _bare(nm)]
    assert cd, f"no (c,d) engineered feature found in {names}"
    best = max(_mi(_binned(Xt[:, i]), yb) for i in cd)
    assert best >= 0.90 * true_mi, (
        f"clean (c,d) form suppressed with gate ON: best MI={best:.4f} < 0.90*true {true_mi:.4f}; "
        f"names={names}"
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_case2_warped_cd_interaction_still_captured_with_gate_on():
    """CASE2: the gate genuinely helps -- the warped (c,d) interaction is captured (via the gate)
    and carries materially more MI than either raw operand alone."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(0)
    n = N
    a = rng.random(n); b = rng.random(n); c = rng.random(n)
    d = rng.random(n); e = rng.random(n); f = rng.random(n)
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})

    fs = MRMR(verbose=0)  # gate ON
    fs.fit(df, pd.Series(y, name="y"))

    yb = _binned(y)
    raw_floor = max(_mi(_binned(c), yb), _mi(_binned(d), yb))
    names = list(fs.get_feature_names_out())
    Xt = np.asarray(fs.transform(df))
    # Any engineered feature that references BOTH c and d -- including via a gate_mask__c__d
    # column (whose c/d are buried in the name, so match the substring, not just bare tokens).
    cd = [
        i for i, nm in enumerate(names)
        if ("c" in nm and "d" in nm) and ("(" in nm or "gate_mask" in nm or "argmax" in nm)
    ]
    assert cd, f"the (c,d) interaction was NOT captured with gate ON: {names}"
    best = max(_mi(_binned(Xt[:, i]), yb) for i in cd)
    assert best > raw_floor + 0.02, (
        f"captured (c,d) feature MI={best:.4f} does not beat the raw operand floor {raw_floor:.4f}; "
        f"names={names}"
    )
