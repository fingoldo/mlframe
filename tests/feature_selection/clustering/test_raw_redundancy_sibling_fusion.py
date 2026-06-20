"""Regression test for the NON-INVERTIBLE-FUSION raw-redundancy drop (BUG1, 2026-06-16).

A consuming engineered composite can fuse a raw operand with a SECOND signal-bearing
operand in a form NOT invertible from the composite alone -- the canonical case is an
additive ``add(a, sin(c))`` (raw ``a`` enters LINEARLY, ``sin(c)`` is a nuisance term).
Conditioning ``a`` on the fused sum ALONE leaves the ``sin(c)`` variation un-held across
the conditioning strata, so a FULLY-subsumed ``a`` keeps a spurious finite-sample residual
above the 5% self-retention bar and is wrongly KEPT.

This is the exact end-to-end failure on the seed-909 ``ratio_plus_trig`` / ``uniform`` /
``classification`` case at n=25000: the fitted MRMR produced the survivors
``add(a,sin(c))`` and ``div(sqr(a),exp(b))`` and KEPT raw ``a`` (subsumed, no private term)
because its base-conditioned residual was 6.66% > the 5% bar. The fix ALSO measures the
residual with the consumer's OTHER signal-bearing operands (the siblings) added to the
conditioning -- which HOLDS the nuisance term fixed -- and uses the SMALLEST debiased excess
across {base, +siblings} as the verdict (``a``: 6.66% -> 0.25%, cmi below floor -> DROP).

The test reproduces that EXACT decision deterministically by driving
``drop_redundant_raw_operands`` with the real generator's ``a/b/c`` columns and binary
target at seed 909 plus the two real engineered survivors -- no full MRMR fit / RNG
sensitivity. PRE-FIX it KEEPS ``a`` (verified by disabling the sibling block: base frac
6.66% > 5%); POST-FIX it DROPS ``a``. The over-drop control adds a genuine private linear
term and asserts ``a`` is then KEPT.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")
import pandas as pd

from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
    drop_redundant_raw_operands,
)
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin


def _bin10(x: np.ndarray) -> np.ndarray:
    return _quantile_bin(np.asarray(x, dtype=np.float64), nbins=10).astype(np.int64)


def _s909_fixture(*, private: bool):
    """Build the helper kwargs from the REAL generator data at the failing case.

    ``a`` enters the target ONLY through the additive fusion ``add(a, sin(c))`` and the
    ``div(sqr(a),exp(b))`` ratio (no private term) -> must DROP. When ``private`` is True an
    independent ``3*a`` linear term is fused into the (re-binned) target so ``a`` carries a
    residual the composite+sibling cannot span -> must be KEPT (over-drop control)."""
    from tests.feature_selection._mrmr_realistic_data import make_realistic_case

    df, y, _meta = make_realistic_case(
        seed=909, n=25000, distribution="uniform",
        target_family="ratio_plus_trig", task="classification",
    )
    a = df["a"].to_numpy(); bb = df["b"].to_numpy(); c = df["c"].to_numpy()
    add_ac = a + np.sin(c)          # add(a, sin(c)) -- a enters LINEARLY (non-invertible)
    div_ab = (a ** 2) / np.exp(bb)  # div(sqr(a), exp(b))
    y_arr = np.asarray(y).astype(np.int64)
    y_cont = None
    if private:
        # Re-target on a continuous score carrying a genuine independent 3*a term so the
        # over-drop control has a real private residual to protect.
        y_cont = add_ac + 0.5 * div_ab + 3.0 * a
        y_arr = _bin10(y_cont)

    cols = ["a", "b", "c", "add(a,sin(c))", "div(sqr(a),exp(b))"]
    data = np.column_stack([
        _bin10(a), _bin10(bb), _bin10(c), _bin10(add_ac), _bin10(div_ab),
    ])
    return dict(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2, 3, 4],
        raw_name_set={"a", "b", "c"}, y_binned=y_arr, y_continuous=y_cont,
        engineered_continuous={"add(a,sin(c))": add_ac, "div(sqr(a),exp(b))": div_ab},
        replayable_eng_names={"add(a,sin(c))", "div(sqr(a),exp(b))"},
        recipes=None, raw_X=pd.DataFrame({"a": a, "b": bb, "c": c}), seed=909,
    )


def test_s909_noninvertible_additive_fusion_subsumed_raw_drops():
    """Seed-909 BUG1: subsumed raw ``a`` fused via ``add(a, sin(c))`` (no private term)
    MUST drop. PRE-FIX kept it (base residual 6.66% > 5% bar); the sibling-conditioned
    residual collapses to ~0.25% -> DROP."""
    cfg = _s909_fixture(private=False)
    kept, dropped = drop_redundant_raw_operands(**cfg)
    kept_names = {cfg["cols"][i] for i in kept}
    assert "a" in dropped, (
        f"subsumed raw 'a' (fused via add(a,sin(c)), no private term) not dropped; "
        f"dropped={dropped} kept={sorted(kept_names)}"
    )
    assert "a" not in kept_names, f"'a' still in support: {sorted(kept_names)}"
    # The engineered survivors must remain (the drop only removes subsumed raws).
    assert {"add(a,sin(c))", "div(sqr(a),exp(b))"} <= kept_names


def test_s909_genuine_private_linear_raw_kept_under_sibling_conditioning():
    """OVER-DROP CONTROL: the SAME fusion but ``a`` ALSO carries an independent ``3*a``
    linear term the composite+siblings cannot span -> KEPT. Confirms the
    sibling-conditioning min never over-drops a genuine private raw."""
    cfg = _s909_fixture(private=True)
    kept, dropped = drop_redundant_raw_operands(**cfg)
    assert "a" not in dropped, (
        f"OVER-DROP: 'a' has a genuine private linear term (3*a) yet was dropped: {dropped}"
    )
    assert "a" in {cfg["cols"][i] for i in kept}


def test_floor_margin_mult_default_is_byte_identical():
    """``floor_margin_mult`` defaults to 1.0 = the historical bare ``cmi > floor`` significance
    leg, so every existing caller is byte-identical. Pin it on the s909 fixtures (both legs)."""
    for _private in (False, True):
        cfg = _s909_fixture(private=_private)
        kept_a, dropped_a = drop_redundant_raw_operands(**cfg)
        kept_b, dropped_b = drop_redundant_raw_operands(**dict(cfg), floor_margin_mult=1.0)
        assert dropped_a == dropped_b and kept_a == kept_b


def test_floor_margin_mult_monotone_tightens_significance():
    """The I4b post-retention lever: a higher ``floor_margin_mult`` makes the significance leg
    STRICTER, so it can only ever drop MORE raws (a grazing-the-floor operand fails the tighter
    bar), never fewer. Pin the monotonicity so a future change cannot invert the lever."""
    cfg = _s909_fixture(private=False)
    _k1, d1 = drop_redundant_raw_operands(**dict(cfg), floor_margin_mult=1.0)
    _k2, d2 = drop_redundant_raw_operands(**dict(cfg), floor_margin_mult=1.5)
    _k3, d3 = drop_redundant_raw_operands(**dict(cfg), floor_margin_mult=3.0)
    assert set(d1) <= set(d2) <= set(d3)


def test_floor_margin_mult_keeps_strong_private_raw():
    """OVER-DROP CONTROL: a raw with a genuine independent linear term clears the permutation
    floor by a WIDE margin, so even the stricter ``floor_margin_mult=1.5`` the post-retention
    sweep uses must KEEP it (the lever drops grazing artifacts, never robust private signal)."""
    cfg = _s909_fixture(private=True)
    _kept, dropped = drop_redundant_raw_operands(**dict(cfg), floor_margin_mult=1.5)
    assert "a" not in dropped, (
        f"OVER-DROP: 'a' carries a genuine private 3*a term yet dropped under "
        f"floor_margin_mult=1.5: {dropped}"
    )
    assert "a" in {cfg["cols"][i] for i in _kept}
