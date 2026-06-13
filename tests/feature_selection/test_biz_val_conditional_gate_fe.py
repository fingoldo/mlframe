"""biz_value tests for the CONDITIONAL-GATE and ROW-ARGMAX prototype operators (``filters/_conditional_gate_fe_proto``).

Quantitative-win + specificity floors from ``_benchmarks/bench_frontier_candidates`` (measured single-column binned-MI lift over
the BEST shipped operator on the natural target): gate_select +0.55, gate_mask +0.31, argmax +0.55; all REJECTED (negative lift)
on smooth / noise / ordinary-interaction controls. Floors set ~15% below measured so seed noise does not trip them. n<=2000, fast.

These pin the operators' edge over the existing catalog so a future "the gate / argmax adds nothing" regression fails the win.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._conditional_gate_fe_proto import (
    apply_conditional_gate,
    apply_row_argmax,
    scan_conditional_gate,
    scan_row_argmax,
)
from mlframe.feature_selection.filters._pairwise_modular_fe import _mi

NBINS = 12


def _best_existing_mi(X: pd.DataFrame, y, cols) -> float:
    """Max binned-MI over the best shipped operators a selector already has: raw cols, pairwise diff / prod / ratio, row-max/min."""
    arrs = {c: np.asarray(X[c], dtype=np.float64) for c in cols}
    cands = {f"raw_{c}": arrs[c] for c in cols}
    names = list(cols)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u, v = names[i], names[j]
            cands[f"prod_{u}_{v}"] = arrs[u] * arrs[v]
            cands[f"diff_{u}_{v}"] = arrs[u] - arrs[v]
            cands[f"ratio_{u}_{v}"] = arrs[u] / (np.abs(arrs[v]) + 1e-6)
    stk = np.stack([arrs[c] for c in cols], axis=1)
    cands["rowmax"] = stk.max(axis=1)
    cands["rowmin"] = stk.min(axis=1)
    yi = np.asarray(y).astype(np.int64)
    return max(_mi(np.asarray(v, dtype=np.float64), yi, nbins=NBINS) for v in cands.values())


def _gate_select_target(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    sel = np.where(c > 0.0, a, b)
    return pd.DataFrame({"a": a, "b": b, "c": c}), (sel > np.median(sel)).astype(int)


def _gate_mask_target(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, c = rng.normal(0, 1, n), rng.normal(0, 1, n)
    masked = (c > 0.0).astype(float) * a
    return pd.DataFrame({"a": a, "b": rng.normal(0, 1, n), "c": c}), (masked > np.median(masked)).astype(int)


def _argmax_target(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c}), np.argmax(np.stack([a, b, c], axis=1), axis=1)


def _smooth_target(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c}), ((a + 0.5 * b) > 0).astype(int)


def test_biz_val_gate_select_beats_best_existing_operator():
    """Regime-switch ``c>0 ? a : b`` MI must beat the best shipped operator (raw / diff / prod / ratio / rowmax) by >=0.45. Measured +0.55."""
    lifts = []
    for s in (1, 7, 13, 42, 101):
        X, y = _gate_select_target(s)
        yi = np.asarray(y).astype(np.int64)
        gate = apply_conditional_gate(np.asarray(X["a"]), np.asarray(X["b"]), np.asarray(X["c"]), 0.0, "select")
        lifts.append(_mi(gate, yi, NBINS) - _best_existing_mi(X, y, ("a", "b", "c")))
    assert float(np.mean(lifts)) >= 0.45, f"gate_select lift {np.mean(lifts):.3f} < 0.45"
    assert float(np.min(lifts)) >= 0.40, f"gate_select worst-seed lift {np.min(lifts):.3f} < 0.40"


def test_biz_val_gate_mask_beats_best_existing_operator():
    """Masked interaction ``1[c>0]*a`` MI must beat the best shipped operator by >=0.25. Measured +0.31."""
    lifts = []
    for s in (1, 7, 13, 42, 101):
        X, y = _gate_mask_target(s)
        yi = np.asarray(y).astype(np.int64)
        gate = apply_conditional_gate(np.asarray(X["a"]), np.asarray(X["b"]), np.asarray(X["c"]), 0.0, "mask")
        lifts.append(_mi(gate, yi, NBINS) - _best_existing_mi(X, y, ("a", "b", "c")))
    assert float(np.mean(lifts)) >= 0.25, f"gate_mask lift {np.mean(lifts):.3f} < 0.25"


def test_biz_val_row_argmax_beats_best_existing_operator():
    """Row-argmax(a,b,c) MI must beat the best shipped operator (incl. all pairwise diffs) by >=0.45. Measured +0.55."""
    lifts = []
    for s in (1, 7, 13, 42, 101):
        X, y = _argmax_target(s)
        yi = np.asarray(y).astype(np.int64)
        am = apply_row_argmax([np.asarray(X[c]) for c in ("a", "b", "c")])
        lifts.append(_mi(am, yi, NBINS) - _best_existing_mi(X, y, ("a", "b", "c")))
    assert float(np.mean(lifts)) >= 0.45, f"argmax lift {np.mean(lifts):.3f} < 0.45"
    assert float(np.min(lifts)) >= 0.40, f"argmax worst-seed lift {np.min(lifts):.3f} < 0.40"


def test_biz_val_gate_does_not_beat_existing_on_smooth_control():
    """Specificity: on a smooth linear target the gate must NOT beat the best existing operator (lift <= 0). Measured -0.29."""
    lifts = []
    for s in (1, 7, 13, 42, 101):
        X, y = _smooth_target(s)
        yi = np.asarray(y).astype(np.int64)
        gate = apply_conditional_gate(np.asarray(X["a"]), np.asarray(X["b"]), np.asarray(X["c"]), 0.0, "select")
        lifts.append(_mi(gate, yi, NBINS) - _best_existing_mi(X, y, ("a", "b", "c")))
    assert float(np.mean(lifts)) <= 0.0, f"gate beat existing on smooth control (lift {np.mean(lifts):.3f})"


def test_biz_val_argmax_does_not_beat_existing_on_smooth_control():
    """Specificity: on a smooth linear target row-argmax must NOT beat the best existing operator. Measured -0.29."""
    lifts = []
    for s in (1, 7, 13, 42, 101):
        X, y = _smooth_target(s)
        yi = np.asarray(y).astype(np.int64)
        am = apply_row_argmax([np.asarray(X[c]) for c in ("a", "b", "c")])
        lifts.append(_mi(am, yi, NBINS) - _best_existing_mi(X, y, ("a", "b", "c")))
    assert float(np.mean(lifts)) <= 0.0, f"argmax beat existing on smooth control (lift {np.mean(lifts):.3f})"


def test_biz_val_argmax_detector_fires_on_target_not_on_noise():
    """The cheap-first detector recovers argmax structure on its target and stays silent on pure noise."""
    Xt, yt = _argmax_target(7)
    assert scan_row_argmax(Xt, yt, list(Xt.columns), rng_seed=7), "argmax detector failed to fire on its target"
    rng = np.random.default_rng(7)
    n = 2000
    Xn = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)})
    yn = rng.integers(0, 2, n)
    assert not scan_row_argmax(Xn, yn, list(Xn.columns), rng_seed=7), "argmax detector fired on pure noise"


def test_biz_val_gate_detector_fires_on_regime_target():
    """The cheap-first gate detector recovers the regime-switch on its natural target."""
    X, y = _gate_select_target(13)
    hits = scan_conditional_gate(X, y, list(X.columns), rng_seed=13)
    assert hits, "gate detector failed to fire on regime-switch target"
    assert hits[0]["mode"] in ("select", "mask")


def test_apply_conditional_gate_rejects_unknown_mode():
    with pytest.raises(ValueError):
        apply_conditional_gate(np.zeros(3), np.zeros(3), np.zeros(3), 0.0, "bogus")
