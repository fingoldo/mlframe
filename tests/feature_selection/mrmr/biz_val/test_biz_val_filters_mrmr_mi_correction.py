"""biz_value + activation tests for ``MRMR(mi_correction='miller_madow')``.

The Miller-Madow correction subtracts the closed-form plug-in MI bias ``(k_x-1)(k_y-1)/(2n)`` from the relevance score so a high-cardinality NOISE feature no
longer out-ranks a low-cardinality TRUE-relevant feature at small n. The knob was previously a DEAD constructor param (validated + stored but never activated in
the fit path); these tests pin (a) that it now actually fires + resets the thread-local, and (b) the quantitative kernel-level bias removal so a future regression
that re-deadens the knob or breaks the kernel fails the win, not just a shape check.

Verdict (qual-21): NOT a default flip -- the default mdlp discretization + permutation null-debias already neutralize this bias end-to-end (see
mlframe/feature_selection/_benchmarks/bench_mi_correction_miller_madow.py). Kept as an opt-in for the legacy fixed-bin / no-permutation-screen regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_biz_val_mi_correction_kernel_removes_high_cardinality_bias():
    """Floor: MM removes >=0.05 nats of spurious MI from a 60-level noise feature vs binary y at n=400. Measured 0.074 (plug-in 0.088 -> MM 0.015); 30% margin.

    A regression that breaks the closed-form bias term, the occupied-bin counting, or the dispatch would drop the removed-bias toward 0 and fail this floor.
    """
    from mlframe.feature_selection.filters.info_theory import (
        compute_mi_from_classes,
        compute_mi_mm_from_classes,
        merge_vars,
    )

    rng = np.random.default_rng(0)
    n = 400
    x = rng.integers(0, 60, n)  # high-cardinality pure noise
    y = rng.integers(0, 2, n)
    fd = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([60, 2])
    cx, fx, _ = merge_vars(factors_data=fd, vars_indices=np.array([0]), var_is_nominal=None, factors_nbins=nbins)
    cy, fy, _ = merge_vars(factors_data=fd, vars_indices=np.array([1]), var_is_nominal=None, factors_nbins=nbins)
    plugin = compute_mi_from_classes(cx, fx, cy, fy)
    mm = compute_mi_mm_from_classes(cx, fx, cy, fy)
    removed = plugin - mm
    assert removed >= 0.05, f"MM should remove >=0.05 nats of plug-in bias; got {removed:.4f} (plugin={plugin:.4f}, mm={mm:.4f})"
    assert mm >= 0.0, "MM-corrected MI is floored at 0"


def test_biz_val_mi_correction_kernel_noop_on_lowcard_signal():
    """MM must NOT damage a genuine low-cardinality (binary) signal: removed bias ~0 (k_x=2 -> (k_x-1)(k_y-1)/(2n) tiny), MM MI stays within 10% of plug-in."""
    from mlframe.feature_selection.filters.info_theory import (
        compute_mi_from_classes,
        compute_mi_mm_from_classes,
        merge_vars,
    )

    rng = np.random.default_rng(1)
    n = 400
    x = rng.integers(0, 2, n)
    y = (x ^ (rng.random(n) < 0.1).astype(int)).astype(np.int32)  # noisy copy of x
    fd = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([2, 2])
    cx, fx, _ = merge_vars(factors_data=fd, vars_indices=np.array([0]), var_is_nominal=None, factors_nbins=nbins)
    cy, fy, _ = merge_vars(factors_data=fd, vars_indices=np.array([1]), var_is_nominal=None, factors_nbins=nbins)
    plugin = compute_mi_from_classes(cx, fx, cy, fy)
    mm = compute_mi_mm_from_classes(cx, fx, cy, fy)
    assert plugin > 0.1, "binary signal should have real MI"
    assert mm >= 0.9 * plugin, f"MM must not damage low-card signal: mm={mm:.4f} vs plugin={plugin:.4f}"


def test_mi_correction_knob_activates_and_resets_thread_local(monkeypatch):
    """Regression: ``mi_correction='miller_madow'`` was a dead knob (validated + stored but NEVER wired into fit). Pin that fit() ACTIVATES the thread-local mid-fit
    and the finally resets it. Captures the state during fit by spying on the setter, so this FAILS on pre-fix code where set_mi_miller_madow was never called.
    """
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.filters.mrmr import _mrmr_class as mc
    from mlframe.feature_selection.filters.info_theory import use_mi_miller_madow

    seen = []
    real_set = mc.set_mi_miller_madow

    def _spy(active):
        """Record every activation call, then delegate to the real setter."""
        seen.append(bool(active))
        return real_set(active)

    # ``_mrmr_class.py`` does ``from ..info_theory import set_mi_miller_madow`` at module scope,
    # binding the name INTO its own namespace -- patching the dispatch module or the facade
    # re-export doesn't touch that local binding, so the spy must patch it directly here.
    monkeypatch.setattr(mc, "set_mi_miller_madow", _spy)

    rng = np.random.default_rng(7)
    n = 500
    xt = rng.integers(0, 2, (n, 2)).astype(float)
    noise = rng.standard_normal((n, 3))
    X = pd.DataFrame(np.column_stack([xt, noise]), columns=[f"f{i}" for i in range(5)])
    y = (xt @ np.array([2.0, -1.5]) + 0.3 * rng.standard_normal(n)).astype(float)

    assert use_mi_miller_madow() is False, "thread-local must start clean"
    sel = MRMR(n_workers=1, verbose=0, fe_max_steps=0, max_runtime_mins=2, mi_correction="miller_madow")
    sel.fit(X, y)

    assert True in seen, "fit() must ACTIVATE the MM thread-local when mi_correction='miller_madow' (dead-knob regression)"
    assert use_mi_miller_madow() is False, "fit() must reset the MM thread-local in finally"
    chosen = {int(c[1:]) for c in sel.transform(X).columns if c.startswith("f") and c[1:].isdigit()}
    assert chosen & {0, 1}, "true binary drivers should survive selection"


def test_mi_correction_invalid_value_rejected():
    """Validation guard: an unknown mi_correction string raises (defensive contract)."""
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] > 0).astype(int).to_numpy()
    sel = MRMR(n_workers=1, verbose=0, fe_max_steps=0, mi_correction="not_a_real_correction")
    with pytest.raises((ValueError, Exception)):
        sel.fit(X, y)
