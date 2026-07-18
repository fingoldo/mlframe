"""biz_value + activation tests for ``MRMR(mi_correction='chao_shen')``.

Companion to ``test_biz_val_filters_mrmr_mi_correction.py``'s Miller-Madow tests. Pins that Chao-Shen
is now genuinely wired into fit() (05_concurrency_and_statistics.md finding #7) rather than silently
degrading to plug-in MI with a warning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def test_biz_val_chao_shen_kernel_removes_high_cardinality_bias():
    """Floor: Chao-Shen removes spurious MI from a sparse high-cardinality noise feature vs binary y."""
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes, compute_mi_cs_from_classes, merge_vars

    rng = np.random.default_rng(0)
    n = 300
    x = rng.integers(0, 60, n)  # high-cardinality pure noise, sparse joint at this n
    y = rng.integers(0, 2, n)
    fd = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([60, 2])
    cx, fx, _ = merge_vars(factors_data=fd, vars_indices=np.array([0]), var_is_nominal=None, factors_nbins=nbins)
    cy, fy, _ = merge_vars(factors_data=fd, vars_indices=np.array([1]), var_is_nominal=None, factors_nbins=nbins)
    plugin = compute_mi_from_classes(cx, fx, cy, fy)
    cs = compute_mi_cs_from_classes(cx, fx, cy, fy)
    assert cs <= plugin, f"Chao-Shen must not INCREASE relevance vs plug-in on pure noise; plugin={plugin:.4f}, cs={cs:.4f}"
    assert cs < 0.02, f"Chao-Shen should nearly-zero a sparse high-card noise pair; got {cs:.4f}"


def test_biz_val_chao_shen_noop_on_lowcard_signal():
    """Chao-Shen must NOT damage a genuine low-cardinality (binary) signal."""
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes, compute_mi_cs_from_classes, merge_vars

    rng = np.random.default_rng(1)
    n = 400
    x = rng.integers(0, 2, n)
    y = (x ^ (rng.random(n) < 0.1).astype(int)).astype(np.int32)
    fd = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array([2, 2])
    cx, fx, _ = merge_vars(factors_data=fd, vars_indices=np.array([0]), var_is_nominal=None, factors_nbins=nbins)
    cy, fy, _ = merge_vars(factors_data=fd, vars_indices=np.array([1]), var_is_nominal=None, factors_nbins=nbins)
    plugin = compute_mi_from_classes(cx, fx, cy, fy)
    cs = compute_mi_cs_from_classes(cx, fx, cy, fy)
    assert plugin > 0.1, "binary signal should have real MI"
    assert cs >= 0.9 * plugin, f"Chao-Shen must not damage low-card signal: cs={cs:.4f} vs plugin={plugin:.4f}"


def test_mi_correction_chao_shen_activates_and_resets_thread_local(monkeypatch):
    """Pin that fit() ACTIVATES the Chao-Shen thread-local mid-fit and the finally resets it. Mirrors
    test_mi_correction_knob_activates_and_resets_thread_local for Miller-Madow (finding #7)."""
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.filters.mrmr import _mrmr_class as mc
    from mlframe.feature_selection.filters.info_theory import use_mi_chao_shen

    seen = []
    real_set = mc.set_mi_chao_shen

    def _spy(active):
        """Record every activation call, then delegate to the real setter."""
        seen.append(bool(active))
        return real_set(active)

    monkeypatch.setattr(mc, "set_mi_chao_shen", _spy)

    rng = np.random.default_rng(7)
    n = 500
    xt = rng.integers(0, 2, (n, 2)).astype(float)
    noise = rng.standard_normal((n, 3))
    X = pd.DataFrame(np.column_stack([xt, noise]), columns=[f"f{i}" for i in range(5)])
    y = (xt @ np.array([2.0, -1.5]) + 0.3 * rng.standard_normal(n)).astype(float)

    assert use_mi_chao_shen() is False, "thread-local must start clean"
    sel = MRMR(n_workers=1, verbose=0, fe_max_steps=0, max_runtime_mins=2, mi_correction="chao_shen")
    sel.fit(X, y)

    assert True in seen, "fit() must ACTIVATE the Chao-Shen thread-local when mi_correction='chao_shen'"
    assert use_mi_chao_shen() is False, "fit() must reset the Chao-Shen thread-local in finally"
    chosen = {int(c[1:]) for c in sel.transform(X).columns if c.startswith("f") and c[1:].isdigit()}
    assert chosen & {0, 1}, "true binary drivers should survive selection"


def test_mi_correction_chao_shen_no_longer_warns():
    """Regression: mi_correction='chao_shen' must NOT emit the old "not yet wired" UserWarning now
    that it is genuinely wired (finding #7)."""
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] > 0).astype(int).to_numpy()
    sel = MRMR(n_workers=1, verbose=0, fe_max_steps=0, mi_correction="chao_shen")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sel.fit(X, y)
    stale = [w for w in rec if "not yet wired" in str(w.message)]
    assert not stale, f"chao_shen must no longer emit the stale not-wired warning; got {[str(w.message) for w in stale]}"
