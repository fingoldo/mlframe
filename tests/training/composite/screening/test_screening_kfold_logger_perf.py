"""Regression + perf-sentinel tests for two cheap screening wins in
``discovery/_screening_tiny.py``:

1. ``_cached_kfold_splits`` -- hoist/cache the shuffled-KFold fold indices once
   per ``(n_rows, cv_folds, random_state)`` so the N_SPECS sweep stops rebuilding
   identical splits (and skips KFold.split's per-call x re-validation).
2. ``_silence_tiny_model_output(family)`` -- skip the lightgbm-logger ``setLevel``
   bump (which fires ``logging.Manager._clear_cache`` over the whole logger tree
   twice/fold) for non-lgb families.

Bit-identity is required for (1): cached splits MUST equal a fresh
``KFold(...).split(x)``, and the y-scale RMSE MUST be unchanged across repeated
specs sharing the same (n, cv_folds, seed).
"""
from __future__ import annotations

import logging
import re
import time
import warnings

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.training.composite import transforms as T
from mlframe.training.composite.discovery import _screening_tiny as S
from mlframe.training.composite.discovery._screening_tiny import (
    _cached_kfold_splits,
    _family_uses_lgb,
    _silence_tiny_model_output,
    _tiny_cv_rmse_y_scale,
)


# --------------------------------------------------------------------------- #
# Lead 1: KFold split cache -- bit-identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_rows,cv_folds,seed", [(200, 3, 42), (137, 5, 7), (1000, 4, 0)])
def test_cached_splits_equal_fresh_kfold(n_rows, cv_folds, seed):
    """Cached fold indices are bit-identical to a fresh KFold.split, and
    independent of the feature VALUES (split is a pure function of n_rows)."""
    S._KFOLD_SPLIT_CACHE.clear()
    cached = _cached_kfold_splits(n_rows, cv_folds, seed)
    # Fresh KFold on a DIFFERENT array of the same length, different dtype/values.
    x = np.random.default_rng(123).normal(size=(n_rows, 6)) + 50.0
    fresh = list(KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(x))
    assert len(cached) == len(fresh) == cv_folds
    for (c_tr, c_va), (f_tr, f_va) in zip(cached, fresh):
        assert np.array_equal(c_tr, f_tr)
        assert np.array_equal(c_va, f_va)


def test_cache_hits_return_same_object():
    """Second lookup of the same key reuses the cached list (no rebuild)."""
    S._KFOLD_SPLIT_CACHE.clear()
    a = _cached_kfold_splits(300, 3, 11)
    b = _cached_kfold_splits(300, 3, 11)
    assert a is b


def test_cache_bounded_reset():
    """Cache is bounded: it never grows past the cap (cheap reset on overflow)."""
    S._KFOLD_SPLIT_CACHE.clear()
    for i in range(S._KFOLD_SPLIT_CACHE_MAX + 5):
        _cached_kfold_splits(50 + i, 2, 0)
    assert len(S._KFOLD_SPLIT_CACHE) <= S._KFOLD_SPLIT_CACHE_MAX


def _run_spec(y, base, X, seed, family="linear"):
    tr = T.get_transform("additive_residual")
    fp = tr.fit(y, base)
    fp = fp if isinstance(fp, dict) else {}
    return _tiny_cv_rmse_y_scale(
        y, base, tr, fp, X,
        family=family, n_estimators=10, num_leaves=7,
        learning_rate=0.1, cv_folds=3, random_state=seed,
    )


def test_y_scale_rmse_bit_identical_across_repeated_specs():
    """Repeated specs sharing (n, cv_folds, seed) -> identical RMSE whether the
    split cache is cold or warm. Caching must not perturb the numeric result."""
    rng = np.random.default_rng(0)
    n = 300
    base = rng.normal(size=n)
    y = 2 * base + rng.normal(scale=0.3, size=n)

    # First spec (cold cache).
    S._KFOLD_SPLIT_CACHE.clear()
    X1 = rng.normal(size=(n, 4)); X1[:, 0] = base
    r_cold = _run_spec(y, base, X1, seed=42)

    # Many further specs with DIFFERENT feature matrices but same (n, folds, seed):
    # all must reuse the same splits and stay bit-identical given the same X.
    rmses = []
    for _ in range(5):
        Xk = rng.normal(size=(n, 4)); Xk[:, 0] = base
        rmses.append(_run_spec(y, base, Xk, seed=42))

    # Re-run the very first spec again (warm cache) -> must equal the cold result.
    r_warm = _run_spec(y, base, X1, seed=42)
    assert r_warm == r_cold, "warm-cache RMSE diverged from cold-cache RMSE"
    assert all(np.isfinite(r) for r in rmses)


# --------------------------------------------------------------------------- #
# Lead 2: lightgbm-logger gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "family,expected", [(None, True), ("lgb", True), ("lightgbm", True),
                        ("LightGBM", True), ("linear", False), ("ridge", False),
                        ("cb", False), ("xgb", False)],
)
def test_family_uses_lgb_gate(family, expected):
    assert _family_uses_lgb(family) is expected


def test_silence_skips_lgb_logger_for_linear():
    """Non-lgb family must NOT touch the lightgbm logger level."""
    lgb_logger = logging.getLogger("lightgbm")
    lgb_logger.setLevel(logging.DEBUG)
    with _silence_tiny_model_output("linear"):
        assert lgb_logger.level == logging.DEBUG  # untouched
    assert lgb_logger.level == logging.DEBUG


def test_silence_still_bumps_lgb_logger_for_lgb():
    """When family IS lgb the logger is bumped to ERROR inside the context and
    restored on exit -- the silencing contract must survive the gate."""
    lgb_logger = logging.getLogger("lightgbm")
    lgb_logger.setLevel(logging.DEBUG)
    inside = {}
    with _silence_tiny_model_output("lightgbm"):
        inside["level"] = lgb_logger.level
    assert inside["level"] == logging.ERROR
    assert lgb_logger.level == logging.DEBUG  # restored


def test_silence_default_none_still_bumps():
    """Legacy callers (no family) keep the always-bump behaviour."""
    lgb_logger = logging.getLogger("lightgbm")
    lgb_logger.setLevel(logging.DEBUG)
    with _silence_tiny_model_output():
        assert lgb_logger.level == logging.ERROR
    assert lgb_logger.level == logging.DEBUG


def test_silence_reentrant_only_outermost_touches_level(monkeypatch):
    """Nested silence contexts (outer per-CV-call wrap + inner per-fold) must call
    ``setLevel`` exactly twice total (one bump on outer enter, one restore on outer
    exit), NOT 2x per nesting level. Each setLevel fires Manager._clear_cache over
    the whole tree, so collapsing N inner calls into the outer is the perf win."""
    lgb_logger = logging.getLogger("lightgbm")
    lgb_logger.setLevel(logging.DEBUG)
    calls = {"n": 0}
    real_setlevel = type(lgb_logger).setLevel

    def _counting_setlevel(self, level):
        if self is lgb_logger:
            calls["n"] += 1
        return real_setlevel(self, level)

    monkeypatch.setattr(type(lgb_logger), "setLevel", _counting_setlevel)
    with _silence_tiny_model_output("lgb"):
        assert lgb_logger.level == logging.ERROR
        for _ in range(5):  # emulate 5 per-fold inner silences
            with _silence_tiny_model_output("lgb"):
                assert lgb_logger.level == logging.ERROR  # still silenced
    assert lgb_logger.level == logging.DEBUG  # restored by outermost only
    assert calls["n"] == 2, f"expected 2 setLevel calls (bump+restore), got {calls['n']}"


def test_silence_reentrant_restores_after_inner_exits():
    """The reentrancy depth must not leak: after a full nested enter/exit cycle the
    depth returns to 0 so the next top-level call bumps again (and restores)."""
    lgb_logger = logging.getLogger("lightgbm")
    lgb_logger.setLevel(logging.WARNING)
    with _silence_tiny_model_output("lgb"):
        with _silence_tiny_model_output("lgb"):
            pass
    assert lgb_logger.level == logging.WARNING
    # A fresh top-level call must work identically (depth was reset to 0).
    with _silence_tiny_model_output("lgb"):
        assert lgb_logger.level == logging.ERROR
    assert lgb_logger.level == logging.WARNING


# --------------------------------------------------------------------------- #
# Perf sentinels (wall, generous floors to avoid CI flakiness)
# --------------------------------------------------------------------------- #
def test_perf_silence_linear_cheaper_than_lgb_bump():
    """Entering/exiting the silencer for a non-lgb family must be cheaper than
    the lgb path (which fires logging.Manager._clear_cache twice). Generous
    floor: the linear path takes < 70% of the lgb path over many iters."""
    iters = 4000

    t0 = time.perf_counter()
    for _ in range(iters):
        with _silence_tiny_model_output("lightgbm"):
            pass
    t_lgb = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        with _silence_tiny_model_output("linear"):
            pass
    t_linear = time.perf_counter() - t0

    assert t_linear < 0.7 * t_lgb, (
        f"linear-silence not cheaper: linear={t_linear*1e3:.2f}ms "
        f"lgb={t_lgb*1e3:.2f}ms"
    )


def test_perf_cached_splits_faster_than_fresh():
    """Warm split-cache lookup must be far cheaper than rebuilding the KFold
    split each call (the per-sweep win). Floor: cache < 30% of fresh build."""
    n_rows, cv_folds, seed = 5000, 4, 3
    x = np.random.default_rng(1).normal(size=(n_rows, 8))
    iters = 2000

    S._KFOLD_SPLIT_CACHE.clear()
    _cached_kfold_splits(n_rows, cv_folds, seed)  # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        _cached_kfold_splits(n_rows, cv_folds, seed)
    t_cache = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        list(KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(x))
    t_fresh = time.perf_counter() - t0

    assert t_cache < 0.3 * t_fresh, (
        f"cache not faster: cache={t_cache*1e3:.2f}ms fresh={t_fresh*1e3:.2f}ms"
    )


def test_silence_uses_precompiled_message_regexes_with_identical_semantics():
    """``_silence_tiny_model_output`` installs four ignore filters using precompiled message regexes (``_FEATURE_NAMES_RE`` / ``_SKIPPING_FEATURES_RE``) rather than re-compiling per fold.

    Pins both halves: (a) the four filter tuples inside the context exactly mirror what four ``warnings.filterwarnings("ignore", ...)`` calls produce (action/category/case-insensitive message), and (b) the matching warnings are suppressed inside and the prior filter state is restored on exit.
    """
    from sklearn.exceptions import ConvergenceWarning

    # (a) shape + reuse of the module-level compiled regexes (no per-call recompile).
    with _silence_tiny_model_output("ridge"):
        top4 = list(warnings.filters[:4])
    assert top4 == [
        ("ignore", None, RuntimeWarning, None, 0),
        ("ignore", None, ConvergenceWarning, None, 0),
        ("ignore", S._SKIPPING_FEATURES_RE, UserWarning, None, 0),
        ("ignore", S._FEATURE_NAMES_RE, UserWarning, None, 0),
    ], top4
    assert S._FEATURE_NAMES_RE.flags & re.IGNORECASE
    assert S._SKIPPING_FEATURES_RE.flags & re.IGNORECASE

    # (b) suppression inside + restoration outside (outer error-filter must bite again after exit).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with _silence_tiny_model_output("lgb"):
            warnings.warn("X has feature names, but X was fitted without", UserWarning)
            warnings.warn("Skipping features without any observed values: [0]", UserWarning)
            warnings.warn("conv", ConvergenceWarning)
            warnings.warn("rt", RuntimeWarning)
        raised = False
        try:
            warnings.warn("unrelated user warning", UserWarning)
        except UserWarning:
            raised = True
        assert raised, "outer error filter must be restored after the silence context exits"
