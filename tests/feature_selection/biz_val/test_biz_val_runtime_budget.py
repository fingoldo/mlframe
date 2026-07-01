"""biz_value: wall-clock budget-respect + scaling-envelope contract for MRMR and RFECV.

Two guarantees that no existing test pins for the filter / wrapper selectors
(BorutaShap / ShapProxiedFS already have their own budget test at
``test_boruta_shapproxied_budget_stop.py`` -- this file deliberately does NOT
duplicate that):

(a) BUDGET-RESPECT -- a deliberately oversized fit handed a tiny ``max_runtime_mins``
    must abort within a small multiple of that budget AND still expose a USABLE
    partial selection (``support_`` non-empty, ``transform`` round-trips). A selector
    that ignored its own budget would run to completion (tens of seconds to minutes)
    and trip the ``elapsed <= 4 * budget`` ceiling.

(b) SCALING ENVELOPE -- a complexity-class sensor (NOT a micro-perf gate). Simple-mode
    MRMR wall-time at 4x the rows must stay within 8x (linear+ in n has headroom 2x
    over the theoretical 4x), and at 4x the features within 25x (the candidate search
    is ~O(p^2), theoretical ~16x, headroom ~1.6x). Only a genuine complexity REGRESSION
    (e.g. an accidental O(n^2) or O(p^3) path) trips these; routine constant-factor
    drift does not.

Measured floors (run once on the dev box, asserts set generously above):
- (a) MRMR full n=4000 p=120 budget=3s: elapsed 5.5-8.5s across seeds (<= 12s = 4x).
- (a) RFECV LR n=2000 p=60 budget=3s: elapsed ~3.0s (<= 12s = 4x).
- (b) simple-mode ratio_n ~2.06 (bound 8), ratio_p ~2.31 (bound 25).

Wall-clock asserts are unreliable under ``-n`` xdist contention (a worker can be
starved for seconds), so the timing assertions are SKIPPED when running under an
xdist worker (``running_under_xdist()``); the structural USABILITY asserts
(``support_`` / ``transform``) always run.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from tests.conftest import running_under_xdist, perf_time_budget
from tests.feature_selection.conftest import is_fast_mode


# 0.05 min == 3 s. Small enough that an oversized full-mode fit must abort early,
# large enough that the per-candidate budget-check granularity can react.
_BUDGET_MINS = 0.05
_BUDGET_SECS = _BUDGET_MINS * 60.0
# 4x is generous on purpose: it absorbs (1) the in-flight candidate-eval batch that
# completes after the deadline is crossed, and (2) one-shot numba JIT on a cold path.
_BUDGET_SLACK = 4.0



pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)

def _signal_noise_df(n: int, p: int, seed: int = 42):
    """``p``-column frame: 3 informative + (p-3) pure-noise; linear binary target.
    Returns ``(DataFrame, Series)`` with ``x0..x{p-1}`` column names."""
    rng = np.random.default_rng(seed)
    x_sig = rng.normal(size=(n, 3))
    x_noise = rng.normal(size=(n, p - 3))
    X = np.column_stack([x_sig, x_noise])
    y = (x_sig.sum(axis=1) + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    cols = [f"x{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def _make_mrmr(*, simple: bool, budget=None, seed: int = 0):
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(
        min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False,
        full_npermutations=3, random_seed=seed, min_features_fallback=1, verbose=False,
        use_simple_mode=simple, max_runtime_mins=budget, n_workers=1, use_gpu=False,
    )


def _n_selected(sel) -> int:
    supp = np.asarray(sel.support_)
    return int(supp.sum()) if supp.dtype == bool else int(supp.size)


@pytest.fixture(scope="module")
def _warm_numba():
    """Pay the one-time numba JIT (and the simple/full code paths) on a tiny fit so the
    timed fits below measure steady-state work, not compile time."""
    df, y = _signal_noise_df(400, 12, seed=0)
    _make_mrmr(simple=True).fit(df, y)
    _make_mrmr(simple=False).fit(df, y)
    return True


# ---------------------------------------------------------------------------
# (a) budget-respect: MRMR
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_biz_val_mrmr_respects_runtime_budget_and_yields_usable_partial(_warm_numba):
    """MRMR full mode on a deliberately oversized problem (n=4000, p=120) handed a
    3s budget must abort within 4x the budget AND leave a usable partial selection.

    If MRMR ignored ``max_runtime_mins`` the full-mode fit would run to completion
    (much longer than 12s on this shape), so this is a genuine behavioural sensor
    for the budget guard, not a micro-perf gate. Measured 5.5-8.5s across seeds.
    """
    df, y = _signal_noise_df(4000, 120, seed=42)
    sel = _make_mrmr(simple=False, budget=_BUDGET_MINS, seed=0)

    t0 = time.perf_counter()
    sel.fit(df, y)
    elapsed = time.perf_counter() - t0

    # USABILITY (always asserted -- independent of wall-clock contention).
    assert _n_selected(sel) >= 1, "partial fit must expose a non-empty support_"
    Xt = sel.transform(df)
    assert Xt.shape[0] == df.shape[0], "transform must round-trip the same row count"
    assert Xt.shape[1] == _n_selected(sel), "transform width must match support_ size"

    # BUDGET-RESPECT (skipped under xdist contention; relaxed factor otherwise).
    if running_under_xdist():
        pytest.skip("wall-clock budget assert unreliable under xdist contention")
    ceiling = perf_time_budget(_BUDGET_SLACK * _BUDGET_SECS)
    assert elapsed <= ceiling, (
        f"MRMR ignored max_runtime_mins: elapsed {elapsed:.2f}s > {ceiling:.1f}s "
        f"(budget {_BUDGET_SECS:.0f}s, slack {_BUDGET_SLACK}x). PROD BUG if persistent."
    )


# ---------------------------------------------------------------------------
# (a) budget-respect: RFECV
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_biz_val_rfecv_respects_runtime_budget_and_yields_usable_partial(_warm_numba):
    """RFECV with a cheap LogisticRegression refit on an oversized search universe
    (n=2000, p=60) handed a 3s ``max_runtime_mins`` must abort within 4x the budget
    AND finalize a usable ``support_`` / ``transform``.

    A cheap per-refit estimator keeps the outer-loop budget-check granularity fine
    (each refit is ~ms), so the abort lands close to the deadline. Measured ~3.0s.
    The wrapper still finalizes ``select_optimal_nfeatures_`` after the time-out, so
    the partial result is a real, transformable selection.
    """
    from sklearn.linear_model import LogisticRegression
    from mlframe.feature_selection.wrappers import RFECV

    df, y = _signal_noise_df(2000, 60, seed=42)
    sel = RFECV(
        estimator=LogisticRegression(max_iter=200, random_state=0), cv=3,
        max_runtime_mins=_BUDGET_MINS, random_state=0, leakage_corr_threshold=None,
        n_features_selection_rule="argmax", max_noimproving_iters=10_000,
        max_refits=10_000, verbose=0,
    )

    t0 = time.perf_counter()
    sel.fit(df, y)
    elapsed = time.perf_counter() - t0

    supp = np.asarray(sel.support_)
    nsel = int(supp.sum()) if supp.dtype == bool else int(supp.size)
    assert nsel >= 1, "RFECV partial fit must expose a non-empty support_"
    Xt = sel.transform(df)
    assert Xt.shape[0] == df.shape[0], "transform must round-trip the same row count"
    assert Xt.shape[1] == nsel, "transform width must match support_ size"

    if running_under_xdist():
        pytest.skip("wall-clock budget assert unreliable under xdist contention")
    ceiling = perf_time_budget(_BUDGET_SLACK * _BUDGET_SECS)
    assert elapsed <= ceiling, (
        f"RFECV ignored max_runtime_mins: elapsed {elapsed:.2f}s > {ceiling:.1f}s "
        f"(budget {_BUDGET_SECS:.0f}s, slack {_BUDGET_SLACK}x). PROD BUG if persistent."
    )


# ---------------------------------------------------------------------------
# (b) scaling envelope: simple-mode MRMR complexity-class sensor
# ---------------------------------------------------------------------------


def _time_simple_fit(n: int, p: int, seed: int = 42) -> float:
    df, y = _signal_noise_df(n, p, seed=seed)
    sel = _make_mrmr(simple=True, seed=0)
    t0 = time.perf_counter()
    sel.fit(df, y)
    return time.perf_counter() - t0


@pytest.mark.slow
def test_biz_val_mrmr_simple_mode_row_scaling_envelope(_warm_numba):
    """Simple-mode MRMR wall-time growing the ROW count 4x (n=2000 -> n=8000, p=40)
    must stay within 8x -- a complexity-class envelope, not a micro-perf gate.

    The per-candidate MI work is ~linear in n (binning + plug-in MI over n rows), so
    the theoretical ratio is ~4x; the 8x bound (2x headroom) trips only on a real
    complexity regression (e.g. an accidental O(n^2) gather / sort). Measured ~2.06x.
    """
    t_n = _time_simple_fit(2000, 40, seed=42)
    t_4n = _time_simple_fit(8000, 40, seed=42)

    if running_under_xdist():
        pytest.skip("wall-clock scaling ratio unreliable under xdist contention")
    # Guard against a degenerate sub-ms baseline that would make the ratio meaningless.
    assert t_n > 0.05, f"baseline fit too fast to measure a meaningful ratio: {t_n:.4f}s"
    ratio = t_4n / t_n
    assert ratio <= 8.0, (
        f"simple-mode MRMR row-scaling regressed: t(4n)/t(n)={ratio:.2f} > 8.0 "
        f"(t(n=2000)={t_n:.3f}s, t(n=8000)={t_4n:.3f}s). Suspect an O(n^2) path."
    )


@pytest.mark.slow
def test_biz_val_mrmr_simple_mode_feature_scaling_envelope(_warm_numba):
    """Simple-mode MRMR wall-time growing the FEATURE count 4x (p=40 -> p=160, n=2000)
    must stay within 25x -- the candidate search is ~O(p^2), theoretical ~16x, so the
    25x bound (~1.6x headroom) trips only on a real complexity regression (e.g. an
    accidental O(p^3) pairwise-redundancy blow-up). Measured ~2.31x.
    """
    t_p = _time_simple_fit(2000, 40, seed=42)
    t_4p = _time_simple_fit(2000, 160, seed=42)

    if running_under_xdist():
        pytest.skip("wall-clock scaling ratio unreliable under xdist contention")
    assert t_p > 0.05, f"baseline fit too fast to measure a meaningful ratio: {t_p:.4f}s"
    ratio = t_4p / t_p
    assert ratio <= 25.0, (
        f"simple-mode MRMR feature-scaling regressed: t(4p)/t(p)={ratio:.2f} > 25.0 "
        f"(t(p=40)={t_p:.3f}s, t(p=160)={t_4p:.3f}s). Suspect an O(p^3) path."
    )


# ---------------------------------------------------------------------------
# Fast representative (MLFRAME_FAST=1): one small, structural-only path so the
# slow timing tests above still have a green smoke under fast iteration.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_fast_mode(), reason="fast representative; runs only under MLFRAME_FAST=1")
def test_biz_val_runtime_budget_fast_representative():
    """Small structural smoke of the budget contract under MLFRAME_FAST=1: a tiny
    oversized-relative-to-budget MRMR full-mode fit must still produce a usable
    partial selection. No wall-clock assertion (timing is meaningless at this size);
    this exercises the budget code path so the @slow tests' import + API stay green
    in fast mode."""
    df, y = _signal_noise_df(800, 20, seed=42)
    sel = _make_mrmr(simple=False, budget=_BUDGET_MINS, seed=0)
    sel.fit(df, y)
    assert _n_selected(sel) >= 1, "fast-rep partial fit must expose a non-empty support_"
    Xt = sel.transform(df)
    assert Xt.shape[0] == df.shape[0]
    assert Xt.shape[1] == _n_selected(sel)
